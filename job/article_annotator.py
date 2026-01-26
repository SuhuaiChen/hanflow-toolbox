import logging
from typing import Dict, Any, List, Tuple, Set
from datetime import datetime, timezone
from pymongo import UpdateOne
from cedict import load_cedict_simplified
from hsk_data import load_hsk_words
from nlp import tokenize_with_pos, transform_tokens, split_to_sentence_objects
from pinyin_convert import pinyin_num_to_tone
from translate import llm_translate_text_triple
from db import ensure_dict_entry, create_mongo_client
from oss import attach_tts_audio
import hashlib

def _sha1_10(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()[:10]

def make_article_id(raw: Dict[str, Any]) -> str:
    published = (raw.get("published") or raw.get("date") or "unknown-date").strip()
    link = (raw.get("link") or "").strip()
    return f"{published}-{_sha1_10(link)}"

def _has_content(raw: Dict[str, Any]) -> bool:
    return bool((raw.get("content") or "").strip())

def enrich_tokens_with_dict(tokens: List[Dict], vocab_col, cedict_lexicon, hsk_words, s3) -> List[Dict]:
    """
    tokens: output of transform_tokens(segmented)
    returns: same tokens but meanings.en/es/pt guaranteed (for non-punct)
    """
    for tok in tokens:
        if tok.get("type") == "punct":
            tok["meanings"] = {}
            continue

        w = tok["t"]
        entry = ensure_dict_entry(w, vocab_col, cedict_lexicon, hsk_words, s3)

        tok.setdefault("meanings", {})
        # Ensure strings (cedict might give list)
        tok["meanings"]["en"] = entry["meanings"].get("en", "") or ""
        tok["meanings"]["es"] = entry["meanings"].get("es", "") or ""
        tok["meanings"]["pt"] = entry["meanings"].get("pt", "") or ""

        if entry.get("pinyin") and not tok.get("pinyin"):
            tok["pinyin"] = pinyin_num_to_tone(entry["pinyin"])

    return tokens

def annotate_sentence(
    sid: str,
    zh: str,
    vocab_col,
    cedict_lexicon,
    hsk_words: Dict[str, str],
    s3
) -> Dict[str, Any]:
    # 1) translate sentence
    triple = llm_translate_text_triple(zh)

    # 2) tokenize with POS
    segmented: List[Tuple[str, str]] = tokenize_with_pos(zh)

    # 3) transform tokens into target schema-ish tokens
    tokens = transform_tokens(segmented, cedict_lexicon, hsk_words)

    # 4) enrich meanings via Mongo cache (and fill missing)
    tokens = enrich_tokens_with_dict(tokens, vocab_col, cedict_lexicon, hsk_words, s3)

    # TTS
    audio_url = attach_tts_audio(text=zh, s3=s3)

    return {
        "sid": sid,
        "zh": zh,
        "translations": triple,
        "tokens": tokens,
        "audio": audio_url,
    }

def annotate_article(
    raw_article: Dict[str, Any],
    vocab_col,
    cedict_lexicon,
    hsk_words: Dict[str, str],
    s3
) -> Dict[str, Any]:
    """
    raw_article input structure:
      {
        "source": "...",
        "title": "...",
        "link": "...",
        "published": "YYYY-MM-DD",
        "content": "...",
        "level": "beginner",
        "excerpt": "..."
      }
    """

    article_id = make_article_id(raw_article)

    # title/excerpt translations
    title_triple = llm_translate_text_triple(raw_article["title"])
    excerpt_triple = llm_translate_text_triple(raw_article["excerpt"])

    # Audio for title and excerpt
    title_audio = attach_tts_audio(text=raw_article["title"], s3=s3)
    excerpt_audio = attach_tts_audio(text=raw_article["excerpt"], s3=s3)

    # split into sentences
    sents = split_to_sentence_objects(raw_article.get("content", ""))

    annotated_sents = [
        annotate_sentence(s["sid"], s["zh"], vocab_col, cedict_lexicon, hsk_words, s3)
        for s in sents
        if s.get("zh") and s["zh"].strip()
    ]

    return {
        "id": article_id,
        "title": raw_article["title"],
        "titleAudio": title_audio,
        "titleTranslations": title_triple,
        "excerpt": raw_article.get("excerpt") or "",
        "excerptAudio": excerpt_audio,
        "excerptTranslations": excerpt_triple,
        "level": raw_article.get("level") or "intermediate",
        "date": raw_article.get("published") or raw_article.get("date") or "unknown-date",  
        "sentences": annotated_sents,
        "meta": {
            "source": raw_article.get("source"),
            "link": raw_article.get("link"),
            "original_title": raw_article.get("original_title"),
            "original_content": raw_article.get("original_content"),
        }
    }

def fetch_existing_ids(articles_col, ids: List[str]) -> Set[str]:
    if not ids:
        return set()
    return {d["_id"] for d in articles_col.find({"_id": {"$in": ids}}, {"_id": 1})}


def bulk_upsert_articles(articles_col, docs: List[Dict[str, Any]]) -> None:
    if not docs:
        return

    now = datetime.now(timezone.utc)
    ops: List[UpdateOne] = []

    for doc in docs:
        aid = doc.get("id")
        if not aid:
            raise ValueError("Annotated article missing `id`")

        mongo_doc = dict(doc)
        mongo_doc["_id"] = aid
        mongo_doc["updatedAt"] = now

        ops.append(UpdateOne({"_id": aid}, {"$set": mongo_doc}, upsert=True))

    articles_col.bulk_write(ops, ordered=False)

def annotate_articles(
    raw_articles: List[Dict[str, Any]],
    *,
    articles_col,          # Mongo collection: core.articles
    vocab_col,
    cedict_lexicon,
    hsk_words: Dict[str, str],
    s3,
    mode: str = "skip",    # "skip" | "upsert"
) -> List[Dict[str, Any]]:
    """
    Annotate + persist articles.

    mode="skip":
      - only annotate articles that are not already stored (saves LLM/TTS cost)
      - still upserts the newly annotated ones

    mode="upsert":
      - re-annotate everything and overwrite in DB (useful after you improve your pipeline)

    Returns: the list of annotated docs that were processed in this run.
    """
    if mode not in ("skip", "upsert"):
        raise ValueError("mode must be 'skip' or 'upsert'")

    # 1) filter + compute ids
    items: List[Tuple[str, Dict[str, Any]]] = []
    for raw in raw_articles:
        if not _has_content(raw):
            continue
        aid = make_article_id(raw)
        items.append((aid, raw))

    if not items:
        return []

    # 2) dedupe within this batch (same id can appear twice)
    # keep the last occurrence (or first; choose one; last is fine)
    dedup: Dict[str, Dict[str, Any]] = {aid: raw for aid, raw in items}
    ids = list(dedup.keys())

    # 3) DB existence check (one query)
    existing_ids: Set[str] = set()
    if mode == "skip":
        existing_ids = fetch_existing_ids(articles_col, ids)

    # 4) annotate required
    annotated: List[Dict[str, Any]] = []
    for aid, raw in dedup.items():
        logging.info(f"Annotating article {aid}...")
        if mode == "skip" and aid in existing_ids:
            continue

        doc = annotate_article(
            raw_article=raw,
            vocab_col=vocab_col,
            cedict_lexicon=cedict_lexicon,
            hsk_words=hsk_words,
            s3=s3
        )

        # enforce stable id (even if annotate_article computes differently internally)
        doc["id"] = aid
        annotated.append(doc)

    # 5) persist
    bulk_upsert_articles(articles_col, annotated)
    return annotated


if __name__ == "__main__":
    lexicon = load_cedict_simplified("data/cedict_ts.u8")
    hsk_words = load_hsk_words()
    client = create_mongo_client()
    db = client["core"]
    vocab_col = db["dict"]
    test_sentence = "我喜欢学习汉语，因为它很有趣！"
    annotated = annotate_sentence(
        sid="s1",
        zh=test_sentence,
        vocab_col=vocab_col,
        cedict_lexicon=lexicon,
        hsk_words=hsk_words
    )
    print(annotated)

