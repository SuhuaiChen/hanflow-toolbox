import logging
from typing import Dict, Any, List, Tuple, Set
from datetime import datetime, timezone
from pymongo import UpdateOne
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

def enrich_tokens_with_dict(tokens: List[Dict], vocab_col, cedict_all: dict, hsk_words, s3) -> List[Dict]:
    for tok in tokens:
        if tok.get("type") == "punct":
            tok["meanings"] = {}
            continue

        w = tok["t"]
        entry = ensure_dict_entry(w, vocab_col, cedict_all, hsk_words, s3)

        tok.setdefault("meanings", {})
        # Read from new DictEntry format: senses[0].defn
        senses = entry.get("senses") or []
        primary = next((s for s in senses if s.get("primary")), senses[0] if senses else {})
        defn = primary.get("defn", {})
        tok["meanings"]["en"] = defn.get("en", "")
        tok["meanings"]["es"] = defn.get("es", "")
        tok["meanings"]["pt"] = defn.get("pt", "")

        # Read pinyin from py[0].mark
        py_list = entry.get("py") or []
        if py_list and not tok.get("pinyin"):
            tok["pinyin"] = py_list[0].get("mark", "")

        # Read audio from py[0].audio
        tok["audio"] = py_list[0].get("audio", "") if py_list else ""

    return tokens

def annotate_sentence(
    sid: str,
    zh: str,
    vocab_col,
    cedict_all: dict,
    hsk_words: Dict[str, str],
    s3,
    date: str = "",
    cedict_lexicon=None,   # old simplified format, for transform_tokens
) -> Dict[str, Any]:
    triple = llm_translate_text_triple(zh)
    segmented: List[Tuple[str, str]] = tokenize_with_pos(zh)
    tokens = transform_tokens(segmented, cedict_lexicon or {}, hsk_words)
    tokens = enrich_tokens_with_dict(tokens, vocab_col, cedict_all, hsk_words, s3)
    audio_url = attach_tts_audio(text=zh, s3=s3, date=date)
    return {"sid": sid, "zh": zh, "translations": triple, "tokens": tokens, "audio": audio_url}

def annotate_article(
    raw_article: Dict[str, Any],
    vocab_col,
    cedict_all: dict,
    hsk_words: Dict[str, str],
    s3,
    cedict_lexicon=None,   # old simplified format, passed down to transform_tokens
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
    published_date = raw_article.get("published") or raw_article.get("date") or "unknown-date"

    # title/excerpt translations
    title_triple = llm_translate_text_triple(raw_article["title"])
    excerpt_triple = llm_translate_text_triple(raw_article["excerpt"])

    # Audio for title and excerpt
    title_audio = attach_tts_audio(text=raw_article["title"], s3=s3, date=published_date)
    excerpt_audio = attach_tts_audio(text=raw_article["excerpt"], s3=s3, date=published_date)

    # split into sentences
    sents = split_to_sentence_objects(raw_article.get("content", ""))

    annotated_sents = [
        annotate_sentence(s["sid"], s["zh"], vocab_col, cedict_all, hsk_words, s3, published_date, cedict_lexicon=cedict_lexicon)
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
    cedict_all: dict,
    cedict_lexicon=None,
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
            cedict_all=cedict_all,
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
    from cedict import load_cedict_simplified, load_cedict_all
    lexicon = load_cedict_simplified("data/cedict_ts.u8")
    cedict_all_data = load_cedict_all("data/cedict_ts.u8")
    hsk_words = load_hsk_words()
    client = create_mongo_client()
    db = client["core"]
    vocab_col = db["dict"]
    test_sentence = "我喜欢学习汉语，因为它很有趣！"
    annotated = annotate_sentence(
        sid="s1",
        zh=test_sentence,
        vocab_col=vocab_col,
        cedict_all=cedict_all_data,
        hsk_words=hsk_words,
        cedict_lexicon=lexicon,
    )
    print(annotated)
