from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from datetime import datetime, timezone
from typing import List
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from dict_types import DictEntry, Pronunciation
from cedict import cedict_lookup_all, cedict_entries_to_text
from pinyin_convert import pinyin_num_to_tone, pinyin_tone_to_num
from translate import llm_enrich_dict_entry
from oss import attach_tts_audio


def create_mongo_client() -> MongoClient:
    client = MongoClient(os.environ["MONGODB_URI"], server_api=ServerApi("1"))
    try:
        client.admin.command("ping")
        print("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as e:
        print(e)
    return client


def enrich_dict_entry(
    hz: str,
    *,
    cedict_all: dict,
    hsk_words: dict,
    s3,
) -> DictEntry:
    """
    Run the full enrichment pipeline for a single word.

    Args:
        hz: simplified Chinese headword
        cedict_all: from load_cedict_all() — all entries per headword
        hsk_words: from load_hsk_words() — {word: level}
        s3: boto3 R2 client from r2_client()

    Returns:
        A complete DictEntry ready to upsert to core.dict.
    """
    # Step 1: CEDICT lookup (all entries)
    cedict_entries = cedict_lookup_all(hz, cedict_all)
    cedict_text = cedict_entries_to_text(cedict_entries)
    src_base = "cedict" if cedict_entries else "llm"

    # Step 2: HSK level
    hsk_level_raw = hsk_words.get(hz)
    try:
        hsk_level = int(hsk_level_raw) if hsk_level_raw is not None else None
    except (ValueError, TypeError):
        hsk_level = None

    # Step 3: Pinyin
    if cedict_entries:
        # Use first entry's pinyin (most common pronunciation)
        py_num = cedict_entries[0]["pinyin"]          # e.g. "fa1 zhan3"
        py_mark = pinyin_num_to_tone(py_num).replace(" ", "")  # e.g. "fāzhǎn"
        src_py = "cedict"
    else:
        # Fallback: use pypinyin directly to avoid importing nlp (which requires ltp)
        from pypinyin import pinyin as _pypinyin, Style as _Style
        py_mark = "".join(s[0] for s in _pypinyin(hz, style=_Style.TONE, strict=False))
        py_num = pinyin_tone_to_num(py_mark)
        src_py = "pypinyin"

    # Step 4: LLM enrichment
    llm_result = llm_enrich_dict_entry(hz, cedict_text)

    # Step 5: Build search index
    search_hz = [hz] + list(hz)           # word + individual characters
    search_py = [py_num, py_mark]
    search_kw = llm_result.get("search_kw", {"en": [], "es": [], "pt": []})

    # Step 6: TTS audio for headword
    pronunciation: Pronunciation = {"num": py_num, "mark": py_mark}
    try:
        audio_url = attach_tts_audio(text=hz, s3=s3, kind="tokens")
        pronunciation["audio"] = audio_url
    except Exception as e:
        print(f"TTS failed for '{hz}': {e}")

    # Step 7: Assemble DictEntry
    entry: DictEntry = {
        "_id": hz,
        "hz": hz,
        "py": [pronunciation],
        "pos": llm_result.get("pos", []),
        "clf": llm_result.get("clf", []),
        "lvl": {"hsk": hsk_level},
        "senses": llm_result.get("senses", []),
        "search": {
            "hz": search_hz,
            "py": search_py,
            "kw": search_kw,
        },
        "src": {"base": src_base, "py": src_py},
        "ed": {"status": "draft", "reviewed": False, "needsReview": False},
        "updatedAt": datetime.now(timezone.utc),
    }
    return entry


def ensure_dict_entry(
    word: str,
    col,
    cedict_all: dict,
    hsk_words: dict,
    s3,
) -> dict:
    """
    Ensure a DictEntry exists in the collection for `word`.

    Cache hit: doc already has `senses` field (new model) → return as-is.
    Cache miss: run full enrichment pipeline and upsert.
    """
    doc = col.find_one({"_id": word})

    if doc and "senses" in doc:
        return doc

    entry = enrich_dict_entry(word, cedict_all=cedict_all, hsk_words=hsk_words, s3=s3)
    col.replace_one({"_id": word}, entry, upsert=True)
    return entry
