# backend/job/dict_enricher.py
"""
Standalone dictionary enrichment job.

Usage:
    python job/dict_enricher.py                     # enrich all (HSK + core.dict)
    python job/dict_enricher.py --limit 50          # test with first 50 words
    python job/dict_enricher.py --word 你好          # single word
    python job/dict_enricher.py --workers 3         # control concurrency
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from cedict import load_cedict_all
from db import create_mongo_client, enrich_dict_entry
from hsk_data import load_hsk_words
from oss import r2_client


def build_word_list(
    hsk_words: dict,
    dict_col,
    *,
    word: Optional[str] = None,
    limit: Optional[int] = None,
) -> list:
    """
    Build the ordered list of words to enrich.

    Order: HSK words first (by insertion order from CSV), then any non-HSK
    words already present in core.dict.

    Args:
        hsk_words: {word: level} from load_hsk_words()
        dict_col:  MongoDB collection core.dict
        word:      if set, return only [word] and skip everything else
        limit:     truncate result to first N words

    Returns:
        Deduplicated list of simplified Chinese headwords.
    """
    if word:
        return [word]

    hsk_set = set(hsk_words.keys())
    words = list(hsk_words.keys())                    # HSK first

    for doc in dict_col.find({}, {"_id": 1}):        # non-HSK from dict
        _id = doc["_id"]
        if _id not in hsk_set:
            words.append(_id)

    if limit is not None:
        words = words[:limit]

    return words


def _enrich_one(hz: str, *, cedict_all: dict, hsk_words: dict, s3, dict_col) -> str:
    """Enrich a single word and upsert. Returns 'ok' or 'error: <msg>'."""
    try:
        entry = enrich_dict_entry(hz, cedict_all=cedict_all, hsk_words=hsk_words, s3=s3)
        dict_col.replace_one({"_id": hz}, entry, upsert=True)
        return "ok"
    except Exception as exc:
        return f"error: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Enrich core.dict with the new DictEntry model.")
    parser.add_argument("--word",    type=str, default=None, help="Enrich a single specific word")
    parser.add_argument("--limit",   type=int, default=None, help="Process only first N words (for testing)")
    parser.add_argument("--workers", type=int, default=6,    help="ThreadPoolExecutor max workers (default: 6)")
    args = parser.parse_args()

    cedict_path = os.environ.get("CEDICT_PATH", "/app/data/cedict_ts.u8")
    hsk_path    = os.environ.get("HSK_PATH",    "/app/data/words.csv")

    print("Loading CEDICT (all entries)...")
    cedict_all = load_cedict_all(cedict_path)
    print(f"  {len(cedict_all)} headwords loaded")

    print("Loading HSK words...")
    hsk_words = load_hsk_words(hsk_path)
    print(f"  {len(hsk_words)} HSK words loaded")

    print("Connecting to MongoDB...")
    mongo_client = create_mongo_client()
    dict_col = mongo_client["core"]["dict"]

    print("Initialising R2 client...")
    s3 = r2_client()

    words = build_word_list(hsk_words, dict_col, word=args.word, limit=args.limit)
    total = len(words)
    print(f"Words to enrich: {total}  (workers={args.workers})")

    ok = errors = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _enrich_one, hz,
                cedict_all=cedict_all,
                hsk_words=hsk_words,
                s3=s3,
                dict_col=dict_col,
            ): hz
            for hz in words
        }
        for i, future in enumerate(as_completed(futures), 1):
            hz     = futures[future]
            result = future.result()
            if result == "ok":
                ok += 1
            else:
                errors += 1
                print(f"  [{i}/{total}] FAIL {hz!r}: {result}")
            if i % 100 == 0 or i == total:
                print(f"  Progress: {i}/{total}  ok={ok}  errors={errors}")

    print(f"\nDone.  ok={ok}  errors={errors}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
