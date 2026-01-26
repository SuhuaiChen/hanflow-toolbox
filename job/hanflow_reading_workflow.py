from __future__ import annotations

import os
from typing import Dict
from pymongo import MongoClient
from openai import OpenAI

from cedict import load_cedict_simplified
from hsk_data import load_hsk_words
from scraper import scrape
from selector import build_leveled_set
from article_annotator import annotate_articles
from oss import r2_client
from sentence_of_day_to_mongo import upsert_sentence_of_day

try :
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

def hanflow_reading_workflow(
    *,
    db_client: MongoClient,
    lexicon,
    hsk_words: Dict[str, str],
    s3,
):
    articles = scrape()
    leveled = build_leveled_set(articles, parallel=True)

    annotated = annotate_articles(
        raw_articles=leveled,
        articles_col=db_client["core"]["articles"],
        vocab_col=db_client["core"]["dict"],
        cedict_lexicon=lexicon,
        hsk_words=hsk_words,
        s3=s3,
        mode=os.getenv("ANNOTATE_MODE", "skip"),  # "skip" default saves time/cost
    )
    return annotated

def _build_mongo_client() -> MongoClient:
    # Prefer env var in Cloud Run
    uri = os.environ["MONGODB_URI"]
    return MongoClient(uri)

def main() -> int:
    # Paths inside container (COPY data -> /app/data)
    cedict_path = os.getenv("CEDICT_PATH", "/app/data/cedict_ts.u8")
    hsk_path = os.getenv("HSK_PATH", "/app/data/words.csv")

    lexicon = load_cedict_simplified(cedict_path)
    hsk_words = load_hsk_words(hsk_path) 

    s3 = r2_client()
    mongo_client = _build_mongo_client()

    annotated_articles = hanflow_reading_workflow(
        db_client=mongo_client,
        lexicon=lexicon,
        hsk_words=hsk_words,
        s3=s3,
    )

    print(f"Annotated {len(annotated_articles)} articles.")
    for article in annotated_articles[:3]:
        print(article.get("id"), article.get("title"))

    # Sentence of the day (OpenAI key comes from env)
    upsert_sentence_of_day(
        mongo_client=mongo_client,
        openai_client=OpenAI(api_key=os.getenv("OPENAI_API_KEY")),
    )

    return 0

if __name__ == "__main__":
    raise SystemExit(main())