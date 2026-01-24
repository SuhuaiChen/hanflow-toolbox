from typing import Dict
from cedict import load_cedict_simplified
from hsk_data import load_hsk_words
from db import client
from scraper import scrape
from selector import build_leveled_set
from article_annotator import annotate_articles
from oss import r2_client


def hanflow_reading_workflow(
        db_client,
        lexicon,
        hsk_words: Dict[str, str],
):

    articles = scrape()

    leveled = build_leveled_set(articles, parallel=True)

    s3 = r2_client()

    annotated = annotate_articles(
        raw_articles=leveled,
        articles_col=db_client["core"]["articles"],
        vocab_col=db_client["core"]["dict"],
        cedict_lexicon=lexicon,
        hsk_words=hsk_words,
        s3=s3
    )

    return annotated

if __name__ == "__main__":
    lexicon = load_cedict_simplified("data/cedict_ts.u8")
    hsk_words = load_hsk_words()
    annotated_articles = hanflow_reading_workflow(
        db_client=client,
        lexicon=lexicon,
        hsk_words=hsk_words
    )
    print(f"Annotated {len(annotated_articles)} articles.")
    for article in annotated_articles[:3]:
       print(article)