import logging
import os
import json
import time
from typing import List, Dict, Any, Literal
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from openai import OpenAI
from scraper import scrape
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

Level = Literal["beginner", "intermediate", "advanced"]

MODEL_PICK = os.getenv("OPENAI_MODEL_PICK", "gpt-5-mini")
MODEL_REWRITE = os.getenv("OPENAI_MODEL_REWRITE", "gpt-5-mini")

client = OpenAI()  # uses OPENAI_API_KEY env var by default


# ---------------------------
# Helpers
# ---------------------------
def _call_json_schema(prompt_messages: List[Dict[str, str]], schema_name: str, schema: Dict[str, Any],
                      model: str,retries: int = 5) -> Dict[str, Any]:
    """
    Calls Responses API and returns parsed JSON.
    Retries on transient errors.
    """
    last_err = None
    for attempt in range(retries):
        try:
            resp = client.responses.create(
                model=model,
                input=prompt_messages,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": schema_name,
                        "strict": True,
                        "schema": schema,
                    }
                },
            )
            # print(resp)
            return json.loads(resp.output_text)
        except Exception as e:
            last_err = e
            # basic backoff
            sleep_s = min(2 ** attempt, 20)
            time.sleep(sleep_s)
    raise RuntimeError(f"OpenAI call failed after {retries} retries: {last_err}")


def _level_guidelines(level: Level) -> str:
    if level == "beginner":
        return (
            "Target: beginner Chinese learners (roughly HSK1-2).\n"
            "- Use very common words, avoid jargon and idioms.\n"
            "- Prefer short sentences (8–15 chars).\n"
            "- If a technical term is necessary, explain it in plain Chinese.\n"
            "- Keep only the key facts; you may summarize.\n"
            "- Output in Simplified Chinese.\n"
            "- ≤ 150 Chinese characters.\n"
        )
    if level == "intermediate":
        return (
            "Target: intermediate learners (roughly HSK3-4).\n"
            "- Use common words if possible and explain rare terms briefly if necessary.\n"
            "- Use a mix of short and medium-length sentences (8–25 chars).\n"
            "- Output in Simplified Chinese.\n"
            "- Around 150 to 300 Chinese characters. If content is longer, summarize while preserving key facts but make sure it is natural\n"
        )
    return (
        "Target: advanced learners (roughly HSK5-6+).\n"
        "- You can keep details; you may improve style and structure.\n"
        "- Use a mix of short and medium-length sentences (8–25 chars).\n"
        "- Output in Simplified Chinese.\n"
        "- Around 300 to 600 Chinese characters. If content is longer, summarize while preserving key facts but make sure it is natural\n"
    )


# ---------------------------
# Step 1: pick 3/3/3 based on titles
# ---------------------------
def pick_articles_by_title(articles: List[Dict[str, Any]]) -> Dict[str, List[int]]:
    if len(articles) < 9:
        raise ValueError(f"Need at least 9 articles, got {len(articles)}")

    title_list = [
        {"idx": i, "source": a.get("source", ""), "title": a.get("title", "")}
        for i, a in enumerate(articles)
    ]

    schema = {
        "type": "object",
        "properties": {
            "beginner": {
                "type": "array",
                "minItems": 3,
                "maxItems": 3,
                "items": {"type": "integer"},
            },
            "intermediate": {
                "type": "array",
                "minItems": 3,
                "maxItems": 3,
                "items": {"type": "integer"},
            },
            "advanced": {
                "type": "array",
                "minItems": 3,
                "maxItems": 3,
                "items": {"type": "integer"},
            },
        },
        "required": ["beginner", "intermediate", "advanced"],
        "additionalProperties": False,
    }

    messages = [
        {
            "role": "system",
            "content": (
                "You are an editor for a Chinese reading app.\n"
                "Task: pick articles by the title provided.\n"
                "Rules:\n"
                "- The articles picked should be light and fun and ideally not related to politics.\n"
                "- Pick 3 suitable for beginner, 3 for intermediate, 3 for advanced.\n"
                "- Each index must be unique across all 9 picks.\n"
                "- Beginner: concrete daily-life topics, simple, low jargon.\n"
                "- Intermediate: general news/knowledge, moderate abstraction.\n"
                "- Advanced: abstract, policy/economy/science, heavier vocabulary.\n"
            ),
        },
        {"role": "user", "content": f"Here are candidate titles:\n{json.dumps(title_list, ensure_ascii=False)}"},
    ]

    picked = _call_json_schema(
        prompt_messages=messages,
        schema_name="picked_articles",
        schema=schema,
        model=MODEL_PICK,
    )

    # Basic validation (unique and in range)
    all_idxs = picked["beginner"] + picked["intermediate"] + picked["advanced"]
    if len(set(all_idxs)) != 9:
        raise ValueError(f"Model returned duplicate indices: {picked}")
    for i in all_idxs:
        if not (0 <= i < len(articles)):
            raise ValueError(f"Index out of range: {i}")

    return picked


# ---------------------------
# Step 2: rewrite title + content for each level
# ---------------------------
def rewrite_article(article: Dict[str, Any], level: Level) -> Dict[str, Any]:
    logging.info(f"Rewriting article {article.get('title', '')} for level {level}...")
    schema = {
        "type": "object",
        "properties": {
            "level": {"type": "string", "enum": ["beginner", "intermediate", "advanced"]},
            "title": {"type": "string"},
            "excerpt": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["level", "title", "excerpt", "content"],
        "additionalProperties": False,
    }

    original_title = article.get("title", "")
    original_content = article.get("content", "")

    messages = [
        {
            "role": "system",
            "content": (
                "You rewrite Chinese news for language learners.\n"
                "Hard rules:\n"
                "- Exclude any images, captions, ads, or any text that is not part of the article.\n"
                "- DO NOT add new facts, numbers, names, or claims that aren't in the original.\n"
                "- If something is uncertain in the original, keep it uncertain.\n"
                "- Keep the meaning faithful; you may simplify or reorganize.\n"
                "- Make the content engaging and easy to read for the target level.\n"
                "- Output ONLY the JSON that matches the schema.\n\n"
                f"{_level_guidelines(level)}"
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "original_title": original_title,
                    "original_content": original_content,
                    "target_level": level,
                },
                ensure_ascii=False,
            ),
        },
    ]

    out = _call_json_schema(
        prompt_messages=messages,
        schema_name="rewritten_article",
        schema=schema,
        model=MODEL_REWRITE,
    )

    # merge with original metadata
    return {
        **article,
        "level": out["level"],
        "original_title": original_title,
        "original_content": original_content,
        "title": out["title"],
        "content": out["content"],
        "excerpt": out["excerpt"],
    }


def build_leveled_set(articles: List[Dict[str, Any]], parallel: bool = True) -> List[Dict[str, Any]]:
    picked = pick_articles_by_title(articles)

    tasks: List[tuple[int, Level]] = []
    for lvl in ("beginner", "intermediate", "advanced"):
        for idx in picked[lvl]:
            tasks.append((idx, lvl))  # (article_idx, level)

    results: List[Dict[str, Any]] = []

    if parallel:
        with ThreadPoolExecutor(max_workers=6) as ex:
            futs = {ex.submit(rewrite_article, articles[idx], lvl): (idx, lvl) for idx, lvl in tasks}
            for fut in as_completed(futs):
                idx, lvl = futs[fut]
                results.append(fut.result())
    else:
        for idx, lvl in tasks:
            results.append(rewrite_article(articles[idx], lvl))

    # Optional: stable ordering by level then published/title
    order = {"beginner": 0, "intermediate": 1, "advanced": 2}
    results.sort(key=lambda a: (order.get(a["level"], 9), a.get("published", ""), a.get("title", "")))
    return results
   
if __name__ == "__main__":

    articles = scrape()
    leveled = build_leveled_set(articles, parallel=True)
    print(json.dumps(leveled, ensure_ascii=False, indent=2))
    # store the output
    now = datetime.now()
    with open(f"news_leveled_{now.strftime('%y_%m_%d')}.json", "w", encoding="utf-8") as f:
        json.dump(leveled, f, ensure_ascii=False, indent=4)