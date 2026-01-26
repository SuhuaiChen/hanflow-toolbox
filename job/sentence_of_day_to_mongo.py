import re
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Callable, Optional
from zoneinfo import ZoneInfo
from openai import OpenAI
from db import create_mongo_client
from translate import llm_translate_text_triple
from nlp import sentence_to_tokens, to_pinyin
from dotenv import load_dotenv

try:
    load_dotenv()
except Exception:
    pass

# =========================
# Config
# =========================

DB_NAME = "core"
COLLECTION_NAME = "sentence"
TIMEZONE = "America/Sao_Paulo"

MODEL = "gpt-5-mini-2025-08-07"
ATTEMPTS = 3  # generation retries


# =========================
# Prompt
# =========================

GEN_PROMPT = """You are generating the “Sentence of the Day” for a Mandarin reading app called HanFlow.

GOAL
Generate ONE high-quality Chinese sentence suitable for a landing page.
The sentence should feel calm, modern, and reflective, not like a slogan.

STRICT RULES (must follow all):
- Language: Simplified Chinese only
- Length: 12–20 Chinese characters
- No quotes, no emojis, no exclamation marks
- Grammar level: HSK2–HSK4 only
- Natural, spoken Mandarin
- Must be understandable without any context
- No metaphors that require explanation

TONE
- Gentle, grounded, emotionally warm

THEMES: choose ONE of them
- daily life
- personal growth
- aesthetics of nature
- chinese philosophy
- mindfulness and well-being

OUTPUT FORMAT (VERY IMPORTANT):
Only output the Chinese sentence.
Do not output pinyin, translation, explanation, or any extra text.
"""


# =========================
# Data model
# =========================

@dataclass(frozen=True)
class SentenceDoc:
    _id: str   # YYYY-MM-DD in TIMEZONE
    date: str  # YYYY-MM-DD
    tokens: list[dict] # [{"t": str, "pinyin": str}, ...]
    translations: dict  # {"en": str, "es": str, "pt": str}

    def to_mongo(self) -> dict:
        return asdict(self)


# =========================
# Helpers: sanitize + validate
# =========================

_CH = re.compile(r"[\u4e00-\u9fff]")
_EMOJI = re.compile(
    "[" +
    "\U0001F300-\U0001FAFF" +  # emojis/symbols
    "\U00002700-\U000027BF" +  # dingbats
    "\U00002600-\U000026FF" +  # misc symbols
    "]+",
    flags=re.UNICODE,
)
_FORBIDDEN = {'"', "“", "”", "‘", "’", "'", "!", "！", "?", "？"}
_ALLOWED_PUNCT = {"，", "。"}  # landing-page clean


def first_line(text: str) -> str:
    for ln in (text or "").splitlines():
        ln = ln.strip()
        if ln:
            return ln
    return (text or "").strip()


def cn_len(s: str) -> int:
    return len(_CH.findall(s))


def validate_sentence(
    s: str,
    *,
    min_cn: int = 12,
    max_cn: int = 20,
    max_commas: int = 1,
) -> tuple[bool, str]:
    s = (s or "").strip()
    if not s:
        return False, "empty"
    if any(ch in _FORBIDDEN for ch in s):
        return False, "forbidden punctuation/quotes"
    if _EMOJI.search(s):
        return False, "emoji"
    if s.count("，") > max_commas:
        return False, "too many commas"

    # Only allow: Chinese chars, digits (rare but ok), whitespace, and minimal punctuation
    for ch in s:
        if _CH.match(ch) or ch.isdigit() or ch.isspace() or ch in _ALLOWED_PUNCT:
            continue
        return False, "extra symbols/punctuation"

    n = cn_len(s)
    if n < min_cn or n > max_cn:
        return False, f"cn length {n} not in [{min_cn},{max_cn}]"

    # Avoid “slogan-y absolutes” (tiny heuristic)
    if any(w in s for w in ("必须", "永远", "一定")):
        return False, "too absolute / slogan-like"

    return True, ""


# =========================
# OpenAI generation
# =========================

def generate_once(
    client: OpenAI,
    *,
    model: str,
) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a precise Chinese copywriter for a language-learning app."},
            {"role": "user", "content": GEN_PROMPT},
        ],
    )
    return first_line(resp.choices[0].message.content or "")


def build_sentence_doc(
    *,
    client: OpenAI,
    timezone: str = TIMEZONE,
    model: str = MODEL,
    attempts: int = ATTEMPTS,
    # Plug your own checker if you have it:
    hsk_check: Optional[Callable[[str], bool]] = None,
) -> SentenceDoc:
    tz = ZoneInfo(timezone)
    key = datetime.now(tz).date().isoformat()

    last_text = ""
    last_reason = ""
    for _ in range(attempts):
        text = generate_once(client, model=model)
        last_text = text

        ok, reason = validate_sentence(text)
        if not ok:
            last_reason = reason
            continue

        if hsk_check and not hsk_check(text):
            last_reason = "hsk_check failed"
            continue

        # Build tokens
        tokens = sentence_to_tokens(text)
        tokens = [{
            "t": token["token"],
            "pinyin": to_pinyin(token["token"])
        } for token in tokens]

        # Build translations
        translations = llm_translate_text_triple(text)

        return SentenceDoc(
            _id=key,
            date=key,
            tokens=tokens,
            translations=translations,
        )

    raise RuntimeError(
        f"Failed after {attempts} attempts. last_reason={last_reason!r}, last_text={last_text!r}"
    )


# =========================
# MongoDB upsert
# =========================

def upsert_sentence_of_day(
    *,
    mongo_client,
    openai_client: OpenAI,
    db_name: str = DB_NAME,
    collection_name: str = COLLECTION_NAME,
) -> dict:
    col = mongo_client[db_name][collection_name]

    doc = build_sentence_doc(client=openai_client)

    col.update_one(
        {"_id": doc._id},
        {"$set": doc.to_mongo()},
        upsert=True,
    )
    return doc.to_mongo()


# =========================
# CLI entry
# =========================

if __name__ == "__main__":
    openai_client = OpenAI()
    client = create_mongo_client()
    saved = upsert_sentence_of_day(
        mongo_client=client,
        openai_client=openai_client,
    )
    print("Saved:", saved)
