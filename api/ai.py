import os
import json
from datetime import date
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from openai import AsyncOpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()                      # looks for .env in cwd
    load_dotenv('../secrets.env')      # fallback for local dev
except Exception:
    pass

router = APIRouter(prefix="/ai", tags=["ai"])

MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")

Language = Literal["en", "es", "pt"]

LANG_NAMES: Dict[str, str] = {
    "en": "English",
    "es": "Latin American Spanish",
    "pt": "Brazilian Portuguese",
}

CAT_SYSTEM = (
    "You are 墨 (Mò), a witty ink-black cat who reads Chinese articles alongside language learners. "
    "You are literary, a little sassy, and deeply knowledgeable about Chinese culture and language. "
    "You use cat mannerisms (* purrs *, * flicks tail *, * tilts head *) very sparingly — once at most per message. "
    "Keep all responses short (2–3 sentences max)."
)


def _client() -> AsyncOpenAI:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")
    return AsyncOpenAI(api_key=key)


async def _chat_json(prompt: str, max_tokens: int = 1000) -> Dict[str, Any]:
    resp = await _client().chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        max_completion_tokens=max_tokens,
        reasoning_effort="low"
    )
    # print("DEBUG: _chat_json response:", resp)
    return json.loads(resp.choices[0].message.content or "{}")


# ─── Cat Companion Endpoints ──────────────────────────────────────────────────

class CatReactIn(BaseModel):
    sentence_zh: str
    sentence_translation: str
    emoji: str
    language: Language


class CatAnswerIn(BaseModel):
    sentence_zh: str
    sentence_translation: str
    question: str
    language: Language


@router.post("/cat/answer")
async def cat_answer(req: CatAnswerIn) -> Dict[str, Any]:
    prompt = f"""{CAT_SYSTEM}

Sentence just read:
Chinese: {req.sentence_zh}
Translation: {req.sentence_translation}

Reader's question: "{req.question}"

Respond with JSON: {{"message": "<your answer>"}}
Answer in {LANG_NAMES[req.language]}. Be helpful and concise (2–3 sentences). Weave in a Chinese language or culture insight if relevant."""

    data = await _chat_json(prompt, max_tokens=1000)

    return {"message": data.get("message", "Mrow... interesting question.")}


class CatGuessIn(BaseModel):
    current_translation: str
    guess: str
    next_translation: str
    language: Language


@router.post("/cat/guess")
async def cat_guess(req: CatGuessIn) -> Dict[str, Any]:
    prompt = f"""{CAT_SYSTEM}

What the reader just read: "{req.current_translation}"
Reader's guess for the next sentence: "{req.guess}"
Actual next sentence: "{req.next_translation}"

Is the guess reasonably close in meaning or direction (exact match not required)?
Respond with JSON: {{"isCorrect": true or false, "message": "<your response>"}}
Respond in {LANG_NAMES[req.language]}.
- If correct: celebrate briefly and playfully.
- If wrong: tease gently without revealing the actual content; give a tiny directional hint at most.
Keep under 30 words."""

    data = await _chat_json(prompt, max_tokens=1000)
    return {
        "isCorrect": bool(data.get("isCorrect", False)),
        "message": data.get("message", "Hmm, not quite..."),
    }


# ─── Comprehension Test ───────────────────────────────────────────────────────

class GenerateQuestionsIn(BaseModel):
    full_text: str
    level: str


@router.post("/questions/generate")
async def generate_questions(req: GenerateQuestionsIn) -> List[Dict[str, Any]]:
    prompt = f"""Article Level: {req.level}
Article Text (Simplified Chinese): {req.full_text}

Generate exactly 3 reading comprehension questions.
- Questions must be in Simplified Chinese.
- Difficulty must match the article level.
- Provide translations in English (en), Spanish (es), and Portuguese (pt).
- Provide a concise model answer in Chinese.

Respond with JSON:
{{
  "questions": [
    {{
      "id": "q1",
      "question": "<Chinese question>",
      "translation": {{"en": "...", "es": "...", "pt": "..."}},
      "answer": "<Chinese model answer>"
    }},
    {{"id": "q2", ...}},
    {{"id": "q3", ...}}
  ]
}}"""

    data = await _chat_json(prompt, max_tokens=1000)
    return data.get("questions", [])


class EvaluateAnswerIn(BaseModel):
    question: str
    user_answer: str
    article_context: str
    language: Language
    attempt: int = 1
    level: str


@router.post("/answer/evaluate")
async def evaluate_answer(req: EvaluateAnswerIn) -> Dict[str, Any]:
    prompt = f"""Context: {req.article_context}
Question (Chinese): {req.question}
User Answer: {req.user_answer}
Feedback Language: {req.language}
Attempt: {req.attempt} (1 = first try, 2 = second try)
Article Level: {req.level}

Evaluate if the user's answer is correct based on the context.

Constraints:
- For Intermediate or Advanced articles, the answer MUST be in Chinese characters. If it's in Latin script or Pinyin, mark isCorrect false and say "Please answer in Chinese characters." in {req.language}. Arabic numerals are OK.

Scoring:
- Correct: isCorrect true, brief praise + explanation in {req.language}.
- Incorrect attempt 1: helpful hint pointing to the article without giving the answer away.
- Incorrect attempt 2: stronger hint or beginning of the answer.
Be kind and teacher-like.

Respond with JSON: {{"isCorrect": true or false, "feedback": "...", "chineseFeedback": "..."}}"""

    data = await _chat_json(prompt, max_tokens=1000)
    return {
        "isCorrect": bool(data.get("isCorrect", False)),
        "feedback": data.get("feedback", ""),
        "chineseFeedback": data.get("chineseFeedback"),
    }
