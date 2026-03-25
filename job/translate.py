from functools import lru_cache
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
from dotenv import load_dotenv
from openai import OpenAI
from openai import APIError, APITimeoutError, RateLimitError

try:
    load_dotenv()
except Exception:
    pass

@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    # if OPENAI_API_KEY is set, the SDK will pick it up automatically
    return OpenAI()

# -------------------------
# Config
# -------------------------

@dataclass(frozen=True)
class LLMConfig:
    model_gloss: str = os.getenv("OPENAI_MODEL_GLOSS", "gpt-5-mini-2025-08-07")
    model_translate: str = os.getenv("OPENAI_MODEL_TRANSLATE", "gpt-5-mini-2025-08-07")
    max_retries: int = int(os.getenv("OPENAI_MAX_RETRIES", "3"))
    base_sleep: float = float(os.getenv("OPENAI_RETRY_BASE_SLEEP", "0.6"))

CFG = LLMConfig()

# -------------------------
# Helpers
# -------------------------

def split_gloss(en_gloss: str) -> List[str]:
    return [p.strip() for p in (en_gloss or "").split(";") if p.strip()]

def join_gloss(parts: List[str]) -> str:
    return "; ".join(p.strip() for p in parts if p and p.strip())

def _retry(call: Callable[[], Any], max_retries: int, base_sleep: float) -> Any:
    last: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            return call()
        except (RateLimitError, APITimeoutError, APIError) as e:
            last = e
            if attempt == max_retries:
                raise
            sleep = base_sleep * (2 ** (attempt - 1)) + random.uniform(0, 0.2)
            time.sleep(sleep)
    raise last  # pragma: no cover

def _json_schema_call(
    *,
    model: str,
    instructions: str,
    payload: dict,
    schema: dict,
    max_retries: int,
    base_sleep: float,
) -> Any:
    def _call():
        client = get_openai_client()
        resp = client.responses.create(
            model=model,
            reasoning={"effort": "low"},
            instructions=instructions,
            input=json.dumps(payload, ensure_ascii=False),
            text={"format": {"type": "json_schema", "name": "hanflow_json_schema", "schema": schema}},
        )
        return json.loads(resp.output_text)

    return _retry(_call, max_retries=max_retries, base_sleep=base_sleep)

# -------------------------
# Schemas
# -------------------------

def _schema_en_glosses(max_meanings: int) -> dict:
    return {
        "type": "object",
        "properties": {
            "glosses_en": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "minItems": 1,
                "maxItems": max_meanings,
            }
        },
        "required": ["glosses_en"],
        "additionalProperties": False,
    }

SCHEMA_ES_PT = {
    "type": "object",
    "properties": {
        "es": {"type": "array", "items": {"type": "string", "minLength": 1}},
        "pt": {"type": "array", "items": {"type": "string", "minLength": 1}},
    },
    "required": ["es", "pt"],
    "additionalProperties": False,
}

SCHEMA_TRIPLE = {
    "type": "object",
    "properties": {
        "en": {"type": "string", "minLength": 1},
        "es": {"type": "string", "minLength": 1},
        "pt": {"type": "string", "minLength": 1},
    },
    "required": ["en", "es", "pt"],
    "additionalProperties": False,
}

# -------------------------
# Dict enrichment schema + function
# -------------------------

_REGISTER_TAGS = [
    "daily", "spoken", "written", "formal", "casual",
    "colloquial", "slang", "internet", "figurative",
    "textbook", "polite", "blunt",
]

_VIBE_TAGS = [
    "warm", "soft", "cute", "playful", "cool", "sharp",
    "blunt", "judgmental", "emotional", "serious", "trendy",
    "professional", "everyday", "traditional-food", "intimate",
]

_POS_TAGS = ["n", "v", "adj", "adv", "pron", "prep", "conj", "mw", "idiom", "phrase"]

_LANG_TEXT_SCHEMA = {
    "type": "object",
    "properties": {
        "en": {"type": "string"},
        "es": {"type": "string"},
        "pt": {"type": "string"},
    },
    "required": ["en", "es", "pt"],
    "additionalProperties": False,
}

SCHEMA_DICT_ENTRY: dict = {
    "type": "object",
    "properties": {
        "senses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id":      {"type": "integer"},
                    "primary": {"type": "boolean"},
                    "pos":     {"type": "string", "enum": _POS_TAGS},
                    "defn":    _LANG_TEXT_SCHEMA,
                    "reg":     {"type": "array", "items": {"type": "string", "enum": _REGISTER_TAGS}},
                    "vibe":    {"type": "array", "items": {"type": "string", "enum": _VIBE_TAGS}},
                    "nat": {
                        "type": "object",
                        "properties": {
                            "sp":  {"type": "number", "minimum": 0, "maximum": 1},
                            "wr":  {"type": "number", "minimum": 0, "maximum": 1},
                            "net": {"type": "number", "minimum": 0, "maximum": 1},
                        },
                        "required": ["sp", "wr", "net"],
                        "additionalProperties": False,
                    },
                    "ex": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "zh": {"type": "string"},
                                "py": {"type": "string"},
                                "tr": _LANG_TEXT_SCHEMA,
                            },
                            "required": ["zh", "py", "tr"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["id", "primary", "pos", "defn", "reg", "vibe", "nat", "ex"],
                "additionalProperties": False,
            },
        },
        "pos": {
            "type": "array",
            "items": {"type": "string", "enum": _POS_TAGS},
        },
        "clf": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "hz":   {"type": "string"},
                    "num":  {"type": "string"},
                    "mark": {"type": "string"},
                },
                "required": ["hz", "num", "mark"],
                "additionalProperties": False,
            },
        },
        "search_kw": {
            "type": "object",
            "properties": {
                "en": {"type": "array", "items": {"type": "string"}},
                "es": {"type": "array", "items": {"type": "string"}},
                "pt": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["en", "es", "pt"],
            "additionalProperties": False,
        },
    },
    "required": ["senses", "pos", "clf", "search_kw"],
    "additionalProperties": False,
}

_DICT_ENRICH_INSTRUCTIONS = (
    "You are a Chinese lexicographer for Spanish and Portuguese learners. "
    "Given a Chinese word and its CEDICT entries, produce a structured dictionary entry. "
    "Use CEDICT senses as the basis for splitting senses — do not invent senses absent from the source. "
    "If no CEDICT entries are provided, infer senses from the word itself. "
    "Generate one example sentence per sense. Be concise."
)


def llm_enrich_dict_entry(hz: str, cedict_text: str) -> dict:
    """
    Call OpenAI to enrich a Chinese word into the DictEntry sense structure.

    Args:
        hz: simplified Chinese headword
        cedict_text: formatted CEDICT entries from cedict_entries_to_text(),
                     or empty string if no CEDICT entry exists

    Returns:
        dict with keys: senses, pos, clf, search_kw
    """
    payload: Dict[str, Any] = {"word": hz}
    if cedict_text:
        payload["cedict_entries"] = cedict_text

    return _json_schema_call(
        model=CFG.model_gloss,
        instructions=_DICT_ENRICH_INSTRUCTIONS,
        payload=payload,
        schema=SCHEMA_DICT_ENTRY,
        max_retries=CFG.max_retries,
        base_sleep=CFG.base_sleep,
    )

# -------------------------
# Public functions
# -------------------------

def llm_generate_en_meanings(word: str, max_meanings: int = 5) -> str:
    instructions = (
        "Create learner-friendly English dictionary glosses for the given Chinese word.\n"
        f"Return 1 to {max_meanings} concise glosses.\n"
        "Rules:\n"
        "- Each gloss: 1–4 words, dictionary-style (not a sentence).\n"
        "- Do NOT include the Chinese word in the gloss.\n"
        "- No explanations.\n"
    )

    data = _json_schema_call(
        model=CFG.model_gloss,
        instructions=instructions,
        payload={"word": word},
        schema=_schema_en_glosses(max_meanings),
        max_retries=CFG.max_retries,
        base_sleep=CFG.base_sleep,
    )

    glosses = [g.strip() for g in data["glosses_en"] if isinstance(g, str) and g.strip()]
    if not glosses:
        raise RuntimeError(f"Empty English glosses for: {word}")
    return join_gloss(glosses[:max_meanings])


def llm_translate_es_pt_from_en(en_meanings: str) -> Dict[str, str]:
    glosses_en = split_gloss(en_meanings)
    if not glosses_en:
        return {"es": "", "pt": ""}

    instructions = (
        "Translate each English gloss into Spanish (es) and Brazilian Portuguese (pt-BR).\n"
        "Rules:\n"
        "- Keep 1-to-1 mapping by index.\n"
        "- Output es and pt lists MUST match the input list length.\n"
        "- Each item short (1–4 words), dictionary-style.\n"
        "- No explanations.\n"
    )

    data = _json_schema_call(
        model=CFG.model_translate,
        instructions=instructions,
        payload={"glosses_en": glosses_en},
        schema=SCHEMA_ES_PT,
        max_retries=CFG.max_retries,
        base_sleep=CFG.base_sleep,
    )

    es_list = [s.strip() for s in data["es"]]
    pt_list = [s.strip() for s in data["pt"]]

    # minimal safety check (still worth it)
    if len(es_list) != len(glosses_en) or len(pt_list) != len(glosses_en):
        raise RuntimeError("LLM returned mismatched gloss list lengths")

    return {"es": join_gloss(es_list), "pt": join_gloss(pt_list)}


def llm_translate_text_triple(zh_text: str, max_retries: int = 3) -> Dict[str, str]:
    """
    Translate Chinese text into EN/ES/PT (natural sentence translation).
    """
    instructions = (
        "Translate the given Chinese text into:\n"
        "- English (en)\n"
        "- Spanish (es)\n"
        "- Brazilian Portuguese (pt)\n"
        "Rules:\n"
        "- Keep meaning faithful, natural.\n"
        "- Output JSON only.\n"
    )

    def _call():
        client = get_openai_client()
        resp = client.responses.create(
            model=CFG.model_translate,
            reasoning={"effort": "low"},
            instructions=instructions,
            input=json.dumps({"zh": zh_text}, ensure_ascii=False),
            text={"format": {"type": "json_schema", "name": "triple_text", "schema": SCHEMA_TRIPLE}},
        )
        return json.loads(resp.output_text)

    return _retry(_call, max_retries=max_retries, base_sleep=0.6)


if __name__ == "__main__":
    print(llm_translate_text_triple("我喜欢学习汉语，因为它很有趣！"))