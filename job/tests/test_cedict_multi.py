# backend/job/tests/test_cedict_multi.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dict_types import DictEntry, Sense, LangText, Pronunciation, LevelInfo, EditorialInfo


# ── Task 1 test (keep) ─────────────────────────────────────────────────────

def test_dict_entry_is_typed_dict():
    entry: DictEntry = {
        "_id": "发展",
        "hz": "发展",
        "py": [{"num": "fa1 zhan3", "mark": "fāzhǎn"}],
        "lvl": {"hsk": 2},
        "senses": [{"id": 1, "primary": True, "defn": {"en": "development"}}],
        "ed": {"status": "draft", "reviewed": False, "needsReview": False},
    }
    assert entry["hz"] == "发展"
    assert entry["senses"][0]["defn"]["en"] == "development"


# ── Task 2 tests ───────────────────────────────────────────────────────────

from cedict import load_cedict_all, cedict_lookup_all, cedict_entries_to_text

# Minimal in-memory CEDICT content for testing
_FAKE_CEDICT = """\
# CEDICT comment line
中 中 [Zhong1] /China/Chinese/
中 中 [zhong1] /middle/center/within/
中 中 [zhong4] /hit (a target)/to be hit/
發展 发展 [fa1 zhan3] /development/growth/to develop/
"""

def _parse_fake(content: str) -> dict:
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".u8", delete=False, encoding="utf-8") as f:
        f.write(content)
        path = f.name
    result = load_cedict_all(path)
    os.unlink(path)
    return result


def test_load_cedict_all_returns_all_entries_for_homograph():
    lex = _parse_fake(_FAKE_CEDICT)
    entries = lex["中"]
    assert len(entries) == 3
    pinyins = [e["pinyin"] for e in entries]
    assert "Zhong1" in pinyins
    assert "zhong1" in pinyins
    assert "zhong4" in pinyins


def test_load_cedict_all_single_entry_word():
    lex = _parse_fake(_FAKE_CEDICT)
    entries = lex["发展"]
    assert len(entries) == 1
    assert entries[0]["pinyin"] == "fa1 zhan3"
    assert "development" in entries[0]["senses"]


def test_cedict_lookup_all_miss_returns_empty():
    lex = _parse_fake(_FAKE_CEDICT)
    assert cedict_lookup_all("不存在", lex) == []


def test_cedict_entries_to_text_formats_correctly():
    entries = [
        {"pinyin": "fa1 zhan3", "senses": ["development", "growth"]},
        {"pinyin": "fa1 zhan4", "senses": ["to expand"]},
    ]
    text = cedict_entries_to_text(entries)
    assert "1. [fa1 zhan3] development; growth" in text
    assert "2. [fa1 zhan4] to expand" in text


def test_cedict_entries_to_text_empty():
    assert cedict_entries_to_text([]) == ""
