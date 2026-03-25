# backend/job/tests/test_cedict_multi.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dict_types import DictEntry, Sense, LangText, Pronunciation, LevelInfo, EditorialInfo


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
    assert entry["lvl"]["hsk"] == 2
    assert entry["ed"]["status"] == "draft"
