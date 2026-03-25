# backend/job/tests/test_enrich_pipeline.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

from db import enrich_dict_entry, ensure_dict_entry


# ── helpers ───────────────────────────────────────────────────────────────

def _fake_llm_result():
    return {
        "senses": [{
            "id": 1, "primary": True, "pos": "v",
            "defn": {"en": "to develop", "es": "desarrollar", "pt": "desenvolver"},
            "reg": ["daily"], "vibe": ["serious"],
            "nat": {"sp": 0.7, "wr": 0.9, "net": 0.4},
            "ex": [{"zh": "经济发展。", "py": "jīngjì fāzhǎn.", "tr": {"en": "Economy develops.", "es": "La economía.", "pt": "A economia."}}],
        }],
        "pos": ["v"],
        "clf": [],
        "search_kw": {"en": ["develop"], "es": ["desarrollar"], "pt": ["desenvolver"]},
    }


def _fake_cedict_all():
    return {
        "发展": [{"pinyin": "fa1 zhan3", "senses": ["development", "growth"]}],
    }


def _fake_hsk_words():
    return {"发展": 2}


def _make_mock_s3():
    return MagicMock()


# ── tests ─────────────────────────────────────────────────────────────────

@patch("db.attach_tts_audio", return_value="https://r2.example.com/audio.mp3")
@patch("db.llm_enrich_dict_entry", return_value=_fake_llm_result())
def test_enrich_dict_entry_shape(mock_llm, mock_tts):
    s3 = _make_mock_s3()
    entry = enrich_dict_entry(
        "发展",
        cedict_all=_fake_cedict_all(),
        hsk_words=_fake_hsk_words(),
        s3=s3,
    )

    assert entry["_id"] == "发展"
    assert entry["hz"] == "发展"
    assert entry["lvl"]["hsk"] == 2
    assert entry["py"][0]["num"] == "fa1 zhan3"
    assert entry["py"][0]["mark"] == "fāzhǎn"
    assert entry["py"][0]["audio"] == "https://r2.example.com/audio.mp3"
    assert entry["senses"][0]["defn"]["en"] == "to develop"
    assert entry["src"]["base"] == "cedict"
    assert entry["src"]["py"] == "cedict"
    assert entry["ed"]["status"] == "draft"
    assert entry["ed"]["reviewed"] is False
    assert isinstance(entry["updatedAt"], datetime)


@patch("db.attach_tts_audio", return_value="https://r2.example.com/audio.mp3")
@patch("db.llm_enrich_dict_entry", return_value=_fake_llm_result())
def test_enrich_dict_entry_search_index(mock_llm, mock_tts):
    entry = enrich_dict_entry(
        "发展",
        cedict_all=_fake_cedict_all(),
        hsk_words=_fake_hsk_words(),
        s3=_make_mock_s3(),
    )
    assert "发展" in entry["search"]["hz"]
    assert "发" in entry["search"]["hz"]
    assert "展" in entry["search"]["hz"]
    assert "fa1 zhan3" in entry["search"]["py"]
    assert "fāzhǎn" in entry["search"]["py"]
    assert "develop" in entry["search"]["kw"]["en"]


@patch("db.attach_tts_audio", return_value="https://r2.example.com/audio.mp3")
@patch("db.llm_enrich_dict_entry", return_value=_fake_llm_result())
def test_enrich_dict_entry_cedict_miss_uses_pypinyin(mock_llm, mock_tts):
    """When word is not in CEDICT, src.base = 'llm' and src.py = 'pypinyin'."""
    entry = enrich_dict_entry(
        "发展",
        cedict_all={},          # empty: CEDICT miss
        hsk_words={},
        s3=_make_mock_s3(),
    )
    assert entry["src"]["base"] == "llm"
    assert entry["src"]["py"] == "pypinyin"
    assert entry["lvl"]["hsk"] is None


@patch("db.attach_tts_audio", side_effect=Exception("TTS down"))
@patch("db.llm_enrich_dict_entry", return_value=_fake_llm_result())
def test_enrich_dict_entry_tts_failure_still_upserts(mock_llm, mock_tts):
    """TTS failure should not raise — entry upserted without audio."""
    entry = enrich_dict_entry(
        "发展",
        cedict_all=_fake_cedict_all(),
        hsk_words=_fake_hsk_words(),
        s3=_make_mock_s3(),
    )
    assert "audio" not in entry["py"][0]


@patch("db.attach_tts_audio", return_value="https://r2.example.com/audio.mp3")
@patch("db.llm_enrich_dict_entry", return_value=_fake_llm_result())
def test_ensure_dict_entry_cache_hit_skips_pipeline(mock_llm, mock_tts):
    col = MagicMock()
    col.find_one.return_value = {"_id": "发展", "senses": [{"id": 1}]}

    result = ensure_dict_entry(
        "发展", col,
        cedict_all=_fake_cedict_all(),
        hsk_words=_fake_hsk_words(),
        s3=_make_mock_s3(),
    )

    mock_llm.assert_not_called()
    assert result["_id"] == "发展"


@patch("db.attach_tts_audio", return_value="https://r2.example.com/audio.mp3")
@patch("db.llm_enrich_dict_entry", return_value=_fake_llm_result())
def test_ensure_dict_entry_cache_miss_runs_pipeline(mock_llm, mock_tts):
    col = MagicMock()
    col.find_one.return_value = None  # cache miss

    result = ensure_dict_entry(
        "发展", col,
        cedict_all=_fake_cedict_all(),
        hsk_words=_fake_hsk_words(),
        s3=_make_mock_s3(),
    )

    mock_llm.assert_called_once()
    col.replace_one.assert_called_once()
    assert result["senses"][0]["defn"]["en"] == "to develop"
