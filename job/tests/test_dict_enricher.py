# backend/job/tests/test_dict_enricher.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import MagicMock
from dict_enricher import build_word_list


def test_build_word_list_hsk_words_come_first():
    hsk_words = {"你好": 1, "发展": 2, "中文": 3}
    dict_col = MagicMock()
    dict_col.find.return_value = [
        {"_id": "你好"},
        {"_id": "发展"},
        {"_id": "未知词"},   # non-HSK word from dict
    ]

    words = build_word_list(hsk_words, dict_col)

    # HSK words come first
    assert words[0] in hsk_words
    assert words[1] in hsk_words
    assert words[2] in hsk_words
    # Non-HSK word appended after
    assert "未知词" in words
    assert words.index("未知词") > words.index("你好")


def test_build_word_list_no_duplicates():
    hsk_words = {"你好": 1}
    dict_col = MagicMock()
    dict_col.find.return_value = [{"_id": "你好"}]  # also in HSK

    words = build_word_list(hsk_words, dict_col)
    assert words.count("你好") == 1


def test_build_word_list_single_word_flag():
    hsk_words = {"你好": 1, "发展": 2}
    dict_col = MagicMock()

    words = build_word_list(hsk_words, dict_col, word="发展")
    assert words == ["发展"]
    dict_col.find.assert_not_called()


def test_build_word_list_limit():
    hsk_words = {str(i): i for i in range(100)}
    dict_col = MagicMock()
    dict_col.find.return_value = []

    words = build_word_list(hsk_words, dict_col, limit=10)
    assert len(words) == 10
