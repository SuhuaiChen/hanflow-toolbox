# backend/job/tests/test_translate_schema.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from translate import SCHEMA_DICT_ENTRY, llm_enrich_dict_entry
import jsonschema


def test_schema_dict_entry_structure():
    """Verify the schema has the required top-level keys."""
    props = SCHEMA_DICT_ENTRY["properties"]
    assert "senses" in props
    assert "pos" in props
    assert "clf" in props
    assert "search_kw" in props
    assert SCHEMA_DICT_ENTRY.get("additionalProperties") is False


def test_schema_senses_items_have_required_fields():
    sense_schema = SCHEMA_DICT_ENTRY["properties"]["senses"]["items"]
    required = sense_schema["required"]
    for field in ("id", "primary", "pos", "defn", "reg", "vibe", "nat", "ex"):
        assert field in required, f"'{field}' missing from senses required"


def test_schema_reg_enum_values():
    sense_schema = SCHEMA_DICT_ENTRY["properties"]["senses"]["items"]
    reg_enum = sense_schema["properties"]["reg"]["items"]["enum"]
    assert "daily" in reg_enum
    assert "slang" in reg_enum
    assert "blunt" in reg_enum
    # Verify no invalid tags
    assert "angry" not in reg_enum


def test_schema_vibe_enum_values():
    sense_schema = SCHEMA_DICT_ENTRY["properties"]["senses"]["items"]
    vibe_enum = sense_schema["properties"]["vibe"]["items"]["enum"]
    assert "warm" in vibe_enum
    assert "traditional-food" in vibe_enum
    assert "intimate" in vibe_enum
    assert "boring" not in vibe_enum


def test_schema_validates_a_well_formed_llm_response():
    """
    jsonschema-validate a response that looks like what the LLM would return.
    """
    sample = {
        "senses": [{
            "id": 1,
            "primary": True,
            "pos": "v",
            "defn": {"en": "to develop", "es": "desarrollar", "pt": "desenvolver"},
            "reg": ["daily", "written"],
            "vibe": ["serious", "professional"],
            "nat": {"sp": 0.7, "wr": 0.9, "net": 0.4},
            "ex": [{
                "zh": "经济发展很快。",
                "py": "jīngjì fāzhǎn hěn kuài.",
                "tr": {"en": "The economy develops rapidly.", "es": "La economía se desarrolla rápidamente.", "pt": "A economia se desenvolve rapidamente."},
            }],
        }],
        "pos": ["v"],
        "clf": [],
        "search_kw": {"en": ["develop", "growth"], "es": ["desarrollar"], "pt": ["desenvolver"]},
    }
    # Should not raise
    jsonschema.validate(instance=sample, schema=SCHEMA_DICT_ENTRY)


def test_schema_rejects_invalid_reg_tag():
    import pytest
    bad = {
        "senses": [{
            "id": 1, "primary": True, "pos": "n",
            "defn": {"en": "x", "es": "x", "pt": "x"},
            "reg": ["not-a-real-tag"],
            "vibe": [],
            "nat": {"sp": 0.5, "wr": 0.5, "net": 0.5},
            "ex": [],
        }],
        "pos": ["n"], "clf": [],
        "search_kw": {"en": [], "es": [], "pt": []},
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=bad, schema=SCHEMA_DICT_ENTRY)
