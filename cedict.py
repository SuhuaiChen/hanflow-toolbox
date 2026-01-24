import re
from typing import Dict
from pypinyin import pinyin, Style

def to_pinyin(word: str) -> str:
    # TODO: need to handle polyphonic characters
    pys = pinyin(word, style=Style.TONE, strict=False)
    return "".join(s[0] for s in pys)

# trad simp [pinyin] /sense1/sense2/.../
CEDICT_RE = re.compile(r"^(\S+)\s+(\S+)\s+\[(.+?)\]\s+/(.+)/$")

def load_cedict_simplified(path: str) -> Dict[str, dict]:
    """
    Return:
    {
      simp_word: {"pinyin": "fa1 zhan3", "senses": ["development", "growth"]}
    }
    """
    lexicon: Dict[str, dict] = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            m = CEDICT_RE.match(line)
            if not m:
                continue

            trad, simp, pinyin, senses_raw = m.groups()
            senses = [s for s in senses_raw.split("/") if s]

            # for now, only simplified lexicon
            if simp not in lexicon:
                lexicon[simp] = {"pinyin": pinyin, "senses": senses}

    return lexicon

def cedict_lookup_en(word: str, lexicon: Dict[str, dict]) -> str | None:
    """
    Return the English meanings string (semicolon separated), or None if not found.
    """
    entry = lexicon.get(word)
    if entry:
        return "; ".join(entry["senses"])
    return None

def cedict_lookup_pinyin(word: str, lexicon: Dict[str, dict]) -> str | None:
    """
    Return the pinyin string, or None if not found.
    """
    entry = lexicon.get(word)
    if entry:
        return entry["pinyin"]
    return None

if __name__ == "__main__":
    cedict_path = "data/cedict_ts.u8"
    lexicon = load_cedict_simplified(cedict_path)
    print(f"Loaded {len(lexicon)} entries from CEDICT.")
    # Example lookup
    example_word = "发展"
    if example_word in lexicon:
        entry = lexicon[example_word]
        print(f"Word: {example_word}")
        print(f"Pinyin: {entry['pinyin']}")
        print(f"Senses: {', '.join(entry['senses'])}")
    else:
        print(f"Word '{example_word}' not found in CEDICT.")
    
    print(cedict_lookup_en(example_word, lexicon))
    print(to_pinyin(example_word))
    # TODO covert fa1 zhan3 to fāzhǎn