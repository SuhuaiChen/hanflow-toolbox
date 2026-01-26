import re
from typing import Literal

UmlautStyle = Literal["ü", "u:", "v"]
NeutralStyle = Literal["omit", "5"]  # omit neutral tone number, or append "5"


# -----------------------------
# Tone mark tables
# -----------------------------
_TONE_MARKS = {
    "a": ["a", "ā", "á", "ǎ", "à"],
    "e": ["e", "ē", "é", "ě", "è"],
    "i": ["i", "ī", "í", "ǐ", "ì"],
    "o": ["o", "ō", "ó", "ǒ", "ò"],
    "u": ["u", "ū", "ú", "ǔ", "ù"],
    "ü": ["ü", "ǖ", "ǘ", "ǚ", "ǜ"],
    "A": ["A", "Ā", "Á", "Ǎ", "À"],
    "E": ["E", "Ē", "É", "Ě", "È"],
    "I": ["I", "Ī", "Í", "Ǐ", "Ì"],
    "O": ["O", "Ō", "Ó", "Ǒ", "Ò"],
    "U": ["U", "Ū", "Ú", "Ǔ", "Ù"],
    "Ü": ["Ü", "Ǖ", "Ǘ", "Ǚ", "Ǜ"],
}

# Reverse lookup: accented vowel -> (base vowel char, tone 1-4)
_REVERSE_TONE = {}
for base, arr in _TONE_MARKS.items():
    for tone, ch in enumerate(arr):
        if tone == 0:
            continue
        _REVERSE_TONE[ch] = (base, tone)

_VOWELS_ALL = set("aeiouüAEIOUÜ")

# Build a pinyin chunk regex that matches:
# - ASCII letters
# - ü/Ü
# - "u:" / "U:" (kept as ':' inside chunk)
# - accented vowels (āáǎà etc.)
# - optional trailing tone digit [0-5]
_ACCENTED = "".join(_REVERSE_TONE.keys())
_PINYIN_CHUNK_RE = re.compile(
    rf"[A-Za-zÜü:{re.escape(_ACCENTED)}]+[0-5]?"
)


def _normalize_umlaut_in_num(s: str) -> str:
    # accept "u:" / "U:" / "v" / "V" as ü
    return (
        s.replace("u:", "ü")
         .replace("U:", "Ü")
         .replace("v", "ü")
         .replace("V", "Ü")
    )


def _apply_umlaut_style_num(s: str, style: UmlautStyle) -> str:
    if style == "ü":
        return s
    if style == "u:":
        return s.replace("ü", "u:").replace("Ü", "U:")
    if style == "v":
        return s.replace("ü", "v").replace("Ü", "V")
    raise ValueError(f"Unknown umlaut_style: {style}")


def _tone_target_index(syllable: str) -> int:
    """
    Decide which vowel should get the tone mark according to standard pinyin rules.
    Input syllable has NO tone number.
    Returns index of the vowel to mark, or -1 if none.
    """
    s = syllable
    lower = s.lower()

    # Special cases: iu / ui
    if "iu" in lower:
        return lower.rfind("iu") + 1  # mark 'u'
    if "ui" in lower:
        return lower.rfind("ui") + 1  # mark 'i'

    # Priority: a, e, or 'o' in 'ou'
    for v in ("a", "e"):
        pos = lower.find(v)
        if pos != -1:
            return pos

    ou_pos = lower.find("ou")
    if ou_pos != -1:
        return ou_pos  # mark 'o'

    # Otherwise, mark the last vowel
    for i in range(len(s) - 1, -1, -1):
        if s[i] in _VOWELS_ALL:
            return i

    return -1


def pinyin_num_to_tone(pinyin_num: str) -> str:
    """
    Convert pinyin with tone numbers -> tone marks.

    Examples:
      "pin1 yin1" -> "pīn yīn"
      "lü4 se4"   -> "lǜ sè"
      "nu:3"      -> "nǚ"
      "lv3"       -> "lǚ"

    Neutral tone (0/5 or missing number) -> no mark.
    """
    text = _normalize_umlaut_in_num(pinyin_num)

    out = []
    i = 0
    while i < len(text):
        m = _PINYIN_CHUNK_RE.match(text, i)
        if not m:
            out.append(text[i])
            i += 1
            continue

        chunk = m.group(0)
        i = m.end()

        tone = 0
        if chunk and chunk[-1].isdigit():
            tone = int(chunk[-1])
            chunk = chunk[:-1]

        # Normalize any leftover u:/v just in case (should already be done)
        chunk = _normalize_umlaut_in_num(chunk)

        # Neutral: 0/5 or no tone
        if tone in (0, 5) or not chunk:
            out.append(chunk)
            continue

        idx = _tone_target_index(chunk)
        if idx == -1:
            out.append(chunk)
            continue

        base_vowel = chunk[idx]
        if base_vowel not in _TONE_MARKS:
            out.append(chunk)
            continue

        marked = _TONE_MARKS[base_vowel][tone]
        out.append(chunk[:idx] + marked + chunk[idx + 1:])

    return "".join(out)


def pinyin_tone_to_num(
    pinyin_tone: str,
    *,
    neutral_style: NeutralStyle = "omit",
    umlaut_style: UmlautStyle = "ü",
) -> str:
    """
    Convert pinyin with tone marks -> tone numbers.

    Examples:
      "pīn yīn" -> "pin1 yin1"
      "lǜ sè"   -> "lü4 se4"

    Neutral syllables:
      neutral_style="omit" => "shi"
      neutral_style="5"    => "shi5"

    Output ü style:
      umlaut_style="ü"  => "lü4"
      umlaut_style="u:" => "lu:4"
      umlaut_style="v"  => "lv4"
    """
    text = pinyin_tone

    out = []
    i = 0
    while i < len(text):
        m = _PINYIN_CHUNK_RE.match(text, i)
        if not m:
            out.append(text[i])
            i += 1
            continue

        chunk = m.group(0)
        i = m.end()

        tone = 0
        chars = list(chunk)

        # Replace the FIRST accented vowel we find in this chunk
        for j, ch in enumerate(chars):
            if ch in _REVERSE_TONE:
                base, t = _REVERSE_TONE[ch]
                tone = t
                chars[j] = base
                break

        plain = "".join(chars)

        # Normalize ü inputs (in case input includes u: or v somehow)
        plain = _normalize_umlaut_in_num(plain)
        # Then apply desired output ü style
        plain = _apply_umlaut_style_num(plain, umlaut_style)

        if tone == 0:
            if neutral_style == "5":
                out.append(plain + "5")
            else:
                out.append(plain)
        else:
            out.append(plain + str(tone))

    return "".join(out)


# -----------------------------
# Sanity tests
# -----------------------------
if __name__ == "__main__":
    assert pinyin_num_to_tone("pin1 yin1") == "pīn yīn"
    assert pinyin_num_to_tone("lv3") == "lǚ"
    assert pinyin_num_to_tone("nu:3") == "nǚ"
    assert pinyin_num_to_tone("lü4 se4") == "lǜ sè"

    assert pinyin_tone_to_num("pīn yīn") == "pin1 yin1"
    assert pinyin_tone_to_num("lǚ", umlaut_style="v") == "lv3"
    assert pinyin_tone_to_num("shi", neutral_style="5") == "shi5"
    assert pinyin_tone_to_num("lǜ sè") == "lü4 se4"
    print("All tests passed.")
    print(pinyin_num_to_tone("Zhong1 guo2 ren2"))  # Zhōng guó rén
    print(pinyin_num_to_tone("qiu1 hou4 suan4 zhang4")) # qiū hòu suàn zhàng
    print(pinyin_tone_to_num("qiū hòu suàn zhàng")) # qiu1 hou4 suan4 zhang4
    print(pinyin_num_to_tone("hui4 bu hui4"))  # huì bù huì
    print(pinyin_tone_to_num("qíng tiān"))  # qing2 tian1
    print(pinyin_tone_to_num("nǔ hái"))  # nü3 hai2


