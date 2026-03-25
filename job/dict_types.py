# backend/job/dict_types.py
from typing import TypedDict, Literal, List, Optional
from datetime import datetime


class LangText(TypedDict, total=False):
    en: str
    es: str
    pt: str


class Naturalness(TypedDict, total=False):
    sp: float   # spoken naturalness 0–1
    wr: float   # written naturalness 0–1
    net: float  # internet/chat naturalness 0–1


RegisterTag = Literal[
    "daily", "spoken", "written", "formal", "casual",
    "colloquial", "slang", "internet", "figurative",
    "textbook", "polite", "blunt",
]

VibeTag = Literal[
    "warm", "soft", "cute", "playful", "cool", "sharp",
    "blunt", "judgmental", "emotional", "serious", "trendy",
    "professional", "everyday", "traditional-food", "intimate",
]

PosTag = Literal["n", "v", "adj", "adv", "pron", "prep", "conj", "mw", "idiom", "phrase"]
EntryType = Literal["word", "phrase", "idiom", "char"]
EditorialStatus = Literal["draft", "reviewed", "approved"]


class Pronunciation(TypedDict, total=False):
    num: str    # tone-numbered: fa1 zhan3
    mark: str   # tone-marked: fāzhǎn
    audio: str  # R2 MP3 URL


class Classifier(TypedDict, total=False):
    hz: str
    num: str
    mark: str


class ExampleSentence(TypedDict, total=False):
    zh: str
    py: str
    tr: LangText


class Sense(TypedDict, total=False):
    id: int
    primary: bool
    pos: PosTag
    defn: LangText          # single meaning field
    reg: List[RegisterTag]
    vibe: List[VibeTag]
    nat: Naturalness
    ex: List[ExampleSentence]


class SearchKeywords(TypedDict, total=False):
    en: List[str]
    es: List[str]
    pt: List[str]


class SearchIndex(TypedDict, total=False):
    hz: List[str]   # word + individual characters
    py: List[str]   # pinyin variants (num + mark)
    kw: SearchKeywords


class LevelInfo(TypedDict, total=False):
    hsk: Optional[int]  # 1–6, or None if not in HSK


class SourceInfo(TypedDict, total=False):
    base: str   # "cedict" | "llm"
    py: str     # "cedict" | "pypinyin"


class EditorialInfo(TypedDict, total=False):
    status: EditorialStatus
    reviewed: bool
    needsReview: bool


class DictEntry(TypedDict, total=False):
    _id: str            # simplified Chinese headword
    type: EntryType
    hz: str
    py: List[Pronunciation]
    pos: List[PosTag]
    clf: List[Classifier]
    lvl: LevelInfo
    senses: List[Sense]
    search: SearchIndex
    src: SourceInfo
    ed: EditorialInfo
    updatedAt: datetime  # BSON date in MongoDB
