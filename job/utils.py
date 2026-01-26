# 词性标签说明
# https://ltp.ai/docs/appendix.html

HSK_Tags = {'介', '动', '形', '量', '名', '数', '代', '连', '前缀', '拟声', '叹', '副', '助', '后缀'}

LTP_POS_TAGS = {
    # ===== 名词类 =====
    "n":  "一般名词  general noun，例如：苹果",
    "nh": "人名     person name，例如：杜甫, 汤姆",
    "ns": "地名     geographical name，例如：北京",
    "nt": "时间名词 temporal noun，例如：近日, 明代",
    "nl": "处所名词 location noun，例如：城郊",
    "nd": "方位名词 direction noun，例如：右侧",
    "ni": "机构名   organization name，例如：保险公司",
    "nz": "其他专名 other proper noun，例如：诺贝尔奖",
    "ws": "外来词   foreign word，例如：CPU",

    # ===== 实词：动词 / 形容词 / 数量词等 =====
    "v":  "动词     verb，例如：跑, 学习",
    "a":  "形容词   adjective，例如：美丽",
    "d":  "副词     adverb，例如：很",
    "m":  "数词     number，例如：一, 第一",
    "q":  "量词     measure word / quantity，例如：个",
    "z":  "状态词   descriptive word，例如：瑟瑟, 匆匆",
    "i":  "成语     idiom，例如：百花齐放",

    # ===== 虚词类 =====
    "p":  "介词     preposition，例如：在, 把",
    "u":  "助词     auxiliary，例如：的, 地",
    "r":  "代词     pronoun，例如：我们",
    "c":  "连词     conjunction，例如：和, 虽然",
    "e":  "叹词     exclamation，例如：哎",
    "o":  "拟声词   onomatopoeia，例如：哗啦",
    "wp": "标点符号 punctuation，例如：，。！",

    # ===== 构词成分 / 其他 =====
    "g":  "语素     morpheme（成词的一部分），例如：茨, 甥",
    "h":  "前缀     prefix，例如：阿, 伪",
    "k":  "后缀     suffix，例如：界, 率",
    "j":  "简称     abbreviation，例如：公检法",
    "b":  "名词性修饰语 other noun-modifier，例如：大型, 西式",
    "x":  "非成词语素 non-lexeme，例如：萄, 翱",
}
