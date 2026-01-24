from typing import List, Dict, Tuple
from ltp import LTP, StnSplit
from cedict import load_cedict_simplified
from hsk_data import load_hsk_characters, load_hsk_words, df_load_hsk_grammar_with_regex
from pypinyin import pinyin, Style
import re

ltp = LTP("LTP/small") #  other models: LTP/base, LTP/small, LTP/tiny, LTP/legacy

def tokenize_with_pos(sentence):
    cws, pos = ltp.pipeline([sentence], tasks=["cws", "pos"]).to_tuple()
    # print(cws, pos)
    return list(zip(cws[0], pos[0]))

# classify using hsk_words dict
def classify_word_level(word, hsk_words):
    level = hsk_words.get(word, None)
    return level

def classify_words_levels(words, hsk_words):
    word_levels = {
        "一级": [],
        "二级": [],
        "三级": [],
        "四级": [],  
        "五级": [],
        "六级": [],
        "高等": [],
        "非HSK词汇": []
    }
    for word in words:
        level = classify_word_level(word, hsk_words)
        if level:
            word_levels[level].append(word)
        else:
            word_levels["非HSK词汇"].append(word)
    return word_levels

def classify_sentence_levels(sentence, hsk_words):
    tokens = tokenize_with_pos(sentence)
    words = [token for token, pos in tokens]
    word_levels = classify_words_levels(words, hsk_words)
    return word_levels

def classify_paragraph_levels(paragraph, hsk_words):
    punctuation_pattern = r'[。！？]|……'
    sentences = re.split(punctuation_pattern, paragraph)  # 简单按句号分割句子
    word_levels = {
        "一级": [],
        "二级": [],
        "三级": [],
        "四级": [],  
        "五级": [],
        "六级": [],
        "高等": [],
        "非HSK词汇": []
    } 
    for sentence in sentences:
        if sentence.strip():  # 忽略空句子
            word_levels_of_a_sentence = classify_sentence_levels(sentence, hsk_words)
            for level in word_levels:
                word_levels[level].extend(word_levels_of_a_sentence[level])
    return word_levels

# classify using hsk characters dict
def classify_character_level(character, hsk_characters):
    level = hsk_characters.get(character, None)
    return level

def classify_characters_levels(paragraph, hsk_characters):
    character_levels = {
        "一级": [],
        "二级": [],
        "三级": [],
        "四级": [],  
        "五级": [],
        "六级": [],
        "高等": [],
        "非HSK汉字": []
    }
    characters = list(paragraph)
    # 去除标点符号和空白字符
    characters = [char for char in characters if re.match(r'\S', char) and not re.match(r'[。！？！，、；：“”‘’（）《》〈〉]', char)]
    for character in characters:
        level = classify_character_level(character, hsk_characters)
        if level:
            character_levels[level].append(character)
        else:
            character_levels["非HSK汉字"].append(character)
    return character_levels

# classify using hsk grammar
def classify_grammar_points(sentence, hsk_grammar_with_regex):
    matched_grammar_points = []
    for regex, (level, grammar_content) in hsk_grammar_with_regex.items():
        if re.search(regex, sentence):
            matched_grammar_points.append((level, grammar_content))
    return matched_grammar_points

### simple logic to classify the level of hsk only based on words
def classify_hsk_level_of_text(paragraph, hsk_words):
    word_levels = classify_paragraph_levels(paragraph, hsk_words)
   # the level should be determined by the most common level of words in the text as long as the higher level words are not too many
    level_counts = {level: len(word_levels[level]) for level in word_levels}
    total_words = sum(level_counts.values())

    # show word level distribution
    print("Word Level Distribution:", level_counts) 

    # iterate the levels in reverse order
    acc_count = 0
    for level in ["高等", "六级", "五级", "四级", "三级", "二级", "一级"]:
        acc_count += level_counts[level]
        if acc_count / total_words > 0.2:  # more than 20% words are of this level
            return level
        

def split_to_sentence_objects(text: str) -> List[Dict]:
    """
    Output format:
    [
      { "sid": "s1", "zh": "..." },
      { "sid": "s2", "zh": "..." },
      ...
    ]
    """
    sents = StnSplit().split(text)
    return [{"sid": f"s{i+1}", "zh": s} for i, s in enumerate(sents)]

def sentence_to_tokens(sentence: str) -> List[Dict]:
    """
    Output format:
    [
      { "token": "...", "pos": "..." },
      ...
    """
    tokens = tokenize_with_pos(sentence)
    return [{"token": token, "pos": pos} for token, pos in tokens]

def to_pinyin(word: str) -> str:
    # TODO: need to handle polyphonic characters
    pys = pinyin(word, style=Style.TONE, strict=False)
    return "".join(s[0] for s in pys)

def transform_tokens(
    segmented: List[Tuple[str, str]],
    lexicon: Dict[str, Dict[str, str]],
    hsk_words: Dict[str, str]
) -> List[Dict]:
    tokens = []
    for word, pos in segmented:
        if pos == "wp":
            tokens.append({
                "t": word,
                "type": "punct"
            })
            continue

        meanings = {"en": lexicon[word]["senses"] if word in lexicon else ""}
        hsk_level = classify_word_level(word, hsk_words)

        tokens.append({
            "t": word,
            "pinyin": to_pinyin(word),
            "meanings": meanings,     
            "hsk_level": hsk_level,
            "type": pos         # r / v / n / nt / u ...
        })
    return tokens

from typing import List, Dict

if __name__ == "__main__":
    hsk_characters = load_hsk_characters()
    hsk_words = load_hsk_words()
    hsk_grammar_with_regex = df_load_hsk_grammar_with_regex()
    lexicon = load_cedict_simplified("data/cedict_ts.u8")

    example_paragraph = "我爱学习人工智能。它是未来的发展方向！"
    word_levels = classify_paragraph_levels(example_paragraph, hsk_words)
    character_levels = classify_characters_levels(example_paragraph, hsk_characters)
    print("Word Levels:", word_levels)
    print("Character Levels:", character_levels)

    # example_sentence = "我喜欢做饭和做生意"
    grammar_points = classify_grammar_points(example_paragraph, hsk_grammar_with_regex)
    print("Matched Grammar Points:", grammar_points)
    print(transform_tokens(tokenize_with_pos(example_paragraph), lexicon, hsk_words))

    print(to_pinyin("发展")) # should return "fa1 zhan3"