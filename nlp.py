## quick validation of the data
from ltp import LTP
from hsk_data import hsk_words, hsk_characters, hsk_grammar_with_regex
import re

ltp = LTP("LTP/small") #  other models: LTP/base, LTP/small, LTP/tiny, LTP/legacy

def tokenize_with_pos(sentence):
    cws, pos = ltp.pipeline([sentence], tasks=["cws", "pos"]).to_tuple()
    # print(cws, pos)
    return list(zip(cws[0], pos[0]))

# classify using hsk_words dict
def classify_word_level(word):
    level = hsk_words.get(word, None)
    return level

def classify_words_levels(words):
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
        level = classify_word_level(word)
        if level:
            word_levels[level].append(word)
        else:
            word_levels["非HSK词汇"].append(word)
    return word_levels

def classify_sentence_levels(sentence):
    tokens = tokenize_with_pos(sentence)
    words = [token for token, pos in tokens]
    word_levels = classify_words_levels(words)
    return word_levels

def classify_paragraph_levels(paragraph):
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
            word_levels_of_a_sentence = classify_sentence_levels(sentence)
            for level in word_levels:
                word_levels[level].extend(word_levels_of_a_sentence[level])
    return word_levels

# classify using hsk characters dict
def classify_character_level(character):
    level = hsk_characters.get(character, None)
    return level

def classify_characters_levels(paragraph):
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
        level = classify_character_level(character)
        if level:
            character_levels[level].append(character)
        else:
            character_levels["非HSK汉字"].append(character)
    return character_levels

# classify using hsk grammar
def classify_grammar_points(sentence):
    matched_grammar_points = []
    for regex, (level, grammar_content) in hsk_grammar_with_regex.items():
        if re.search(regex, sentence):
            matched_grammar_points.append((level, grammar_content))
    return matched_grammar_points


example_paragraph = "我爱学习人工智能。它是未来的发展方向！"
word_levels = classify_paragraph_levels(example_paragraph)
character_levels = classify_characters_levels(example_paragraph)
print("Word Levels:", word_levels)
print("Character Levels:", character_levels)

# example_sentence = "我喜欢做饭和做生意"
grammar_points = classify_grammar_points(example_paragraph)
print("Matched Grammar Points:", grammar_points)

### simple logic to classify the level of hsk only based on words

def classify_hsk_level_of_text(paragraph):
    word_levels = classify_paragraph_levels(paragraph)
   # the level should be determined by the most common level of words in the text as long as the higher level words are not too many
    level_counts = {level: len(word_levels[level]) for level in word_levels}
    total_words = sum(level_counts.values())

    # show word level distribution
    print("Word Level Distribution:", level_counts) 

    # iterate the levels in reverse order
    acc_count = 0
    for level in ["高等", "六级", "五级", "四级", "三级", "二级", "一级"]:
        acc_count += level_counts[level]
        if acc_count / total_words > 0.1:  # more than 10% words are of this level
            return level