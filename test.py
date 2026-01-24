from nlp import tokenize_with_pos,classify_hsk_level_of_text
from hsk_data import df_characters, df_words, df_hand_written_characters, df_grammar
import json 

# =========================
# 1. 数据概览（Data Overview）
# =========================

# ---- 1.1 汉字（按级别统计）----
print("\n\n" + "*" * 5 + "汉字" + "*" * 5)
total_rows = len(df_characters)
print("Total rows:", total_rows)
# 按“级别”统计汉字数量（与官方 HSK 标准大致相符，但并非完全一致）
rows_by_level = df_characters['级别'].value_counts()
print(rows_by_level)


# ---- 1.2 词汇（按级别统计）----
print("\n\n" + "*" * 5 + "词汇" + "*" * 5)
total_rows = len(df_words)
print("Total rows:", total_rows)
rows_by_level = df_words['级别'].value_counts()
print(rows_by_level)


# ---- 1.3 手写汉字（按等次统计）----
print("\n\n" + "*" * 5 + "手写汉字" + "*" * 5)
total_rows = len(df_hand_written_characters)
print("Total rows:", total_rows)
# 注意：这里用的是“等次”而不是“级别”
rows_by_level = df_hand_written_characters['等次'].value_counts()
print(rows_by_level)


# ---- 1.4 语法点（按级别统计）----
print("\n\n" + "*" * 5 + "语法" + "*" * 5)
total_rows = len(df_grammar)
print("Total rows:", total_rows)
rows_by_level = df_grammar['级别'].value_counts()
print(rows_by_level)


# =========================
# 2. 数据质量检查
# =========================

print("\n\n" + "*" * 5 + "查重" + "*" * 5)

# 查看前几行，快速确认列结构是否正常
print(df_words.head())
print(df_characters.head())

# ---- 2.1 查重：同一个“词语”是否出现多次 ----
duplicates = df_words[df_words.词语.duplicated(keep=False)].sort_values("词语")
print(duplicates)
# 备注：目前只有约 139 个重复 entry，多为一词多义/多词性或跨级别重复，问题不大。

# ---- 2.1 查重：同一个“汉字”是否出现多次 ----
duplicates = df_characters[df_characters.汉字.duplicated(keep=False)].sort_values("汉字")
print(duplicates)

# =========================
# 3. 词性字段检查
# =========================

print("\n\n" + "*" * 5 + "词性检查" + "*" * 5)

pos_tags = df_words["词性"].unique()
pos_tags_set = set()

# 将“词性”列中的复合标签（例如“名、动”）拆开，收集所有子标签
for tag in pos_tags:
    if not isinstance(tag, str):
        print('Non-string 词性 tag:', tag)
        continue
    for subtag in tag.split("、"):
        pos_tags_set.add(subtag.strip())

print("Unique 词性 tags:", pos_tags_set)
# 例：{'介', '动', '形', '量', '名', '数', '代', '连', '前缀', '拟声', '叹', '副', '助', '后缀'}


# ---- 3.1 查找缺失词性的词 ----
nan_pos_words = df_words[df_words["词性"].isna()]
print("Words with nan 词性:")
print(nan_pos_words)
# 目前约有 1359 个词没有标注“词性”，例如：做饭，做生意，爱国，挨家挨户 等。
# 这些词多为固定搭配或短语，后续可考虑自动补标或人工补标。


# =========================
# 4. 分词 + 词性标注示例
# =========================

# 简单调用一次，确认 tokenize_with_pos 行为是否符合预期
tokenize_with_pos("我喜欢做饭和做生意")


# =========================
# 5.  语法正则表达式检查
# =========================

print("\n\n" + "*" * 5 + "语法正则表达式检查" + "*" * 5)
print(df_grammar.head())

# =========================
# 6.  分级测试示例
# =========================
print("\n\n" + "*" * 5 + "分级测试示例" + "*" * 5)
level = classify_hsk_level_of_text("我觉得大家都该有高远的理想，闯出一片事业。我爱学习人工智能。它是未来的发展方向！")
print("HSK Level Classification Result:", level)


# =========================
# 7.  sample news 等级测试
# =========================
print("\n\n" + "*" * 5 + "Sample News 等级测试" + "*" * 5)

json_file = "news_leveled_26_01_08.json"
with open(json_file, "r", encoding="utf-8") as f:
    articles = json.load(f)

for article in articles:
    title = article.get("title", "")
    content = article.get("content", "")
    full_text = title + "\n" + content
    print(f"Title: {title}\n")
    print(f"target level: {article.get('level', 'N/A')}\n")
    level = classify_hsk_level_of_text(full_text)
    print("HSK Level Classification Result:", level, end="\n")
    print("Full Text:", full_text, end="\n\n")