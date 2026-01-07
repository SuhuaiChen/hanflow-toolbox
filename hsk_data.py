import pandas as pd
import re

df_characters = pd.read_csv('data/characters.csv')
df_words = pd.read_csv('data/words.csv')
df_hand_written_characters = pd.read_csv('data/characters_hand_written.csv')
df_grammar = pd.read_csv('data/grammar_with_regex.csv')

hsk_words = {row.词语: row.级别 for row in df_words.itertuples()} # TODO 之后最好优化一下，有重复词语（不同级别）
hsk_characters = {row.汉字: row.级别 for row in df_characters.itertuples()} # TODO 之后最好优化一下，有重复词语（不同级别）
hsk_grammar_with_regex = {row.正则表达式: (row.级别, row.语法内容) for row in df_grammar.itertuples() if pd.notna(row.正则表达式)} # TODO 存在104个语法点没有正则表达式，需要后续补充完善，已有的正则有些也要完善，语法的检测需要句法依存，光普通regex是不够的