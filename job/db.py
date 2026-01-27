from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
from datetime import datetime, timezone
import os
from hsk_data import load_hsk_words
from translate import llm_generate_en_meanings, llm_translate_es_pt_from_en
from cedict import cedict_lookup_en, cedict_lookup_pinyin, load_cedict_simplified
from nlp import classify_word_level, to_pinyin
from pinyin_convert import pinyin_tone_to_num
from oss import attach_tts_audio
from datetime import datetime, timezone

try:
    load_dotenv()
except Exception:
    pass

def create_mongo_client() -> MongoClient:
    # Create a new client and connect to the server
    client = MongoClient(os.environ["MONGODB_URI"], server_api=ServerApi('1'))
    # Send a ping to confirm a successful connection
    try:
        client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as e:
        print(e)
    return client

def ensure_dict_entry(word: str, col, lexicon, hsk_words, s3) -> dict:
    doc = col.find_one({"_id": word})

    # Cache hit: already processed
    if doc and "hskLevel" in doc:
        print(f"Cache hit for word '{word}'")
        return doc

    # Cache miss or incomplete doc
    doc = doc or {"_id": word}
    doc.setdefault("meanings", {})
    doc.setdefault("source", {})

    pinyin_dic = cedict_lookup_pinyin(word, lexicon)

    if pinyin_dic:  
        doc.setdefault("pinyin", pinyin_dic.lower())
        doc["source"]["pinyin"] = "cedict"
    else:
        doc.setdefault("pinyin", pinyin_tone_to_num(to_pinyin(word)))
        doc["source"]["pinyin"] = "nlp"

    m = doc["meanings"]

    # 1) HSK level (computed once)
    doc["hskLevel"] = classify_word_level(word, hsk_words)
    doc["source"]["hskLevel"] = "local"

    # 2) English meanings
    if not m.get("en"):
        en = cedict_lookup_en(word, lexicon)
        if not en:
            en = llm_generate_en_meanings(word)
            doc["source"]["en"] = "llm"
        else:
            doc["source"]["en"] = "cedict"
        m["en"] = en

    # 3) ES/PT meanings
    if not m.get("es") or not m.get("pt"):
        trans = llm_translate_es_pt_from_en(m["en"])
        m.setdefault("es", trans["es"])
        m.setdefault("pt", trans["pt"])
        doc["source"]["espt"] = "llm"
    
    audio_url = attach_tts_audio(text=word, s3=s3)
    doc.setdefault("audio", audio_url)

    doc["updatedAt"] = datetime.now(timezone.utc)
    col.replace_one({"_id": word}, doc, upsert=True)
    return doc

if __name__ == "__main__":
    client = create_mongo_client()
    db = client["core"]
    vocab_col = db["dict"]
    test_word = "中文"
    # lexicon = load_cedict_simplified("data/cedict_ts.u8")
    # entry = ensure_dict_entry(test_word, vocab_col, lexicon, hsk_words=load_hsk_words())
    # print(entry)