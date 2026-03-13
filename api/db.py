import os
from pymongo import MongoClient
from dotenv import load_dotenv
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()  # no-op if .env missing
except Exception:
    pass  # ignore if .env missing, we'll just fail later if MONGODB_URI not set

_MONGO_CLIENT: Optional[MongoClient] = None
def get_mongo_client() -> MongoClient:
    global _MONGO_CLIENT
    if _MONGO_CLIENT is None:
        import ssl, certifi
        _MONGO_CLIENT = MongoClient(
            os.environ["MONGODB_URI"],
            tls=True,
            tlsCAFile=certifi.where()
        )
    return _MONGO_CLIENT

