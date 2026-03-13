import os
from typing import Any, Dict, Optional, List
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from pymongo import DESCENDING
from pydantic import BaseModel, EmailStr, Field

from db import get_mongo_client
from auth.auth import get_firebase_user_id
from ai import router as ai_router


# ----------------------------
# App
# ----------------------------

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

app = FastAPI(title="HanFlow Reading API")
app.include_router(ai_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("CORS_ORIGIN", "*")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# DB helpers
# ----------------------------

def articles_col():
    c = get_mongo_client()
    return c["core"]["articles"]

def sentences_col():
    c = get_mongo_client()
    return c["core"]["sentence"]

def feedback_col():
    c = get_mongo_client()
    return c["core"]["feedback"]

def user_data_col():
    c = get_mongo_client()
    return c["core"]["user_data"]


# ----------------------------
# Access helpers
# ----------------------------

async def get_firebase_user_id_optional(request: Request) -> Optional[str]:
    auth = request.headers.get("authorization") or ""
    if auth.lower().startswith("bearer "):
        try:
            return await get_firebase_user_id(request)
        except Exception as e:
            print(">>> Auth error:", str(e))
    return None


# ----------------------------
# Query helpers
# ----------------------------

def sanitize_limit(limit: int) -> int:
    return max(1, min(50, limit))

def _article_projection_list() -> Dict[str, int]:
    return {
        "_id": 1,
        "id": 1,
        "title": 1,
        "titleTranslations": 1,
        "titleAudio": 1,
        "excerpt": 1,
        "excerptTranslations": 1,
        "excerptAudio": 1,
        "level": 1,
        "date": 1,
    }

def _normalize_items(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    items = []
    for d in docs:
        item = {k: v for k, v in d.items() if k != "_id"}
        item["id"] = d.get("id") or d.get("_id")
        items.append(item)
    return items



# ----------------------------
# Endpoints
# ----------------------------

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/me")
async def me(request: Request) -> Dict[str, Any]:
    firebase_uid = await get_firebase_user_id_optional(request)
    return {"isPro": True, "firebaseUid": firebase_uid}

@app.get("/articles")
def list_articles(
    level: Optional[str] = Query(default=None),
    date: Optional[str] = Query(default=None),
    limit: int = Query(5, ge=1, le=50),
    cursor: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    col = articles_col()
    limit = sanitize_limit(limit)

    q: Dict[str, Any] = {}
    if level:
        q["level"] = level
    if date:
        q["date"] = date
    if cursor:
        q["_id"] = {"$lt": cursor}

    docs = list(col.find(q, _article_projection_list()).sort("_id", DESCENDING).limit(limit))
    next_cursor = docs[-1].get("_id") if len(docs) == limit else None
    return {"items": _normalize_items(docs), "nextCursor": next_cursor, "date": q.get("date")}


@app.get("/articles/{article_id}")
def get_article(article_id: str) -> Dict[str, Any]:
    col = articles_col()
    doc = col.find_one({"_id": article_id}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="Not found")
    return doc


@app.get("/day_sentence")
def get_daily_sentence() -> Dict[str, Any]:
    col = sentences_col()

    # latest sentence by date
    doc = col.find({}, {"_id": 0, "tokens": 1, "translations": 1, "date": 1}).sort("date", DESCENDING).limit(1)
    docs = list(doc)
    if not docs:
        raise HTTPException(status_code=404, detail="No sentence found")

    d = docs[0]
    return {
        "tokens": d.get("tokens", []),
        "translations": d.get("translations", {}),
    }


class FeedbackIn(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    email: Optional[EmailStr] = None


class TokenIn(BaseModel):
    t: str
    pinyin: Optional[str] = None
    meanings: Optional[Dict[str, str]] = None
    type: Optional[str] = None
    hsk_level: Optional[str] = None
    audio: Optional[str] = None


class TokensPayload(BaseModel):
    tokens: List[TokenIn]


class ReadEntryIn(BaseModel):
    articleId: str
    title: str
    titleTranslation: Optional[str] = None
    level: str
    articleDate: str          # YYYY-MM-DD
    readAt: int               # ms epoch
    completedAt: Optional[int] = None  # ms epoch


class TestResultIn(BaseModel):
    articleId: str
    passed: bool
    score: int
    assistanceCount: int
    lastPassedAt: int         # ms epoch


@app.post("/feedback", status_code=204)
async def create_feedback(payload: FeedbackIn, request: Request):
    msg = payload.message.strip()
    if not msg:
        raise HTTPException(status_code=400, detail="message is required")

    doc = {
        "message": msg,
        "email": payload.email,
        "createdAt": datetime.now(timezone.utc),
        "userAgent": request.headers.get("user-agent"),
        "ip": request.client.host if request.client else None,
    }

    feedback_col().insert_one(doc)
    return None