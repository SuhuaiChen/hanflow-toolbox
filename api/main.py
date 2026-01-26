import os
from typing import Any, Dict, Optional
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pymongo import DESCENDING, MongoClient

# ----------------------------
# App + DB
# ----------------------------

try:
    from dotenv import load_dotenv
    load_dotenv()  # no-op if .env missing
except Exception:
    pass

app = FastAPI(title="HanFlow Reading API")# local only: load .env if it exists

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("CORS_ORIGIN", "*")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_MONGO_CLIENT: Optional[MongoClient] = None


def get_mongo_client() -> MongoClient:
    global _MONGO_CLIENT
    if _MONGO_CLIENT is None:
        _MONGO_CLIENT = MongoClient(os.environ["MONGODB_URI"])
    return _MONGO_CLIENT


def articles_col():
    c = get_mongo_client()
    return c["core"]["articles"]


# ----------------------------
# Tier model (simple for now)
# ----------------------------

class Tier:
    FREE = "free"
    PRO = "pro"


def get_user_tier(x_hanflow_tier: Optional[str] = Header(default=None)) -> str:
    """
    For now: client sends a header:
      X-Hanflow-Tier: free | pro

    Later you replace this with real auth (JWT/session) and server-side tier lookup.
    """
    t = (x_hanflow_tier or Tier.FREE).strip().lower()
    if t not in (Tier.FREE, Tier.PRO):
        t = Tier.FREE
    return t


# ----------------------------
# Query helpers
# ----------------------------

def latest_date_for_query(col, q: Dict[str, Any]) -> Optional[str]:
    doc = col.find(q, {"date": 1}).sort("date", DESCENDING).limit(1)
    docs = list(doc)
    return docs[0]["date"] if docs else None


def sanitize_limit(limit: int) -> int:
    return max(1, min(50, limit))


# ----------------------------
# Endpoints
# ----------------------------

@app.get("/health")
def health():
    return {"ok": True}


@app.get("/articles")
def list_articles(
    tier: str = Depends(get_user_tier),
    level: Optional[str] = Query(default=None),
    date: Optional[str] = Query(default=None),   # YYYY-MM-DD (pro only)
    limit: int = Query(5, ge=1, le=50),
    cursor: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    col = articles_col()
    limit = sanitize_limit(limit)

    q: Dict[str, Any] = {}
    if level:
        q["level"] = level

    # FREE: force latest date, ignore date param
    if tier == Tier.FREE:
        latest = latest_date_for_query(col, q)
        if not latest:
            return {"items": [], "nextCursor": None, "date": None}
        q["date"] = latest
        active_date = latest
    else:
        # PRO: allow optional date filter
        if date:
            q["date"] = date
        active_date = q.get("date")

    # Cursor pagination: request next page "after" the last doc you got.
    # IMPORTANT: cursor only works correctly if frontend keeps the same filters.
    if cursor:
        q["_id"] = {"$lt": cursor}

    projection = {
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

    docs = list(col.find(q, projection).sort("_id", DESCENDING).limit(limit))
    next_cursor = docs[-1].get("_id") if len(docs) == limit else None

    items = [
        {
            **{k: v for k, v in d.items() if k != "_id"},
            "id": d.get("id") or d.get("_id"),
        }
        for d in docs
    ]

    return {"items": items, "nextCursor": next_cursor, "date": active_date}


@app.get("/articles/{article_id}")
def get_article(
    article_id: str,
    tier: str = Depends(get_user_tier),
) -> Dict[str, Any]:
    """
    Returns full annotated article with sentences/tokens.

    FREE tier:
      - only allowed if the article is from today's/latest date.

    PRO tier:
      - can access any article.
    """
    col = articles_col()
    doc = col.find_one({"_id": article_id}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="Not found")

    if tier == Tier.FREE:
        # must be from latest date (optionally could check "level" too)
        latest = latest_date_for_query(col, {})
        if latest and doc.get("date") != latest:
            raise HTTPException(status_code=403, detail="Free tier can only view today's articles")

    return doc
