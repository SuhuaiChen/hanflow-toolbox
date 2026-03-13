import os
import jwt
from jwt import PyJWKClient
from fastapi import HTTPException, Request

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID", "hanflow-reading-app")
FIREBASE_JWKS_URL = "https://www.googleapis.com/service_accounts/v1/jwk/securetoken@system.gserviceaccount.com"

_jwk_client = PyJWKClient(FIREBASE_JWKS_URL)


def verify_firebase_token(token: str) -> dict:
    try:
        signing_key = _jwk_client.get_signing_key_from_jwt(token).key
        claims = jwt.decode(
            token,
            signing_key,
            algorithms=["RS256"],
            audience=FIREBASE_PROJECT_ID,
            issuer=f"https://securetoken.google.com/{FIREBASE_PROJECT_ID}",
        )
        return claims
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid or expired Firebase token")


async def get_firebase_user_id(request: Request) -> str:
    auth = request.headers.get("authorization") or ""
    if not auth.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = auth.split(" ", 1)[1].strip()
    claims = verify_firebase_token(token)
    uid = claims.get("sub") or claims.get("user_id")
    if not uid:
        raise HTTPException(status_code=401, detail="Missing user ID in token")
    return uid
