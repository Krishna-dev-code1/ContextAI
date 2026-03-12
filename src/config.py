# config.py -- simple env wrapper used by the project
# Place this file in the project root (same folder as src/ and .env)

import os
from dotenv import load_dotenv

# load .env from project root (if present)
load_dotenv()

# canonical names used in the code
GEMINI_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_KEY") or ""
OPENAI_KEY = os.getenv("OPENAI_API_KEY") or ""
QDRANT_URL = os.getenv("QDRANT_URL") or ""
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or ""
ENV = os.getenv("ENV", "development").strip().lower()
INGEST_API_KEY = os.getenv("INGEST_API_KEY", "").strip()

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "").split(",")
ALLOWED_ORIGINS = [o.strip() for o in ALLOWED_ORIGINS if o.strip()]
if not ALLOWED_ORIGINS:
    ALLOWED_ORIGINS = ["*"]
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_collection").strip()

# small helper for boolean checks (optional)
def boolenv(v):
    return bool(v and str(v).strip())

def validate_env():
    if ENV == "production" and not INGEST_API_KEY:
        raise RuntimeError("INGEST_API_KEY missing in production")
    if not QDRANT_URL:
        raise RuntimeError("QDRANT_URL missing")
    if not QDRANT_API_KEY:
        raise RuntimeError("QDRANT_API_KEY missing")
    return True
