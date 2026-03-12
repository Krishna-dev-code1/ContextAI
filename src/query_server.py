# src/query_server.py

try:
    from .config import (
        GEMINI_KEY,
        ENV,
        INGEST_API_KEY,
        ALLOWED_ORIGINS,
        validate_env,
    )
except Exception:
    from config import (
        GEMINI_KEY,
        ENV,
        INGEST_API_KEY,
        ALLOWED_ORIGINS,
        validate_env,
    )

# Qdrant RAG helpers
# package-relative import preferred (works with `uvicorn src.main:app`)
# fallback to top-level import for direct script runs.
try:
    from .rag_core import embed_query, search_vectors
    try:
        from .rag_core import refresh_vector_store
    except Exception:
        def refresh_vector_store():
            return None
except Exception:
    from rag_core import embed_query, search_vectors
    try:
        from rag_core import refresh_vector_store
    except Exception:
        def refresh_vector_store():
            return None

VECTOR_BACKEND = "qdrant"

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.responses import FileResponse
import traceback
import time
import random
import subprocess
import sys
import os
import numpy as np

validate_env()
app = FastAPI()

# -----------------------
# Allow browser CORS from common local dev ports
# -----------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Define Project Root FIRST ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# --- Serve the Frontend ---
# This is the path to your 'web' folder
STATIC_DIR = os.path.join(PROJECT_ROOT, "web")

@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
# --- End of Frontend ---

# Path to file storing the last ingested source URL
LAST_SOURCE_PATH = os.path.join(PROJECT_ROOT, ".last_ingested_source.txt")

# Pydantic model for query payload
class Query(BaseModel):
    q: str
    k: int = 4

# -----------------------
# Helper: read last ingested source (returns None if not set)
# -----------------------
def read_last_source():
    try:
        if os.path.exists(LAST_SOURCE_PATH):
            with open(LAST_SOURCE_PATH, "r", encoding="utf-8") as f:
                val = f.read().strip()
                return val or None
    except Exception as e:
        print("[warn] failed to read last_source file:", e)
    return None

# -----------------------
# Ingest endpoint (background)
# -----------------------
@app.post("/ingest")
def ingest_url(body: dict, background_tasks: BackgroundTasks):
    """
    Kick off src/ingest.py in the background using the same Python interpreter.
    Body: {"url": "https://...", "key": "..."} in production.
    """

    # --- Production protection ---
    if ENV == "production":
        if INGEST_API_KEY == "" or INGEST_API_KEY != body.get("key"):
            raise HTTPException(status_code=403, detail="Invalid ingest key")

    _url = body.get("url") if isinstance(body, dict) else None
    if not _url:  # <-- This is the bug
        raise HTTPException(status_code=400, detail="Missing 'url' in request body")

    url = _url.strip()
    # --- Save last ingested source ---
    try:
        with open(LAST_SOURCE_PATH, "w", encoding="utf-8") as f:
            f.write(url)
    except Exception as e:
        print("[ingest-bg] warning: failed to write last_ingested_source file:", e)

    # --- Background task runner ---
    def run_ingest(u: str):
        try:
            project_root = PROJECT_ROOT
            pyexe = sys.executable
            args = [pyexe, "-m", "src.ingest", u]
            print("[ingest-bg] running:", args, "cwd=", project_root)

            proc = subprocess.run(
                args,
                check=False,
                capture_output=True,
                text=True,
                cwd=project_root
            )

            print("[ingest-bg] exitcode:", proc.returncode)
            if proc.stdout:
                print("[ingest-bg] stdout:\n", proc.stdout[:4000])
            if proc.stderr:
                print("[ingest-bg] stderr:\n", proc.stderr[:4000])

        finally:
            try:
                refresh_vector_store()
                print("[ingest-bg] called refresh_vector_store()")
            except Exception:
                pass

    background_tasks.add_task(run_ingest, url)
    return {"status": "started", "url": url}


# -----------------------
# Query endpoint (RAG)
# -----------------------
@app.post("/query")
def answer_question(body: Query):
    query = body.q.strip()

    # --- basic safety ---
    if len(query) < 2:
        raise HTTPException(status_code=400, detail="Query too short")
    if len(query) > 2000:
        raise HTTPException(status_code=400, detail="Query too long")

    try:
        # 1) embed user query
        q_emb = embed_query(query)

        # 2) optional filtering to last ingested source
        last_source = read_last_source()
        k = min(body.k, 10)

        # 3) retrieve top-k
        try:
            results = search_vectors(q_emb, k=k, source=last_source)
        except Exception as e:
            tb = traceback.format_exc()
            print("[query] search_vectors failed:", e, tb)
            raise HTTPException(status_code=500, detail="Vector DB search failed")

        if not results:
            return {"answer": "I don't know from the provided context."}

        # unpack
        docs = [r.get("document", "") for r in results]
        metas = [r.get("metadata", {}) for r in results]

        # build clean context (NO distances)
        context = ""
        for i, (doc, md) in enumerate(zip(docs, metas)):
            src = md.get("source") or md.get("url") or f"chunk-{i}"
            context += f"[{i}] Source: {src}\n{doc}\n\n"

        prompt = f"""Use ONLY the following context to answer the question.
If the answer is not present, reply exactly:
"I don't know from the provided context."

CONTEXT:
{context}

QUESTION:
{query}
"""

        # 4) generate with Gemini
        if GEMINI_KEY:
            try:
                from google import genai as _genai
                client = _genai.Client(api_key=GEMINI_KEY)

                resp = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )

                answer = getattr(resp, "text", None) or str(resp)
                return {"answer": answer}

            except Exception:
                    if ENV == "production":
                        return {"answer": "LLM backend unavailable."}
                    else:
                        # dev fallback: show the raw context so you can debug
                        return {"answer": context, "note": "LLM generation failed."}



        else:
            # no LLM backend → return context directly
            return {"answer": context}

    except HTTPException:
        raise
    except Exception as e:
        print("=== /query ERROR ===", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Server error")

# -----------------------
# Debug endpoint (Qdrant-first, safe)
# -----------------------
@app.post("/debug_query")
def debug_query(body: Query):
    try:
        q_emb = embed_query(body.q)
        last_source = read_last_source()
        k = body.k

       
        if VECTOR_BACKEND == "qdrant":
            try:
                results = search_vectors(q_emb, k=k, source=last_source)
                # normalize into list of dicts
                normalized = []
                for i, r in enumerate(results):
                    normalized.append({
                        "rank": i,
                        "id": r.get("id") or (r.get("metadata") or {}).get("chunk_id"),
                        "text_snippet": (r.get("document") or (r.get("metadata") or {}).get("_text") or "")[:4000],
                        "document": r.get("document", ""),
                        "metadata": r.get("metadata", None),
                        "distance": r.get("distance", None),
                    })
                return {"retrieved": normalized, "raw_result_keys": ["document","metadata","distance"]}
            except Exception as e:
                tb = traceback.format_exc()
                print("[debug_query] search_vectors failed:", e, tb)
                return {"retrieved": [], "warning": "search_vectors failed", "error": str(e)}
        
    except Exception as e:
        tb = traceback.format_exc()
        print("=== ERROR debug_query ===")
        print(tb)
        raise HTTPException(status_code=500, detail=f"Debug query failed: {e}\n\nTrace:\n{tb}")
