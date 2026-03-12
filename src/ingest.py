# src/ingest.py
"""
Ingest script (Qdrant backend).
Usage:
    python src/ingest.py https://en.wikipedia.org/wiki/Artificial_intelligence

What it does:
  1. Fetches cleaned markdown via r.jina.ai proxy
  2. Chunks the text using utils.chunk_text
  3. Embeds chunks using rag_core.embed_passages (Gemini)
  4. Deletes any existing chunks for the same source (safe)
  5. Upserts chunks into Qdrant using rag_core.upsert_chunks
"""

import sys
import requests
import hashlib
import time
from typing import List

from .utils import chunk_text

# Import the rag_core functions we need (Qdrant wrapper)
from .rag_core import embed_passages, upsert_chunks, delete_by_source

FETCH_TIMEOUT = 30.0  # seconds


def fetch_markdown(url: str) -> str:
    """
    Fetch cleaned markdown via r.jina.ai proxy.
    Returns the text or raises requests.HTTPError on non-200.
    """
    proxy_url = "https://r.jina.ai/" + url
    print(f"[fetch] GET {proxy_url}")
    r = requests.get(proxy_url, timeout=FETCH_TIMEOUT)
    r.raise_for_status()
    return r.text


def ingest(url: str):
    """Full ingest pipeline for a single URL."""
    print("[1] Fetching article...")
    try:
        text = fetch_markdown(url)
    except Exception as e:
        print(f"[error] failed to fetch markdown for {url}: {e}")
        raise

    print("[2] Chunking article...")
    chunks: List[str] = chunk_text(text)
    print(f" -> {len(chunks)} chunks")

    if not chunks:
        print("[warn] no chunks produced, aborting ingest")
        return

    print("[3] Embedding chunks...")
    try:
        embeddings = embed_passages(chunks)
    except Exception as e:
        print(f"[error] embedding failed: {e}")
        raise

    # verify shapes
    if len(embeddings) != len(chunks):
        print(f"[warn] embeddings length ({len(embeddings)}) != chunks ({len(chunks)})")

    print("[4] Deleting any existing chunks for this source (safe)...")
    try:
        # best-effort: some rag_core implementations may not expose delete_by_source; handle gracefully
        try:
            delete_by_source(url)
            print("[4] previous chunks (if any) removed for this source")
        except NameError:
            # function not provided; continue without delete
            print("[4] delete_by_source not available in rag_core; continuing")
        except Exception as e:
            # non-fatal
            print(f"[warn] delete_by_source failed (continuing): {e}")
    except Exception:
        pass

    print("[5] Upserting to vector DB (Qdrant)...")
    try:
        # create deterministic UUID ids for Qdrant (UUID required for cloud instances)
        import uuid
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{url}#chunk-{i}")) for i in range(len(chunks))]
        metadatas = [{"source": url, "chunk_id": i} for i in range(len(chunks))]

        # call rag_core.upsert_chunks(ids, documents, embeddings, metadatas)
        upsert_chunks(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)

        print(f"[ingest] upserted {len(ids)} chunks (ids prefix={ids[0][:8]}...)")
    except Exception as e:
        print(f"[error] upsert_chunks failed: {e}")
        raise

    # Windows-safe success print (emoji fallback)
    try:
        print("✅ DONE — Article ingested & indexed successfully!")
    except Exception:
        print("DONE — Article ingested & indexed successfully!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python src/ingest.py https://en.wikipedia.org/wiki/Apple")
        sys.exit(1)

    start = time.time()
    try:
        ingest(sys.argv[1])
    except Exception as e:
        print(f"[fatal] ingest failed: {e}")
        sys.exit(2)
    else:
        print(f"[ok] finished in {time.time() - start:.2f}s")
        sys.exit(0)