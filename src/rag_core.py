# src/rag_core.py
"""
Stable Qdrant-backed rag_core.

Functions exported:
- embed_passages(passages: List[str]) -> List[List[float]]
- embed_query(text: str) -> List[float]
- upsert_chunks(ids, documents, embeddings, metadatas)
- search_vectors(query_vector, k=4, source=None) -> list[{"document","metadata","distance","id"}]
- delete_by_ids(ids)
- delete_by_source(source)
- get_all()
"""

import os
import time
import traceback
import uuid
from typing import List, Optional, Dict, Any

# Qdrant client
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
# qdrant exceptions (best-effort import)
try:
    from qdrant_client.http.exceptions import UnexpectedResponse
except Exception:
    UnexpectedResponse = Exception

# numpy only used in a couple places, keep import in case needed
import numpy as np

# Defensive import for Google GenAI SDK
try:
    from google import genai
except Exception as e:
    genai = None
    GENAI_IMPORT_ERROR = e

# Config - prefer project config.py, fallback to env
try:
    from config import GEMINI_KEY, QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION, QDRANT_DISTANCE
except Exception:
    GEMINI_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_KEY") or None
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "wiki_rag")
    QDRANT_DISTANCE = os.getenv("QDRANT_DISTANCE", "Cosine")

# Create Qdrant client (best-effort)
_qdrant = None
try:
    if QDRANT_API_KEY:
        _qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    else:
        _qdrant = QdrantClient(url=QDRANT_URL)
except Exception as e:
    _qdrant = None
    print("[warn] qdrant client init failed:", e)

# Create genai client wrapper if possible
_genai_client = None
if genai is not None and GEMINI_KEY:
    try:
        # new-style client
        _genai_client = genai.Client(api_key=GEMINI_KEY)
    except Exception:
        try:
            # older style configure + client
            genai.configure(api_key=GEMINI_KEY)
            _genai_client = genai.Client()
        except Exception:
            _genai_client = None

# default embed model name; change if needed
EMBED_MODEL = os.getenv("EMBED_MODEL", "models/text-embedding-004")

# -----------------------
# Qdrant helpers
# -----------------------
def _ensure_collection(dim: int):
    if _qdrant is None:
        raise RuntimeError("Qdrant client not configured. Set QDRANT_URL and QDRANT_API_KEY.")
    try:
        _qdrant.get_collection(collection_name=QDRANT_COLLECTION)
        
        # --- ADD THIS BLOCK ---
        # Ensure the payload index for "source" exists
        try:
            _qdrant.create_payload_index(
                collection_name=QDRANT_COLLECTION,
                field_name="source",
                field_schema=qmodels.PayloadSchemaType.KEYWORD
            )
            print("[info] Ensured Qdrant payload index exists for 'source'.")
        except Exception as e:
            # Handle cases where index might already exist in a conflicting way, etc.
            print(f"[warn] Could not create payload index: {e}")
            pass
        # --- END OF BLOCK ---

        return
    except Exception:
        distance = qmodels.Distance.COSINE if QDRANT_DISTANCE.lower().startswith("cos") else qmodels.Distance.EUCLID
        # Try recreate_collection, fallback to create_collection if signatures differ between versions
        try:
            _qdrant.recreate_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=qmodels.VectorParams(size=dim, distance=distance),
            )
            return
        except TypeError:
            # signature mismatch; try create_collection
            try:
                _qdrant.create_collection(
                    collection_name=QDRANT_COLLECTION,
                    vectors_config=qmodels.VectorParams(size=dim, distance=distance),
                )
                return
            except Exception as e:
                # final attempt: call recreate with minimal args (for some clients)
                try:
                    _qdrant.recreate_collection(QDRANT_COLLECTION, qmodels.VectorParams(size=dim, distance=distance))
                    return
                except Exception as e2:
                    raise

# -----------------------
# Gemini / GenAI embedding wrapper (robust)
# -----------------------
def _parse_embedding_response(resp_item: Any) -> List[float]:
    """
    Given an embedding item (various SDK shapes), return a list[float] embedding.
    """
    # dict shapes
    if isinstance(resp_item, dict):
        for key in ("embedding", "vector", "values"):
            v = resp_item.get(key)
            if v is not None:
                if isinstance(v, dict):
                    for sub in ("values", "vector", "embedding"):
                        if sub in v:
                            return list(v[sub])
                    if isinstance(v, (list, tuple)):
                        return list(v)
                if isinstance(v, (list, tuple)):
                    return list(v)
    # object shapes (SDK objects)
    try:
        if hasattr(resp_item, "embedding"):
            cand = getattr(resp_item, "embedding")
            if isinstance(cand, (list, tuple)):
                return list(cand)
            if hasattr(cand, "values"):
                return list(getattr(cand, "values"))
        if hasattr(resp_item, "values"):
            return list(getattr(resp_item, "values"))
        if hasattr(resp_item, "vector"):
            return list(getattr(resp_item, "vector"))
    except Exception:
        pass

    raise RuntimeError(f"Unable to parse embedding item shape: {repr(resp_item)[:400]}")

def _call_gemini_embeddings(inputs: List[str], model: str = EMBED_MODEL, max_retries: int = 3, sleep: float = 0.1) -> List[List[float]]:
    """
    Robust wrapper for generating embeddings using google.genai (handles multiple SDK shapes).
    Retries on transient errors.
    """
    if not GEMINI_KEY:
        raise RuntimeError("GEMINI_KEY not set; set GEMINI_API_KEY in .env or config.py")

    if genai is None and _genai_client is None:
        raise RuntimeError(f"google.genai import failed: {GENAI_IMPORT_ERROR}")

    all_vecs: List[List[float]] = []

    batch_size = 32
    for start in range(0, len(inputs), batch_size):
        batch = inputs[start:start + batch_size]
        last_exc = None
        resp = None

        for attempt in range(1, max_retries + 1):
            try:
                if _genai_client is not None and hasattr(_genai_client, "models") and hasattr(_genai_client.models, "embed_content"):
                    resp = _genai_client.models.embed_content(model=model, contents=batch)
                elif _genai_client is not None and hasattr(_genai_client.models, "embeddings"):
                    resp = _genai_client.models.embeddings.create(model=model, input=batch)
                else:
                    if hasattr(genai, "embed_content"):
                        try:
                            resp = genai.embed_content(model=model, contents=batch)
                        except TypeError:
                            resp = genai.embed_content(model=model, input=batch)
                    elif hasattr(genai, "embeddings") and hasattr(genai.embeddings, "create"):
                        resp = genai.embeddings.create(model=model, input=batch)
                    else:
                        raise RuntimeError("No supported genai embeddings API found on installed SDK.")
            except TypeError as te:
                last_exc = te
            except Exception as e:
                last_exc = e

            if resp is not None:
                try:
                    emb_items = None
                    if hasattr(resp, "embeddings"):
                        emb_items = resp.embeddings
                    elif isinstance(resp, dict) and "embeddings" in resp:
                        emb_items = resp["embeddings"]
                    elif isinstance(resp, dict) and "data" in resp:
                        emb_items = resp["data"]
                    elif isinstance(resp, list):
                        emb_items = resp
                    elif hasattr(resp, "data"):
                        emb_items = getattr(resp, "data")
                    else:
                        emb_items = resp

                    batch_vecs: List[List[float]] = []
                    for item in emb_items:
                        if isinstance(item, dict) and "embedding" in item and isinstance(item["embedding"], (list, tuple)):
                            batch_vecs.append(list(item["embedding"]))
                        else:
                            batch_vecs.append(_parse_embedding_response(item))

                    if len(batch_vecs) != len(batch):
                        raise RuntimeError(f"Unexpected embeddings count: got {len(batch_vecs)} for input batch {len(batch)}")
                    all_vecs.extend(batch_vecs)
                    break
                except Exception as e:
                    last_exc = e
                    resp = None

            time.sleep(sleep * attempt)

        if resp is None and last_exc is not None:
            raise RuntimeError(f"Gemini embeddings failed after {max_retries} retries. Last error: {last_exc}")

    return all_vecs

# Public wrappers
def embed_passages(passages: List[str]) -> List[List[float]]:
    if not passages:
        return []
    return _call_gemini_embeddings(passages, model=EMBED_MODEL)

def embed_query(text: str) -> List[float]:
    v = _call_gemini_embeddings([text], model=EMBED_MODEL)
    return v[0]

# -----------------------
# Qdrant upsert / search wrappers
# -----------------------
def _to_uuid(id_str: str) -> uuid.UUID:
    """Convert an arbitrary string id into a deterministic UUIDv5 (namespace=URL)."""
    return uuid.uuid5(uuid.NAMESPACE_URL, id_str)

def upsert_chunks(ids: List[str], documents: List[str], embeddings: List[List[float]], metadatas: List[Dict[str,Any]]):
    if _qdrant is None:
        raise RuntimeError("Qdrant client is not configured.")
    if not embeddings:
        return
    dim = len(embeddings[0])
    _ensure_collection(dim)
    points = []
    for i, vec in enumerate(embeddings):
        payload = metadatas[i] if i < len(metadatas) else {}
        payload["_text"] = documents[i] if i < len(documents) else ""
        # convert id to UUID so Qdrant accepts it (UUID is supported server-side)
        try:
             # Convert the UUID object to a string for Qdrant's Pydantic model
            point_id = str(_to_uuid(ids[i]))
        except Exception:
            point_id = ids[i]
        points.append(qmodels.PointStruct(id=point_id, vector=vec, payload=payload))
    _qdrant.upload_points(collection_name=QDRANT_COLLECTION, points=points, wait=True)

def search_vectors(query_vector: List[float], k: int = 4, source: Optional[str] = None) -> List[Dict[str,Any]]:
    """
    Return list of dicts: {"document": str, "metadata": dict, "distance": float, "id": ...}
    Tries server-side payload filtering; if Qdrant rejects the filter (missing index), falls back to
    retrieving extra candidates and filtering client-side.
    """
    if _qdrant is None:
        raise RuntimeError("Qdrant client not configured.")
    _ensure_collection(len(query_vector))

    # if no source filter requested -> simple search
    if not source:
        res = _qdrant.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=query_vector,
            limit=k,
            with_payload=True,
            with_vectors=False,
        )
        out = []
        for hit in res:
            payload = hit.payload or {}
            text = payload.get("_text", "")
            meta = {kk: vv for kk, vv in payload.items() if kk != "_text"}
            score = getattr(hit, "score", None)
            distance = None
            if score is not None:
                try:
                    distance = 1.0 - float(score)
                except Exception:
                    distance = float(score)
            out.append({"document": text, "metadata": meta, "distance": distance, "id": getattr(hit, "id", None)})
        return out

    # Try server-side filter first
    try:
        filter_obj = qmodels.Filter(
            must=[qmodels.FieldCondition(key="source", match=qmodels.MatchValue(value=source))]
        )
        res = _qdrant.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=query_vector,
            limit=k,
            with_payload=True,
            with_vectors=False,
            query_filter=filter_obj
        )
        out = []
        for hit in res:
            payload = hit.payload or {}
            text = payload.get("_text", "")
            meta = {kk: vv for kk, vv in payload.items() if kk != "_text"}
            score = getattr(hit, "score", None)
            distance = None
            if score is not None:
                try:
                    distance = 1.0 - float(score)
                except Exception:
                    distance = float(score)
            out.append({"document": text, "metadata": meta, "distance": distance, "id": getattr(hit, "id", None)})
        return out
    except UnexpectedResponse as ue:
        # Qdrant rejected filter (likely index missing). Fall back to client-side filtering.
        # print warning and continue to fallback below.
        print("[warn] qdrant rejected server-side payload filter, falling back to client-side filter:", ue)
    except Exception as e:
        # if any other error occurs, also fall back to client-side (but log)
        print("[warn] server-side search with filter failed, falling back to client-side:", e)

    # Client-side fallback: fetch more candidates and filter locally
    fetch_k = max(k * 6, k + 50)  # larger fetch to increase chance of matches
    try:
        res = _qdrant.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=query_vector,
            limit=fetch_k,
            with_payload=True,
            with_vectors=False,
        )
    except Exception as e:
        # if even this fails, raise to caller
        raise

    out = []
    for hit in res:
        payload = hit.payload or {}
        hit_source = payload.get("source") or payload.get("url") or None
        if hit_source == source:
            text = payload.get("_text", "")
            meta = {kk: vv for kk, vv in payload.items() if kk != "_text"}
            score = getattr(hit, "score", None)
            distance = None
            if score is not None:
                try:
                    distance = 1.0 - float(score)
                except Exception:
                    distance = float(score)
            out.append({"document": text, "metadata": meta, "distance": distance, "id": getattr(hit, "id", None)})
            if len(out) >= k:
                break
    return out

def delete_by_ids(ids: List[str]):
    if _qdrant is None:
        raise RuntimeError("Qdrant client not configured.")
    # convert any string ids to UUIDs if necessary
    conv = []
    for i in ids:
        try:
            conv.append(_to_uuid(i))
        except Exception:
            conv.append(i)
    _qdrant.delete(collection_name=QDRANT_COLLECTION, points_selector=qmodels.PointIdsList(points=conv))

def delete_by_source(source: str):
    if _qdrant is None:
        raise RuntimeError("Qdrant client not configured.")
    flt = qmodels.Filter(
        must=[qmodels.FieldCondition(key="source", match=qmodels.MatchValue(value=source))]
    )
    # Try FilterSelector or FiltersSelector depending on client version
    selector = None
    try:
        selector = qmodels.FilterSelector(filters=flt)
    except Exception:
        if hasattr(qmodels, "FiltersSelector"):
            selector = qmodels.FiltersSelector(filters=flt)
    if selector is not None:
        try:
            _qdrant.delete(collection_name=QDRANT_COLLECTION, points_selector=selector)
            return
        except Exception as e:
            print("[warn] delete by selector failed, falling back to scan-delete:", e)

    # Fallback: search for matching payloads and delete by ids
    try:
        # Use a payload-only search. If client's API requires a vector, this may raise; handle gracefully.
        try:
            resp = _qdrant.search(
                collection_name=QDRANT_COLLECTION,
                query_vector=[0.0] * 1,  # best-effort: some qdrant RPCs require a vector - may fail
                limit=1000,
                with_payload=True,
                query_filter=flt
            )
            ids = [pt.id for pt in resp]
            if ids:
                _qdrant.delete(collection_name=QDRANT_COLLECTION, points_selector=qmodels.PointIdsList(points=ids))
                return
        except Exception:
            # Last-resort: use scroll and filter payloads server-side (slower)
            resp = _qdrant.scroll(collection_name=QDRANT_COLLECTION, limit=10000, with_payload=True)
            # resp is a tuple: (records, next_page_offset). We need to iterate over records.
            records = resp[0] if isinstance(resp, tuple) else resp
            ids = []
            for pt in records:
                payload = pt.payload or {}
                if (payload.get("source") or payload.get("url")) == source:
                    ids.append(pt.id)
            if ids:
                _qdrant.delete(collection_name=QDRANT_COLLECTION, points_selector=qmodels.PointIdsList(points=ids))
                return
    except Exception as e:
        # If all deletion strategies fail, raise so caller knows
        raise

def get_all() -> Dict[str, List]:
    if _qdrant is None:
        raise RuntimeError("Qdrant client not configured.")
    try:
        resp = _qdrant.scroll(collection_name=QDRANT_COLLECTION, limit=10000, with_payload=True, with_vectors=True)
        records = resp[0] if isinstance(resp, tuple) else resp
        ids = []
        docs = []
        metas = []
        embs = []
        for pt in records:
            ids.append(pt.id)
            payload = pt.payload or {}
            docs.append(payload.get("_text",""))
            metas.append({k:v for k,v in payload.items() if k != "_text"})
            embs.append(pt.vector if hasattr(pt, "vector") else None)
        return {"ids": [ids], "documents": [docs], "metadatas": [metas], "embeddings": [embs]}
    except Exception as e:
        print("[warn] get_all failed:", e)
        return {"ids":[[]], "documents":[[]], "metadatas":[[]], "embeddings":[[]]}
