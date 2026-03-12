# src/utils.py
import re
from typing import List

try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except Exception:
    _TIKTOKEN_AVAILABLE = False
    # small, non-fatal notification for devs
    print("[utils] tiktoken not available — using heuristic fallback for chunking")


def _clean_text_for_chunking(text: str) -> str:
    """
    Minimal cleaning:
    - remove fenced code blocks and inline code
    - strip simple HTML tags
    - collapse many newlines and whitespace
    """
    # remove fenced code ```...```
    text = re.sub(r"```[\s\S]*?```", " ", text)
    # remove inline code `...`
    text = re.sub(r"`[^`]*`", " ", text)
    # remove simple HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # collapse multiple newlines to two newlines (paragraph separator)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    # collapse whitespace (spaces, tabs, etc.)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    return text.strip()


def _trim_to_sentence_boundary(text: str) -> str:
    """
    Try to avoid chopping mid-sentence by trimming to the last sentence end
    (., ?, !) within the chunk. If none found, return original text.
    """
    if not text:
        return text
    # find last sentence-like boundary
    candidates = [text.rfind(". "), text.rfind("? "), text.rfind("! "), text.rfind("\n")]
    last_pos = max(candidates)
    # require the sentence end to be reasonably inside the chunk (heuristic)
    if last_pos > 0 and last_pos >= int(len(text) * 0.4):
        return text[: last_pos + 1].strip()
    return text.strip()


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
    """
    Chunk text into overlapping token windows suitable for embeddings.

    Args:
        text: raw text (markdown/html ok).
        chunk_size: number of tokens per chunk (approx).
        overlap: overlap in tokens between consecutive chunks.

    Returns:
        list of cleaned text chunks (strings).
    """
    if not isinstance(text, str) or not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        # sensible fallback: half overlap
        overlap = max(1, chunk_size // 2)

    cleaned = _clean_text_for_chunking(text)

    # Prefer tokenization via tiktoken when available
    if _TIKTOKEN_AVAILABLE:
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(cleaned)
        chunks: List[str] = []
        i = 0
        total = len(tokens)
        while i < total:
            slice_tokens = tokens[i : i + chunk_size]
            chunk = enc.decode(slice_tokens).strip()
            if not chunk:
                i += max(1, chunk_size - overlap)
                continue

            # try not to cut mid-sentence if the chunk is sufficiently long
            chunk = _trim_to_sentence_boundary(chunk)

            chunks.append(chunk)
            i += chunk_size - overlap

        # final normalization: remove tiny chunks (noise)
        filtered = [c.strip() for c in chunks if len(c.strip()) > 20]
        return filtered

    # Fallback (no tiktoken): chunk by characters with approximate sizing
    approx_chars = int(chunk_size * 4)  # heuristic: ~4 characters per token
    step = approx_chars - int(overlap * 4)
    if step <= 0:
        step = max(1, approx_chars // 2)

    chunks: List[str] = []
    i = 0
    L = len(cleaned)
    while i < L:
        piece = cleaned[i : i + approx_chars].strip()
        if not piece:
            break
        piece = _trim_to_sentence_boundary(piece)
        chunks.append(piece)
        i += step

    return [c for c in chunks if len(c) > 20]
