# src/main.py
"""
Entrypoint for the FastAPI app.
Shows original import traceback to help debugging.
"""

try:
    # package-relative (works for 
    # 1`uvicorn src.main:app`)
    from .query_server import app  # type: ignore
except Exception as e:
    import traceback, sys
    print("FAILED importing .query_server — printing full traceback (original error):", file=sys.stderr)
    traceback.print_exc()
    # re-raise so uvicorn also sees the error
    raise

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
