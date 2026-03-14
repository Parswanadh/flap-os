"""FastAPI app entry point for FLAP."""

from fastapi import FastAPI

app = FastAPI(title="FLAP Backend", version="0.1.0")


@app.get("/health")
async def health() -> dict[str, str]:
    """Basic health endpoint used by orchestration checks."""
    return {"status": "ok"}
