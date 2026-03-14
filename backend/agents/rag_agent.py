"""RAG agent over FLAP memory + project context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.memory.mem0_store import Mem0Store


@dataclass(frozen=True)
class RagResult:
    """RAG retrieval output."""

    query: str
    hits: list[dict[str, Any]]
    summary: str


class RagAgent:
    """Retrieves semantically relevant context from Mem0Store."""

    def __init__(self, *, memory_store: Mem0Store) -> None:
        self.memory_store = memory_store

    async def run(self, *, query: str, limit: int = 5) -> RagResult:
        if not query.strip():
            raise ValueError("query must not be empty")
        if limit < 1:
            raise ValueError("limit must be >= 1")

        hits = await self.memory_store.search_memories(query=query, limit=limit)
        payload = [
            {
                "memory_id": hit.memory_id,
                "text": hit.text,
                "source": hit.source,
                "distance": hit.distance,
                "created_at": hit.created_at,
            }
            for hit in hits
        ]
        summary = (
            "No matching memories found."
            if not payload
            else f"Found {len(payload)} memory hits. Top source: {payload[0]['source']}."
        )
        return RagResult(query=query, hits=payload, summary=summary)
