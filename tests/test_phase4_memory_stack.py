from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import httpx

from backend.memory.clipboard_watcher import SemanticClipboardWatcher
from backend.memory.mem0_store import Mem0Store
from backend.memory.screenpipe_client import ScreenpipeClient


class FakeCollection:
    def __init__(self) -> None:
        self.items: dict[str, dict[str, Any]] = {}

    def upsert(
        self,
        *,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        for memory_id, embedding, document, metadata in zip(ids, embeddings, documents, metadatas, strict=False):
            self.items[memory_id] = {
                "embedding": embedding,
                "document": document,
                "metadata": metadata,
            }

    def query(self, *, query_embeddings: list[list[float]], n_results: int, include: list[str]) -> dict[str, Any]:
        query = query_embeddings[0]
        scored: list[tuple[str, float]] = []
        for memory_id, payload in self.items.items():
            emb = payload["embedding"]
            distance = sum((a - b) ** 2 for a, b in zip(query, emb, strict=False))
            scored.append((memory_id, distance))
        scored.sort(key=lambda item: item[1])
        top = scored[:n_results]
        ids = [memory_id for memory_id, _ in top]
        docs = [self.items[memory_id]["document"] for memory_id in ids]
        metadatas = [self.items[memory_id]["metadata"] for memory_id in ids]
        distances = [distance for _, distance in top]
        return {"ids": [ids], "documents": [docs], "metadatas": [metadatas], "distances": [distances]}

    def delete(self, *, ids: list[str]) -> None:
        for memory_id in ids:
            self.items.pop(memory_id, None)


@pytest.mark.asyncio
async def test_mem0_store_add_search_recent_delete(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    collection = FakeCollection()
    store = Mem0Store(
        sqlite_path=tmp_path / "mem.db",
        chroma_path=tmp_path / "chroma",
        collection=collection,
    )

    async def fake_embed(text: str) -> list[float]:
        value = float(len(text) % 13)
        return [value, value / 10.0]

    monkeypatch.setattr(store, "_embed_text", fake_embed)

    memory_ids = await store.add_memory(
        text="I am building FLAP and my server is Dell 5490.",
        source="chat",
        metadata={"channel": "telegram"},
    )
    assert len(memory_ids) >= 1

    hits = await store.search_memories(query="FLAP server", limit=3)
    assert len(hits) >= 1
    assert hits[0].source == "chat"

    recent = await store.recent_memories(limit=5)
    assert len(recent) >= 1

    await store.delete_memory(memory_ids[0])
    recent_after_delete = await store.recent_memories(limit=5)
    assert all(hit.memory_id != memory_ids[0] for hit in recent_after_delete)


@pytest.mark.asyncio
async def test_clipboard_watcher_stores_unique_content(tmp_path: Path) -> None:
    class FakeMemoryStore:
        def __init__(self) -> None:
            self.saved: list[str] = []

        async def add_memory(self, *, text: str, source: str, metadata: dict[str, Any] | None = None):
            self.saved.append(text)
            return [f"id-{len(self.saved)}"]

    values = iter(["hello world from clipboard", "hello world from clipboard", "new clipboard text"])
    fake_store = FakeMemoryStore()
    watcher = SemanticClipboardWatcher(
        memory_store=fake_store,  # type: ignore[arg-type]
        clipboard_reader=lambda: next(values),
    )

    first = await watcher.run_once()
    second = await watcher.run_once()
    third = await watcher.run_once()

    assert first is True
    assert second is False
    assert third is True
    assert len(fake_store.saved) == 2


@pytest.mark.asyncio
async def test_screenpipe_client_query_normalizes_results(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeAsyncClient:
        def __init__(self, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url: str, params: dict[str, Any] | None = None):
            if url.endswith("/health"):
                return httpx.Response(200, json={"status": "ok"})
            payload = {"results": [{"id": "evt-1", "content": "editing code", "source": "screen", "timestamp": "2026-01-01T10:00:00Z"}]}
            return httpx.Response(200, json=payload)

    monkeypatch.setattr("backend.memory.screenpipe_client.httpx.AsyncClient", FakeAsyncClient)

    client = ScreenpipeClient(base_url="http://screenpipe.local")
    healthy = await client.health()
    assert healthy is True

    results = await client.query(query="editing", limit=5)
    assert len(results) == 1
    assert results[0].event_id == "evt-1"
    assert results[0].content == "editing code"
