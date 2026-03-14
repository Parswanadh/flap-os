"""Persistent memory store using ChromaDB vectors + SQLite logs."""

from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import aiosqlite
from dotenv import load_dotenv
from litellm import aembedding
from litellm.exceptions import APIConnectionError, AuthenticationError, BadRequestError, RateLimitError

load_dotenv()

RETRYABLE_EMBED_ERRORS = (APIConnectionError, RateLimitError, asyncio.TimeoutError)
PROVIDER_ENV_KEYS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


class MemoryStoreError(RuntimeError):
    """Base class for memory-store failures."""


class MemoryEmbeddingError(MemoryStoreError):
    """Raised when no embedding model can produce vectors."""


class VectorCollection(Protocol):
    """Protocol for ChromaDB-like collection methods used by Mem0Store."""

    def upsert(
        self,
        *,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> Any:
        """Insert/update vector documents."""

    def query(
        self,
        *,
        query_embeddings: list[list[float]],
        n_results: int,
        include: list[str],
    ) -> dict[str, Any]:
        """Run a similarity query."""

    def delete(self, *, ids: list[str]) -> Any:
        """Delete vector documents by ID."""


@dataclass(frozen=True)
class MemoryHit:
    """Similarity search result."""

    memory_id: str
    text: str
    source: str
    distance: float | None
    metadata: dict[str, Any]
    created_at: str | None


class Mem0Store:
    """Local memory storage with fact extraction and semantic retrieval."""

    def __init__(
        self,
        *,
        sqlite_path: str | Path = "backend/data/flap_memory.db",
        chroma_path: str | Path = "backend/data/chroma",
        collection_name: str = "flap_memories",
        collection: VectorCollection | None = None,
        embedding_models: tuple[str, ...] | None = None,
    ) -> None:
        self.sqlite_path = Path(sqlite_path)
        self.chroma_path = Path(chroma_path)
        self.collection_name = collection_name
        self._collection = collection
        self._initialized = collection is not None
        self.embedding_models = embedding_models or (
            os.getenv("FLAP_EMBEDDING_MODEL", "ollama/nomic-embed-text"),
            "openai/text-embedding-3-small",
            "mistral/mistral-embed",
        )

    async def initialize(self) -> None:
        """Initialize sqlite schema and Chroma collection."""
        await self._initialize_sqlite()
        if self._initialized:
            return
        self._collection = await self._init_chroma_collection()
        self._initialized = True

    async def _initialize_sqlite(self) -> None:
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.sqlite_path.as_posix()) as connection:
            await connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    raw_text TEXT NOT NULL,
                    extracted_fact TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            await connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    details_json TEXT NOT NULL
                )
                """
            )
            await connection.commit()

    async def _init_chroma_collection(self) -> VectorCollection:
        try:
            import chromadb  # type: ignore
        except ImportError as error:
            raise MemoryStoreError(
                "chromadb is required for Mem0Store. Install it in the FLAP conda env first."
            ) from error

        self.chroma_path.mkdir(parents=True, exist_ok=True)
        client = await asyncio.to_thread(chromadb.PersistentClient, path=self.chroma_path.as_posix())
        return await asyncio.to_thread(
            client.get_or_create_collection,
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def _provider_from_model(model: str) -> str:
        if "/" not in model:
            return "unknown"
        return model.split("/", maxsplit=1)[0].strip().lower()

    @staticmethod
    def _provider_has_credentials(provider: str) -> bool:
        if provider == "ollama":
            return True
        key_name = PROVIDER_ENV_KEYS.get(provider)
        if key_name is None:
            return False
        return bool(os.getenv(key_name))

    async def _embed_text(self, text: str) -> list[float]:
        last_error: Exception | None = None
        for model in self.embedding_models:
            provider = self._provider_from_model(model)
            if not self._provider_has_credentials(provider):
                continue
            try:
                response = await aembedding(model=model, input=[text])
                data = response["data"][0]["embedding"]
                return [float(value) for value in data]
            except RETRYABLE_EMBED_ERRORS as error:
                last_error = error
                continue
            except (AuthenticationError, BadRequestError) as error:
                last_error = error
                continue

        raise MemoryEmbeddingError(
            f"Failed to embed text with configured models: {self.embedding_models}. "
            f"Last error: {type(last_error).__name__ if last_error else 'None'}"
        )

    @staticmethod
    def extract_facts(text: str) -> list[str]:
        """Extract compact memory facts from conversational text."""
        normalized = " ".join(text.strip().split())
        if not normalized:
            raise ValueError("text must not be empty")

        sentence_candidates = re.split(r"(?<=[.!?])\s+", normalized)
        fact_pattern = re.compile(
            r"\b(i am|i'm|my|i use|i work on|project|server|laptop|university|interest|goal)\b",
            re.IGNORECASE,
        )
        facts = [sentence.strip() for sentence in sentence_candidates if fact_pattern.search(sentence)]
        if facts:
            return facts[:6]
        return [normalized[:320]]

    async def _record_event(self, event_type: str, details: dict[str, Any]) -> None:
        created_at = datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(self.sqlite_path.as_posix()) as connection:
            await connection.execute(
                "INSERT INTO memory_events (created_at, event_type, details_json) VALUES (?, ?, ?)",
                (created_at, event_type, json.dumps(details, ensure_ascii=True)),
            )
            await connection.commit()

    async def add_memory(
        self,
        *,
        text: str,
        source: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[str]:
        """Extract facts from text and persist them into vector + sqlite stores."""
        await self.initialize()
        if self._collection is None:
            raise MemoryStoreError("Vector collection is not initialized.")

        facts = self.extract_facts(text)
        created_at = datetime.now(timezone.utc).isoformat()
        user_metadata = metadata or {}
        memory_ids: list[str] = []

        for fact in facts:
            memory_id = str(uuid.uuid4())
            embedding = await self._embed_text(fact)
            payload_metadata = {"source": source, "created_at": created_at, **user_metadata}
            await asyncio.to_thread(
                self._collection.upsert,
                ids=[memory_id],
                embeddings=[embedding],
                documents=[fact],
                metadatas=[payload_metadata],
            )
            async with aiosqlite.connect(self.sqlite_path.as_posix()) as connection:
                await connection.execute(
                    """
                    INSERT INTO memories (memory_id, source, raw_text, extracted_fact, metadata_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        memory_id,
                        source,
                        text,
                        fact,
                        json.dumps(payload_metadata, ensure_ascii=True),
                        created_at,
                    ),
                )
                await connection.commit()
            memory_ids.append(memory_id)

        await self._record_event(
            "memory_added",
            {"source": source, "count": len(memory_ids), "memory_ids": memory_ids},
        )
        return memory_ids

    async def search_memories(self, *, query: str, limit: int = 5) -> list[MemoryHit]:
        """Semantic search across stored memory facts."""
        await self.initialize()
        if self._collection is None:
            raise MemoryStoreError("Vector collection is not initialized.")
        if limit < 1:
            raise ValueError("limit must be >= 1")

        query_embedding = await self._embed_text(query.strip())
        raw = await asyncio.to_thread(
            self._collection.query,
            query_embeddings=[query_embedding],
            n_results=limit,
            include=["documents", "metadatas", "distances"],
        )
        ids = raw.get("ids", [[]])[0]
        docs = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        hits: list[MemoryHit] = []
        for memory_id, doc, metadata, distance in zip(ids, docs, metadatas, distances, strict=False):
            metadata_dict = metadata if isinstance(metadata, dict) else {}
            hits.append(
                MemoryHit(
                    memory_id=str(memory_id),
                    text=str(doc),
                    source=str(metadata_dict.get("source", "unknown")),
                    distance=float(distance) if distance is not None else None,
                    metadata=metadata_dict,
                    created_at=metadata_dict.get("created_at"),
                )
            )

        await self._record_event("memory_search", {"query": query, "limit": limit, "results": len(hits)})
        return hits

    async def recent_memories(self, limit: int = 5) -> list[MemoryHit]:
        """Return most recently added memory facts from sqlite."""
        await self.initialize()
        if limit < 1:
            raise ValueError("limit must be >= 1")

        async with aiosqlite.connect(self.sqlite_path.as_posix()) as connection:
            cursor = await connection.execute(
                """
                SELECT memory_id, extracted_fact, source, metadata_json, created_at
                FROM memories
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = await cursor.fetchall()
            await cursor.close()

        hits: list[MemoryHit] = []
        for memory_id, fact, source, metadata_json, created_at in rows:
            metadata = json.loads(metadata_json)
            hits.append(
                MemoryHit(
                    memory_id=memory_id,
                    text=fact,
                    source=source,
                    distance=None,
                    metadata=metadata,
                    created_at=created_at,
                )
            )
        return hits

    async def delete_memory(self, memory_id: str) -> None:
        """Delete memory from both vector and sqlite stores."""
        await self.initialize()
        if self._collection is None:
            raise MemoryStoreError("Vector collection is not initialized.")
        if not memory_id.strip():
            raise ValueError("memory_id must not be empty")

        await asyncio.to_thread(self._collection.delete, ids=[memory_id])
        async with aiosqlite.connect(self.sqlite_path.as_posix()) as connection:
            await connection.execute("DELETE FROM memories WHERE memory_id = ?", (memory_id,))
            await connection.commit()
        await self._record_event("memory_deleted", {"memory_id": memory_id})
