"""Async client for querying Screenpipe memory history."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from dotenv import load_dotenv
import os

load_dotenv()


class ScreenpipeClientError(RuntimeError):
    """Raised when Screenpipe query operations fail."""


@dataclass(frozen=True)
class ScreenpipeQueryResult:
    """Normalized Screenpipe event payload."""

    event_id: str
    content: str
    source: str
    timestamp: str
    raw: dict[str, Any]


class ScreenpipeClient:
    """HTTP client for Screenpipe's searchable timeline data."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        query_path: str = "/search",
        timeout_s: float = 20.0,
    ) -> None:
        self.base_url = (base_url or os.getenv("SCREENPIPE_BASE_URL", "http://127.0.0.1:3030")).rstrip("/")
        self.query_path = query_path if query_path.startswith("/") else f"/{query_path}"
        self.timeout_s = timeout_s

    async def health(self) -> bool:
        """Check whether Screenpipe is reachable."""
        url = f"{self.base_url}/health"
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            response = await client.get(url)
        return response.status_code == 200

    async def query(
        self,
        *,
        query: str,
        limit: int = 20,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[ScreenpipeQueryResult]:
        """Search Screenpipe indexed screen/audio history."""
        if not query.strip():
            raise ValueError("query must not be empty")
        if limit < 1:
            raise ValueError("limit must be >= 1")

        params: dict[str, str | int] = {"q": query.strip(), "limit": limit}
        if start_time is not None:
            params["start_time"] = start_time.astimezone(timezone.utc).isoformat()
        if end_time is not None:
            params["end_time"] = end_time.astimezone(timezone.utc).isoformat()

        url = f"{self.base_url}{self.query_path}"
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            response = await client.get(url, params=params)

        if response.status_code != 200:
            raise ScreenpipeClientError(
                f"Screenpipe query failed ({response.status_code}): {response.text[:300]}"
            )

        data = response.json()
        raw_items: list[dict[str, Any]]
        if isinstance(data, list):
            raw_items = [item for item in data if isinstance(item, dict)]
        elif isinstance(data, dict):
            candidate = data.get("results", [])
            if not isinstance(candidate, list):
                raise ScreenpipeClientError("Unexpected Screenpipe response format: missing list results")
            raw_items = [item for item in candidate if isinstance(item, dict)]
        else:
            raise ScreenpipeClientError("Unexpected Screenpipe response format")

        normalized: list[ScreenpipeQueryResult] = []
        for item in raw_items:
            event_id = str(item.get("id", item.get("event_id", ""))).strip() or "unknown"
            content = str(item.get("content", item.get("text", ""))).strip()
            source = str(item.get("source", item.get("type", "unknown"))).strip()
            timestamp = str(item.get("timestamp", item.get("created_at", ""))).strip()
            normalized.append(
                ScreenpipeQueryResult(
                    event_id=event_id,
                    content=content,
                    source=source,
                    timestamp=timestamp,
                    raw=item,
                )
            )
        return normalized

    async def what_was_i_doing(self, when: datetime, window_minutes: int = 30, limit: int = 10) -> list[ScreenpipeQueryResult]:
        """Retrieve timeline events around a specific timestamp."""
        if window_minutes < 1:
            raise ValueError("window_minutes must be >= 1")
        start = when - timedelta(minutes=window_minutes)
        end = when + timedelta(minutes=window_minutes)
        # Broad query keeps this endpoint provider-agnostic.
        return await self.query(query="activity", limit=limit, start_time=start, end_time=end)
