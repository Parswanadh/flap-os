"""Browser agent using Playwright MCP-compatible HTTP bridge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
from dotenv import load_dotenv
import os

load_dotenv()


class BrowserAgentError(RuntimeError):
    """Raised when browser-automation requests fail."""


@dataclass(frozen=True)
class BrowserAgentResult:
    """Browser automation response."""

    success: bool
    details: str
    data: dict[str, Any]


class BrowserAgent:
    """Executes browser automation tasks through a Playwright MCP endpoint."""

    def __init__(self, *, mcp_url: str | None = None, timeout_s: float = 30.0) -> None:
        self.mcp_url = (mcp_url or os.getenv("PLAYWRIGHT_MCP_URL", "")).strip().rstrip("/")
        self.timeout_s = timeout_s

    async def run(
        self,
        *,
        task: str,
        url: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BrowserAgentResult:
        if not task.strip():
            raise ValueError("task must not be empty")
        if not self.mcp_url:
            raise BrowserAgentError("PLAYWRIGHT_MCP_URL is not configured.")

        payload = {
            "task": task.strip(),
            "url": url,
            "metadata": metadata or {},
        }
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            response = await client.post(f"{self.mcp_url}/run", json=payload)

        if response.status_code != 200:
            raise BrowserAgentError(
                f"Playwright MCP request failed ({response.status_code}): {response.text[:300]}"
            )
        data = response.json()
        success = bool(data.get("success", True))
        details = str(data.get("summary", data.get("message", "")))
        return BrowserAgentResult(success=success, details=details, data=data)
