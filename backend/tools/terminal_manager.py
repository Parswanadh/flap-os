"""Async client for FLAP terminal-server REST and websocket APIs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv
import os
import websockets

load_dotenv()

ERROR_PATTERN = re.compile(r"\b(error|exception|failed)\b", re.IGNORECASE)


class TerminalManagerError(RuntimeError):
    """Raised when terminal manager operations fail."""


@dataclass(frozen=True)
class TerminalSessionStatus:
    """Session status from terminal-server."""

    name: str
    pid: int
    alive: bool
    buffer_length: int
    started_at: str
    last_error_line: str | None


@dataclass(frozen=True)
class TerminalBuffer:
    """Buffered output snapshot for one session."""

    session: str
    buffer: str
    buffer_length: int


class TerminalManagerClient:
    """Client for 6 named PTY sessions managed by terminal-server."""

    def __init__(self, *, base_url: str | None = None, timeout_s: float = 20.0) -> None:
        self.base_url = (base_url or os.getenv("TERMINAL_SERVER_URL", "http://127.0.0.1:3001")).rstrip("/")
        self.timeout_s = timeout_s

    async def health(self) -> bool:
        """Check terminal-server liveness."""
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            response = await client.get(f"{self.base_url}/health")
        return response.status_code == 200

    async def list_sessions(self) -> list[TerminalSessionStatus]:
        """Return status for all managed PTY sessions."""
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            response = await client.get(f"{self.base_url}/sessions")
        if response.status_code != 200:
            raise TerminalManagerError(f"Failed to list sessions ({response.status_code}): {response.text[:300]}")

        payload = response.json()
        raw_sessions = payload.get("sessions", [])
        if not isinstance(raw_sessions, list):
            raise TerminalManagerError("Invalid terminal-server payload: sessions must be a list")

        result: list[TerminalSessionStatus] = []
        for item in raw_sessions:
            if not isinstance(item, dict):
                continue
            result.append(
                TerminalSessionStatus(
                    name=str(item.get("name", "")),
                    pid=int(item.get("pid", 0)),
                    alive=bool(item.get("alive", False)),
                    buffer_length=int(item.get("bufferLength", 0)),
                    started_at=str(item.get("startedAt", "")),
                    last_error_line=str(item["lastErrorLine"]) if item.get("lastErrorLine") else None,
                )
            )
        return result

    async def get_buffer(self, session_name: str) -> TerminalBuffer:
        """Fetch buffered stdout/stderr for a named session."""
        if not session_name.strip():
            raise ValueError("session_name must not be empty")
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            response = await client.get(f"{self.base_url}/sessions/{session_name}/buffer")
        if response.status_code != 200:
            raise TerminalManagerError(f"Failed to fetch buffer ({response.status_code}): {response.text[:300]}")
        payload = response.json()
        return TerminalBuffer(
            session=str(payload.get("session", session_name)),
            buffer=str(payload.get("buffer", "")),
            buffer_length=int(payload.get("bufferLength", 0)),
        )

    async def send_input(self, session_name: str, input_text: str) -> None:
        """Send terminal input to a named session."""
        if not session_name.strip():
            raise ValueError("session_name must not be empty")
        if not input_text:
            raise ValueError("input_text must not be empty")
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            response = await client.post(
                f"{self.base_url}/sessions/{session_name}/input",
                json={"input": input_text},
            )
        if response.status_code != 200:
            raise TerminalManagerError(f"Failed to send input ({response.status_code}): {response.text[:300]}")

    async def resize(self, session_name: str, *, cols: int, rows: int) -> None:
        """Resize PTY dimensions for a named session."""
        if cols < 20 or rows < 5:
            raise ValueError("cols must be >=20 and rows must be >=5")
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            response = await client.post(
                f"{self.base_url}/sessions/{session_name}/resize",
                json={"cols": cols, "rows": rows},
            )
        if response.status_code != 200:
            raise TerminalManagerError(f"Failed to resize session ({response.status_code}): {response.text[:300]}")

    @staticmethod
    def scan_errors(output_text: str) -> list[str]:
        """Extract suspicious lines containing Error/Exception/FAILED."""
        lines = [line.strip() for line in output_text.splitlines() if line.strip()]
        return [line for line in lines if ERROR_PATTERN.search(line)]

    def websocket_url(self) -> str:
        """Convert HTTP base URL to terminal websocket endpoint URL."""
        parsed = urlparse(self.base_url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        netloc = parsed.netloc
        return f"{scheme}://{netloc}/ws"

    async def stream_events(self, handler: Callable[[dict[str, Any]], Awaitable[None]]) -> None:
        """Subscribe to terminal websocket events and invoke async handler."""
        ws_url = self.websocket_url()
        async with websockets.connect(ws_url, open_timeout=self.timeout_s) as ws:
            async for raw in ws:
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError as error:
                    raise TerminalManagerError("Received invalid JSON event from terminal-server") from error
                if not isinstance(payload, dict):
                    raise TerminalManagerError("Received non-object websocket payload from terminal-server")
                await handler(payload)
