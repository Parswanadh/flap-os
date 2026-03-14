from __future__ import annotations

from typing import Any

import pytest
import httpx

from backend.tools.terminal_manager import TerminalBuffer, TerminalManagerClient


class FakeAsyncClient:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, url: str, **kwargs):
        if url.endswith("/health"):
            return httpx.Response(200, json={"status": "ok"})
        if url.endswith("/sessions"):
            payload = {
                "sessions": [
                    {
                        "name": "bash",
                        "pid": 1234,
                        "alive": True,
                        "bufferLength": 120,
                        "startedAt": "2026-01-01T00:00:00Z",
                        "lastErrorLine": None,
                    }
                ]
            }
            return httpx.Response(200, json=payload)
        if "/buffer" in url:
            return httpx.Response(200, json={"session": "bash", "buffer": "hello", "bufferLength": 5})
        return httpx.Response(404, json={"error": "not found"})

    async def post(self, url: str, **kwargs):
        return httpx.Response(200, json={"ok": True})


@pytest.mark.asyncio
async def test_terminal_manager_rest_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("backend.tools.terminal_manager.httpx.AsyncClient", FakeAsyncClient)
    client = TerminalManagerClient(base_url="http://localhost:3001")

    healthy = await client.health()
    assert healthy is True

    sessions = await client.list_sessions()
    assert len(sessions) == 1
    assert sessions[0].name == "bash"
    assert sessions[0].alive is True

    buffer = await client.get_buffer("bash")
    assert isinstance(buffer, TerminalBuffer)
    assert buffer.buffer == "hello"

    await client.send_input("bash", "ls\n")
    await client.resize("bash", cols=120, rows=40)


def test_terminal_manager_error_scan_and_ws_url() -> None:
    lines = TerminalManagerClient.scan_errors("ok line\nException happened\nbuild FAILED\nall good")
    assert lines == ["Exception happened", "build FAILED"]

    client = TerminalManagerClient(base_url="https://term.local:8443")
    assert client.websocket_url() == "wss://term.local:8443/ws"
