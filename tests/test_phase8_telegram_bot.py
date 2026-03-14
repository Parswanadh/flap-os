from __future__ import annotations

import pytest
import httpx

from backend.telegram.bot import (
    is_destructive_command,
    parse_run_arguments,
    telegram_notify,
    transcribe_groq_voice,
)


def test_is_destructive_command() -> None:
    assert is_destructive_command("rm -rf /tmp/test") is True
    assert is_destructive_command("echo hello") is False


def test_parse_run_arguments() -> None:
    assert parse_run_arguments("") == (False, "")
    assert parse_run_arguments("ls -la") == (False, "ls -la")
    assert parse_run_arguments("--force rm -rf /tmp/a") == (True, "rm -rf /tmp/a")


@pytest.mark.asyncio
async def test_telegram_notify_returns_false_without_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    sent = await telegram_notify("hello")
    assert sent is False


@pytest.mark.asyncio
async def test_transcribe_groq_voice(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeAsyncClient:
        def __init__(self, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url: str, **kwargs):
            return httpx.Response(200, json={"text": "transcribed speech"})

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr("backend.telegram.bot.httpx.AsyncClient", FakeAsyncClient)
    text = await transcribe_groq_voice(audio_bytes=b"abc", filename="voice.ogg")
    assert text == "transcribed speech"
