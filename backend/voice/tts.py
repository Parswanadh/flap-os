"""Text-to-speech services (Deepgram Aura-2 streaming + interrupt)."""

from __future__ import annotations

import asyncio
import json
import os
from typing import AsyncIterator, Awaitable, Callable

from dotenv import load_dotenv
import websockets

load_dotenv()

StreamRunner = Callable[[str], AsyncIterator[bytes]]


class TTSError(RuntimeError):
    """Raised for text-to-speech failures."""


class TTSService:
    """Deepgram Aura-2 streaming TTS with interruption support."""

    def __init__(
        self,
        *,
        deepgram_api_key: str | None = None,
        voice_model: str = "aura-2-thalia-en",
        sample_rate: int = 24_000,
        stream_runner: StreamRunner | None = None,
    ) -> None:
        self.deepgram_api_key = (deepgram_api_key or os.getenv("DEEPGRAM_API_KEY", "")).strip()
        self.voice_model = voice_model
        self.sample_rate = sample_rate
        self._interrupt_event = asyncio.Event()
        self._stream_runner = stream_runner

    def interrupt(self) -> None:
        """Interrupt currently playing synthesis stream."""
        self._interrupt_event.set()

    def clear_interrupt(self) -> None:
        """Clear previous interrupt state."""
        self._interrupt_event.clear()

    async def _default_stream_runner(self, text: str) -> AsyncIterator[bytes]:
        if not self.deepgram_api_key:
            raise TTSError("DEEPGRAM_API_KEY is missing.")
        ws_url = (
            "wss://api.deepgram.com/v1/speak"
            f"?model={self.voice_model}&encoding=linear16&sample_rate={self.sample_rate}"
        )
        headers = {"Authorization": f"Token {self.deepgram_api_key}"}
        async with websockets.connect(ws_url, additional_headers=headers, max_size=2_000_000) as ws:
            await ws.send(json.dumps({"type": "Speak", "text": text}))
            await ws.send(json.dumps({"type": "Flush"}))

            while True:
                if self._interrupt_event.is_set():
                    await ws.send(json.dumps({"type": "Close"}))
                    break
                message = await ws.recv()
                if isinstance(message, bytes):
                    yield message
                    continue
                payload = json.loads(message)
                if payload.get("type") in {"Flushed", "Closed"}:
                    break

    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        """Yield audio chunks for provided text until completion or interruption."""
        if not text.strip():
            raise ValueError("text must not be empty")
        self.clear_interrupt()
        runner = self._stream_runner or self._default_stream_runner
        async for chunk in runner(text.strip()):
            if self._interrupt_event.is_set():
                break
            yield chunk

    async def synthesize_full(self, text: str) -> bytes:
        """Collect stream chunks and return full audio bytes."""
        output = bytearray()
        async for chunk in self.synthesize_stream(text):
            output.extend(chunk)
        return bytes(output)
