"""Speech-to-text services (Deepgram Nova-3 + Groq Whisper fallback)."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, AsyncIterable, AsyncIterator, Awaitable, Callable

import httpx
from dotenv import load_dotenv
import websockets

load_dotenv()

Transcriber = Callable[[bytes, str], Awaitable[str]]


class STTError(RuntimeError):
    """Raised for STT failures."""


class STTService:
    """Speech transcription service with provider fallback logic."""

    def __init__(
        self,
        *,
        deepgram_api_key: str | None = None,
        groq_api_key: str | None = None,
        deepgram_file_transcriber: Transcriber | None = None,
        groq_file_transcriber: Transcriber | None = None,
    ) -> None:
        self.deepgram_api_key = (deepgram_api_key or os.getenv("DEEPGRAM_API_KEY", "")).strip()
        self.groq_api_key = (groq_api_key or os.getenv("GROQ_API_KEY", "")).strip()
        self._deepgram_file_transcriber = deepgram_file_transcriber
        self._groq_file_transcriber = groq_file_transcriber

    async def transcribe_file_deepgram(self, audio_bytes: bytes, filename: str = "audio.wav") -> str:
        """Transcribe full audio file bytes via Deepgram REST."""
        if self._deepgram_file_transcriber is not None:
            return await self._deepgram_file_transcriber(audio_bytes, filename)
        if not self.deepgram_api_key:
            raise STTError("DEEPGRAM_API_KEY is missing.")

        headers = {
            "Authorization": f"Token {self.deepgram_api_key}",
            "Content-Type": "audio/wav",
        }
        url = "https://api.deepgram.com/v1/listen?model=nova-3&smart_format=true"
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(url, headers=headers, content=audio_bytes)
        if response.status_code != 200:
            raise STTError(f"Deepgram transcription failed ({response.status_code}): {response.text[:300]}")
        payload = response.json()
        text = (
            payload.get("results", {})
            .get("channels", [{}])[0]
            .get("alternatives", [{}])[0]
            .get("transcript", "")
        )
        text = str(text).strip()
        if not text:
            raise STTError("Deepgram returned empty transcript.")
        return text

    async def transcribe_file_groq(self, audio_bytes: bytes, filename: str = "audio.wav") -> str:
        """Transcribe audio file bytes via Groq Whisper fallback."""
        if self._groq_file_transcriber is not None:
            return await self._groq_file_transcriber(audio_bytes, filename)
        if not self.groq_api_key:
            raise STTError("GROQ_API_KEY is missing.")
        headers = {"Authorization": f"Bearer {self.groq_api_key}"}
        files = {"file": (filename, audio_bytes, "audio/wav")}
        data = {"model": "whisper-large-v3-turbo"}
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers=headers,
                data=data,
                files=files,
            )
        if response.status_code != 200:
            raise STTError(f"Groq transcription failed ({response.status_code}): {response.text[:300]}")
        payload = response.json()
        text = str(payload.get("text", "")).strip()
        if not text:
            raise STTError("Groq returned empty transcript.")
        return text

    async def transcribe_file_with_fallback(self, audio_bytes: bytes, filename: str = "audio.wav") -> tuple[str, str]:
        """Try Deepgram first, then Groq fallback."""
        try:
            text = await self.transcribe_file_deepgram(audio_bytes, filename)
            return text, "deepgram"
        except STTError:
            text = await self.transcribe_file_groq(audio_bytes, filename)
            return text, "groq"

    async def stream_transcribe_deepgram(
        self,
        audio_stream: AsyncIterable[bytes],
        *,
        sample_rate: int = 16_000,
        encoding: str = "linear16",
    ) -> AsyncIterator[str]:
        """Transcribe an audio stream using Deepgram websocket streaming."""
        if not self.deepgram_api_key:
            raise STTError("DEEPGRAM_API_KEY is missing.")

        ws_url = (
            "wss://api.deepgram.com/v1/listen"
            f"?model=nova-3&encoding={encoding}&sample_rate={sample_rate}&interim_results=true"
        )
        headers = {"Authorization": f"Token {self.deepgram_api_key}"}

        async with websockets.connect(ws_url, additional_headers=headers, max_size=2_000_000) as ws:
            send_done = asyncio.Event()

            async def send_audio() -> None:
                async for chunk in audio_stream:
                    await ws.send(chunk)
                await ws.send(json.dumps({"type": "CloseStream"}))
                send_done.set()

            sender_task = asyncio.create_task(send_audio())
            try:
                while True:
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    except asyncio.TimeoutError:
                        if send_done.is_set():
                            break
                        continue
                    if isinstance(message, bytes):
                        continue
                    payload = json.loads(message)
                    transcript = (
                        payload.get("channel", {})
                        .get("alternatives", [{}])[0]
                        .get("transcript", "")
                    )
                    transcript = str(transcript).strip()
                    if transcript:
                        yield transcript
                    if payload.get("is_final") and send_done.is_set():
                        break
            finally:
                await sender_task
