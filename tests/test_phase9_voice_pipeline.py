from __future__ import annotations

import asyncio

import pytest

from backend.voice.stt import STTError, STTService
from backend.voice.tts import TTSService
from backend.voice.wake_word import WakeWordConfig, WakeWordDetector


@pytest.mark.asyncio
async def test_stt_fallback_to_groq() -> None:
    async def fail_deepgram(audio: bytes, filename: str) -> str:
        raise STTError("deepgram down")

    async def ok_groq(audio: bytes, filename: str) -> str:
        return "fallback transcript"

    stt = STTService(deepgram_file_transcriber=fail_deepgram, groq_file_transcriber=ok_groq)
    text, provider = await stt.transcribe_file_with_fallback(b"audio-bytes", "a.wav")
    assert provider == "groq"
    assert text == "fallback transcript"


@pytest.mark.asyncio
async def test_tts_interrupt_stops_stream() -> None:
    async def runner(text: str):
        yield b"chunk-1"
        await asyncio.sleep(0.01)
        yield b"chunk-2"
        await asyncio.sleep(0.01)
        yield b"chunk-3"

    tts = TTSService(stream_runner=runner)
    chunks: list[bytes] = []

    async for chunk in tts.synthesize_stream("hello"):
        chunks.append(chunk)
        if chunk == b"chunk-1":
            tts.interrupt()

    assert chunks == [b"chunk-1"]


def test_wake_word_detector_process_frame() -> None:
    def fake_processor(frame: list[int]) -> int:
        return 0 if frame and frame[0] == 1 else -1

    detector = WakeWordDetector(
        config=WakeWordConfig(access_key="test-key", keywords=("hey flap",), sensitivities=(0.5,)),
        frame_processor=fake_processor,
    )
    assert detector.process_frame([0, 0, 0]) is False
    assert detector.process_frame([1, 0, 0]) is True


@pytest.mark.asyncio
async def test_wake_word_detect_from_frames() -> None:
    calls = {"count": 0}

    async def on_detected() -> None:
        calls["count"] += 1

    async def frame_stream():
        for frame in ([0, 1], [1, 2], [0, 0], [1, 3]):
            yield frame

    def fake_processor(frame: list[int]) -> int:
        return 0 if frame[0] == 1 else -1

    detector = WakeWordDetector(
        config=WakeWordConfig(access_key="test-key", keywords=("hey flap",), sensitivities=(0.5,)),
        frame_processor=fake_processor,
    )
    await detector.detect_from_frames(frame_stream=frame_stream(), on_detected=on_detected)
    assert calls["count"] == 2
