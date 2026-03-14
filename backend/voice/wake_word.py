"""Wake-word detector using Picovoice Porcupine."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import AsyncIterable, Awaitable, Callable
import os

from dotenv import load_dotenv

load_dotenv()

FrameProcessor = Callable[[list[int]], int]
DetectionHandler = Callable[[], Awaitable[None]]


class WakeWordError(RuntimeError):
    """Raised for wake-word initialization or processing errors."""


@dataclass(frozen=True)
class WakeWordConfig:
    """Wake-word runtime configuration."""

    access_key: str
    keyword_paths: tuple[str, ...] | None = None
    keywords: tuple[str, ...] = ("hey flap",)
    sensitivities: tuple[float, ...] = (0.6,)


class WakeWordDetector:
    """Offline Porcupine wake-word detection wrapper."""

    def __init__(
        self,
        *,
        config: WakeWordConfig | None = None,
        frame_processor: FrameProcessor | None = None,
    ) -> None:
        access_key = os.getenv("PICOVOICE_ACCESS_KEY", "").strip()
        self.config = config or WakeWordConfig(access_key=access_key)
        self._frame_processor = frame_processor
        self._porcupine = None

    def _ensure_processor(self) -> FrameProcessor:
        if self._frame_processor is not None:
            return self._frame_processor
        if not self.config.access_key:
            raise WakeWordError("PICOVOICE_ACCESS_KEY is required for wake-word detection.")
        try:
            import pvporcupine
        except ImportError as error:
            raise WakeWordError("pvporcupine is required for wake-word detection.") from error

        if self._porcupine is None:
            kwargs = {
                "access_key": self.config.access_key,
                "sensitivities": list(self.config.sensitivities),
            }
            if self.config.keyword_paths:
                kwargs["keyword_paths"] = list(self.config.keyword_paths)
            else:
                kwargs["keywords"] = list(self.config.keywords)
            self._porcupine = pvporcupine.create(**kwargs)

        def process_frame(frame: list[int]) -> int:
            return int(self._porcupine.process(frame))

        return process_frame

    async def detect_from_frames(
        self,
        *,
        frame_stream: AsyncIterable[list[int]],
        on_detected: DetectionHandler,
    ) -> None:
        """Run detection loop over incoming PCM frames."""
        processor = self._ensure_processor()
        async for frame in frame_stream:
            result = processor(frame)
            if result >= 0:
                await on_detected()

    def process_frame(self, frame: list[int]) -> bool:
        """Process one frame and return True if wake-word detected."""
        processor = self._ensure_processor()
        return processor(frame) >= 0

    async def close(self) -> None:
        """Release Porcupine resources."""
        if self._porcupine is not None:
            await asyncio.to_thread(self._porcupine.delete)
            self._porcupine = None
