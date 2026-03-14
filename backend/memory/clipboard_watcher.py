"""Semantic clipboard watcher that pushes copied text into memory."""

from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime, timezone
from typing import Awaitable, Callable

from backend.memory.mem0_store import Mem0Store


class ClipboardWatcherError(RuntimeError):
    """Raised for clipboard watcher runtime failures."""


class SemanticClipboardWatcher:
    """Poll clipboard content and persist meaningful changes into memory."""

    def __init__(
        self,
        *,
        memory_store: Mem0Store,
        poll_interval_s: float = 1.0,
        max_chars: int = 4000,
        clipboard_reader: Callable[[], str] | None = None,
        on_store: Callable[[list[str], str], Awaitable[None]] | None = None,
    ) -> None:
        if poll_interval_s <= 0:
            raise ValueError("poll_interval_s must be > 0")
        if max_chars < 32:
            raise ValueError("max_chars must be >= 32")
        self.memory_store = memory_store
        self.poll_interval_s = poll_interval_s
        self.max_chars = max_chars
        self._clipboard_reader = clipboard_reader
        self._on_store = on_store
        self._last_hash: str | None = None
        self._stop_event = asyncio.Event()

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _is_storeworthy(text: str) -> bool:
        stripped = text.strip()
        if len(stripped) < 8:
            return False
        if stripped.isdigit():
            return False
        return True

    async def _read_clipboard(self) -> str:
        if self._clipboard_reader is not None:
            return self._clipboard_reader()
        try:
            import pyperclip
        except ImportError as error:
            raise ClipboardWatcherError(
                "pyperclip is required for clipboard monitoring. Install it in FLAP env."
            ) from error

        try:
            return await asyncio.to_thread(pyperclip.paste)
        except pyperclip.PyperclipException as error:
            raise ClipboardWatcherError(f"Clipboard read failed: {error}") from error

    async def run_once(self) -> bool:
        """Read clipboard once and store if it changed and looks meaningful."""
        content = (await self._read_clipboard()).strip()
        if not self._is_storeworthy(content):
            return False
        content = content[: self.max_chars]
        content_hash = self._hash_text(content)
        if self._last_hash == content_hash:
            return False

        self._last_hash = content_hash
        memory_ids = await self.memory_store.add_memory(
            text=content,
            source="clipboard",
            metadata={"captured_at": datetime.now(timezone.utc).isoformat()},
        )
        if self._on_store is not None:
            await self._on_store(memory_ids, content)
        return True

    async def start(self) -> None:
        """Run polling loop until stop() is requested."""
        self._stop_event.clear()
        while not self._stop_event.is_set():
            await self.run_once()
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self.poll_interval_s)
            except asyncio.TimeoutError:
                continue

    def stop(self) -> None:
        """Stop polling loop."""
        self._stop_event.set()
