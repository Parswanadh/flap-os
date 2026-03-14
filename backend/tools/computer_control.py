"""Async computer-control tools for shell, app launch, and GUI actions."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import shlex
from time import perf_counter
from typing import Literal


class ComputerControlError(RuntimeError):
    """Base class for computer-control failures."""


class CommandTimeoutError(ComputerControlError):
    """Raised when shell commands exceed timeout."""


class CommandExecutionError(ComputerControlError):
    """Raised when shell command execution fails with check=True."""


@dataclass(frozen=True)
class ShellExecutionResult:
    """Structured shell execution output."""

    command: str
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int


class ComputerControl:
    """Provides async shell/app/gui control primitives."""

    def __init__(self, *, default_cwd: str = "/home/parshu") -> None:
        self.default_cwd = default_cwd

    async def execute_shell(
        self,
        command: str,
        *,
        timeout_s: float = 60.0,
        cwd: str | None = None,
        check: bool = False,
    ) -> ShellExecutionResult:
        """Run a shell command asynchronously with timeout and output capture."""
        if not command.strip():
            raise ValueError("command must not be empty")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be > 0")

        start = perf_counter()
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=cwd or self.default_cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(process.communicate(), timeout=timeout_s)
        except asyncio.TimeoutError as error:
            process.kill()
            stdout_bytes, stderr_bytes = await process.communicate()
            duration_ms = int((perf_counter() - start) * 1000)
            raise CommandTimeoutError(
                f"Command timed out after {timeout_s:.2f}s: {command}\n"
                f"stdout={stdout_bytes.decode(errors='replace')[:500]}\n"
                f"stderr={stderr_bytes.decode(errors='replace')[:500]}"
            ) from error

        result = ShellExecutionResult(
            command=command,
            exit_code=process.returncode,
            stdout=stdout_bytes.decode(errors="replace"),
            stderr=stderr_bytes.decode(errors="replace"),
            duration_ms=int((perf_counter() - start) * 1000),
        )
        if check and result.exit_code != 0:
            raise CommandExecutionError(
                f"Command failed with exit code {result.exit_code}: {command}\n"
                f"stderr={result.stderr[:500]}"
            )
        return result

    async def open_app(self, app_or_path: str) -> ShellExecutionResult:
        """Open application/path on Linux using xdg-open or nohup launch."""
        target = app_or_path.strip()
        if not target:
            raise ValueError("app_or_path must not be empty")

        if "/" in target or target.startswith(".") or target.endswith((".desktop", ".url", ".html", ".txt")):
            command = f"xdg-open {shlex.quote(target)}"
            return await self.execute_shell(command, timeout_s=20.0, check=False)

        # App name launch through shell lookup.
        command = f"nohup {shlex.quote(target)} >/tmp/flap-open-app.log 2>&1 &"
        return await self.execute_shell(command, timeout_s=20.0, check=False)

    async def list_windows(self) -> ShellExecutionResult:
        """List active windows via wmctrl."""
        return await self.execute_shell("wmctrl -lx", timeout_s=15.0, check=False)

    @staticmethod
    def _load_pyautogui():
        try:
            import pyautogui  # type: ignore
        except ImportError as error:
            raise ComputerControlError(
                "pyautogui is required for GUI automation. Install it in the FLAP environment."
            ) from error
        return pyautogui

    async def click(
        self,
        *,
        x: int,
        y: int,
        button: Literal["left", "middle", "right"] = "left",
        clicks: int = 1,
    ) -> None:
        """Move cursor and click at coordinates."""
        if clicks < 1:
            raise ValueError("clicks must be >= 1")
        pyautogui = self._load_pyautogui()
        await asyncio.to_thread(pyautogui.click, x=x, y=y, button=button, clicks=clicks)

    async def type_text(self, text: str, *, interval_s: float = 0.01) -> None:
        """Type text into the currently focused input field."""
        if not text:
            raise ValueError("text must not be empty")
        if interval_s < 0:
            raise ValueError("interval_s must be >= 0")
        pyautogui = self._load_pyautogui()
        await asyncio.to_thread(pyautogui.write, text, interval=interval_s)

    async def scroll(self, amount: int) -> None:
        """Scroll up/down by the specified amount."""
        if amount == 0:
            return
        pyautogui = self._load_pyautogui()
        await asyncio.to_thread(pyautogui.scroll, amount)

    async def screenshot(self, path: str | Path | None = None) -> str:
        """Capture screenshot and return saved absolute path."""
        pyautogui = self._load_pyautogui()
        output_path = Path(path) if path else Path(f"/tmp/flap-screenshot-{datetime.now(timezone.utc).timestamp():.0f}.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(pyautogui.screenshot, output_path.as_posix())
        return output_path.as_posix()
