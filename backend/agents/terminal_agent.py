"""Terminal agent for PTY session monitoring and command dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.tools.terminal_manager import TerminalManagerClient


@dataclass(frozen=True)
class TerminalAgentResult:
    """Terminal agent output payload."""

    summary: str
    sessions: list[dict[str, Any]]
    alerts: list[str]


class TerminalAgent:
    """Inspects terminal sessions and optionally dispatches command input."""

    def __init__(self, *, manager: TerminalManagerClient) -> None:
        self.manager = manager

    async def run(
        self,
        *,
        task: str,
        session: str | None = None,
        input_text: str | None = None,
    ) -> TerminalAgentResult:
        if not task.strip():
            raise ValueError("task must not be empty")
        sessions = await self.manager.list_sessions()

        if session is not None and input_text is not None:
            await self.manager.send_input(session, input_text)

        session_payload = [
            {
                "name": item.name,
                "pid": item.pid,
                "alive": item.alive,
                "buffer_length": item.buffer_length,
                "last_error_line": item.last_error_line,
            }
            for item in sessions
        ]
        alerts = [item.last_error_line for item in sessions if item.last_error_line]
        summary = (
            f"Terminal task processed. Active sessions: {sum(1 for item in sessions if item.alive)}/{len(sessions)}."
        )
        return TerminalAgentResult(summary=summary, sessions=session_payload, alerts=alerts)
