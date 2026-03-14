"""Self-healing retry wrapper for FLAP agent execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable


class ReflectionError(RuntimeError):
    """Raised when all reflection retries are exhausted."""


@dataclass(frozen=True)
class ReflectionAttempt:
    """Metadata for each retry attempt."""

    attempt: int
    error_type: str
    error_message: str


RetryCallback = Callable[[ReflectionAttempt], Awaitable[None]]
EscalationCallback = Callable[[str, list[ReflectionAttempt]], Awaitable[None]]


async def run_with_reflection(
    *,
    agent_name: str,
    operation: Callable[[], Awaitable[Any]],
    max_retries: int = 3,
    on_retry: RetryCallback | None = None,
    on_escalate: EscalationCallback | None = None,
) -> Any:
    """Run operation with retry and contextual escalation."""
    if max_retries < 1:
        raise ValueError("max_retries must be >= 1")

    attempts: list[ReflectionAttempt] = []
    for attempt in range(1, max_retries + 1):
        try:
            return await operation()
        except Exception as error:  # noqa: BLE001 - reflection intentionally wraps arbitrary agent failures
            record = ReflectionAttempt(
                attempt=attempt,
                error_type=type(error).__name__,
                error_message=str(error),
            )
            attempts.append(record)
            if attempt < max_retries and on_retry is not None:
                await on_retry(record)
                continue
            if attempt < max_retries:
                continue
            if on_escalate is not None:
                await on_escalate(agent_name, attempts)
            raise ReflectionError(
                f"{agent_name} failed after {max_retries} attempts. "
                f"Last error: {record.error_type}: {record.error_message}"
            ) from error
