"""Coder agent for implementation/review/debug tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.router.llm_router import LLMRouter


@dataclass(frozen=True)
class CoderResult:
    """Structured code-agent response."""

    response: str
    model: str
    provider: str
    route_mode: str
    tokens: int


class CoderAgent:
    """Routes software tasks to code-focused model profiles."""

    def __init__(self, *, router: LLMRouter) -> None:
        self.router = router

    async def run(self, *, task: str, code_context: str | None = None, metadata: dict[str, Any] | None = None) -> CoderResult:
        if not task.strip():
            raise ValueError("task must not be empty")
        prompt = (
            "You are the coding sub-agent.\n"
            f"Task: {task.strip()}\n"
            f"Code context:\n{code_context or '(none)'}\n"
            "Return precise implementation guidance or code edits."
        )
        result = await self.router.generate(
            messages=[{"role": "user", "content": prompt}],
            requested_mode="code",
            metadata={"agent": "coder", **(metadata or {})},
        )
        return CoderResult(
            response=result.output_text,
            model=result.model,
            provider=result.provider,
            route_mode=result.route_mode,
            tokens=result.total_tokens,
        )
