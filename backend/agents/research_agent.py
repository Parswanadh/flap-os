"""Research agent for web-style synthesis using routed LLM reasoning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.router.llm_router import LLMRouter


@dataclass(frozen=True)
class ResearchResult:
    """Structured research output."""

    summary: str
    model: str
    provider: str
    route_mode: str
    tokens: int


class ResearchAgent:
    """Generates concise research synthesis for a task."""

    def __init__(self, *, router: LLMRouter) -> None:
        self.router = router

    async def run(self, *, task: str, context: dict[str, Any] | None = None) -> ResearchResult:
        if not task.strip():
            raise ValueError("task must not be empty")
        context_blob = context or {}
        prompt = (
            "Research the request and produce a compact engineering summary.\n"
            f"Task: {task.strip()}\n"
            f"Context: {context_blob}\n"
            "Return factual bullets and practical next actions."
        )
        result = await self.router.generate(
            messages=[{"role": "user", "content": prompt}],
            requested_mode="reasoning",
            metadata={"agent": "research"},
        )
        return ResearchResult(
            summary=result.output_text,
            model=result.model,
            provider=result.provider,
            route_mode=result.route_mode,
            tokens=result.total_tokens,
        )
