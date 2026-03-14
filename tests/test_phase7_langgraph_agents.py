from __future__ import annotations

from dataclasses import dataclass

import pytest

from backend.agents.reflection import ReflectionError, run_with_reflection
from backend.agents.supervisor import FlapSupervisor


@pytest.mark.asyncio
async def test_reflection_retries_then_succeeds() -> None:
    calls = {"count": 0}

    async def operation():
        calls["count"] += 1
        if calls["count"] < 3:
            raise RuntimeError("temporary")
        return "ok"

    result = await run_with_reflection(agent_name="test", operation=operation, max_retries=3)
    assert result == "ok"
    assert calls["count"] == 3


@pytest.mark.asyncio
async def test_reflection_raises_after_exhaustion() -> None:
    async def operation():
        raise ValueError("always fail")

    with pytest.raises(ReflectionError):
        await run_with_reflection(agent_name="test", operation=operation, max_retries=2)


@pytest.mark.asyncio
async def test_supervisor_runs_selected_agents() -> None:
    class FakeRouter:
        async def generate(self, **kwargs):
            class _Result:
                output_text = "aggregated summary"
            return _Result()

    @dataclass
    class FakeResearch:
        async def run(self, *, task: str, context=None):
            return {"summary": f"research:{task}"}

    @dataclass
    class FakeCoder:
        async def run(self, *, task: str, code_context=None, metadata=None):
            return {"response": f"code:{task}"}

    @dataclass
    class FakeTerminal:
        async def run(self, *, task: str, session=None, input_text=None):
            return {"summary": "terminal-ok", "sessions": [], "alerts": []}

    @dataclass
    class FakeBrowser:
        async def run(self, *, task: str, url=None, metadata=None):
            return {"success": True, "details": "browser-ok", "data": {}}

    @dataclass
    class FakeRag:
        async def run(self, *, query: str, limit: int = 5):
            return {"query": query, "hits": [], "summary": "rag-ok"}

    supervisor = FlapSupervisor(
        router=FakeRouter(),  # type: ignore[arg-type]
        research_agent=FakeResearch(),  # type: ignore[arg-type]
        coder_agent=FakeCoder(),  # type: ignore[arg-type]
        terminal_agent=FakeTerminal(),  # type: ignore[arg-type]
        browser_agent=FakeBrowser(),  # type: ignore[arg-type]
        rag_agent=FakeRag(),  # type: ignore[arg-type]
    )

    result = await supervisor.run(task="Research and code a fix for this terminal error")
    assert "research" in result["agent_outputs"]
    assert "coder" in result["agent_outputs"]
    assert "terminal" in result["agent_outputs"]
    assert result["final_response"] == "aggregated summary"
