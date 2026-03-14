import asyncio
from pathlib import Path
from types import SimpleNamespace

import aiosqlite
import pytest

from backend.router.llm_router import (
    BudgetLimitExceededError,
    BudgetTracker,
    LLMRouter,
    RouteMode,
)


def _response(
    *,
    text: str = "ok",
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text),
                finish_reason="stop",
            )
        ],
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    )


@pytest.mark.asyncio
async def test_route_selection_uses_code_signal(tmp_path: Path) -> None:
    tracker = BudgetTracker(db_path=tmp_path / "usage.db", daily_budget_usd=100.0, monthly_budget_usd=1000.0)
    router = LLMRouter(budget_tracker=tracker)

    mode = router.select_route_mode(
        messages=[{"role": "user", "content": "```python\nprint('hi')\n```"}],
        estimated_tokens=100,
    )
    assert mode == RouteMode.CODE


@pytest.mark.asyncio
async def test_generate_code_prefers_local_ollama(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []
    tracker = BudgetTracker(db_path=tmp_path / "usage.db", daily_budget_usd=100.0, monthly_budget_usd=1000.0)
    router = LLMRouter(budget_tracker=tracker)

    async def fake_acompletion(**kwargs):
        calls.append(kwargs["model"])
        return _response(text="code-answer")

    monkeypatch.setattr("backend.router.llm_router.acompletion", fake_acompletion)
    monkeypatch.setattr("backend.router.llm_router.completion_cost", lambda completion_response: 0.001)

    result = await router.generate(
        messages=[{"role": "user", "content": "Write a Python function to reverse a list."}],
        requested_mode=RouteMode.CODE,
    )

    assert calls[0] == "ollama/deepseek-coder-v2"
    assert result.model == "ollama/deepseek-coder-v2"
    assert result.route_mode == RouteMode.CODE
    assert result.output_text == "code-answer"


@pytest.mark.asyncio
async def test_generate_falls_back_after_timeout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []
    tracker = BudgetTracker(db_path=tmp_path / "usage.db", daily_budget_usd=100.0, monthly_budget_usd=1000.0)
    router = LLMRouter(budget_tracker=tracker, max_retries_per_model=0)

    monkeypatch.setenv("GROQ_API_KEY", "test-groq")
    monkeypatch.setenv("MISTRAL_API_KEY", "test-mistral")

    async def fake_acompletion(**kwargs):
        calls.append(kwargs["model"])
        if kwargs["model"] == "groq/llama-3.3-70b":
            raise asyncio.TimeoutError("upstream timeout")
        return _response(text="fallback-ok")

    monkeypatch.setattr("backend.router.llm_router.acompletion", fake_acompletion)
    monkeypatch.setattr("backend.router.llm_router.completion_cost", lambda completion_response: 0.0)

    result = await router.generate(
        messages=[{"role": "user", "content": "Summarize this in one paragraph."}],
        requested_mode=RouteMode.FAST_CHAT,
    )

    assert calls[0] == "groq/llama-3.3-70b"
    assert calls[1] == "mistral/mistral-large-latest"
    assert result.model == "mistral/mistral-large-latest"
    assert result.output_text == "fallback-ok"


@pytest.mark.asyncio
async def test_budget_limit_blocks_request(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    called = False
    tracker = BudgetTracker(db_path=tmp_path / "usage.db", daily_budget_usd=0.0, monthly_budget_usd=1000.0)
    router = LLMRouter(budget_tracker=tracker)

    async def fake_acompletion(**kwargs):
        nonlocal called
        called = True
        return _response()

    monkeypatch.setattr("backend.router.llm_router.acompletion", fake_acompletion)

    with pytest.raises(BudgetLimitExceededError):
        await router.generate(messages=[{"role": "user", "content": "hello"}])

    assert called is False


@pytest.mark.asyncio
async def test_router_records_usage_event(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    usage_db = tmp_path / "usage.db"
    tracker = BudgetTracker(db_path=usage_db, daily_budget_usd=100.0, monthly_budget_usd=1000.0)
    router = LLMRouter(budget_tracker=tracker)

    async def fake_acompletion(**kwargs):
        return _response(text="logged", prompt_tokens=12, completion_tokens=8)

    monkeypatch.setattr("backend.router.llm_router.acompletion", fake_acompletion)
    monkeypatch.setattr("backend.router.llm_router.completion_cost", lambda completion_response: 0.0025)

    result = await router.generate(messages=[{"role": "user", "content": "quick hi"}])
    assert result.total_tokens == 20

    async with aiosqlite.connect(usage_db.as_posix()) as connection:
        cursor = await connection.execute("SELECT COUNT(*), SUM(success) FROM llm_usage")
        row = await cursor.fetchone()
        await cursor.close()

    assert row is not None
    assert int(row[0]) >= 1
    assert int(row[1]) >= 1
