from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient

from backend.agents.flap_core import FLAP_SYSTEM_PROMPT, build_chat_messages
from backend.main import app
from backend.router.llm_router import BudgetLimitExceededError


def test_flap_system_prompt_is_embedded() -> None:
    assert FLAP_SYSTEM_PROMPT.startswith(
        "You are FLAP — Parshu's personal AI that runs 24/7 on his private Ubuntu server."
    )
    assert "You have 6 live terminal sessions: bash, claude, gemini, copilot, ollama, docker" in FLAP_SYSTEM_PROMPT
    assert "Plans: numbered list → \"Should I start?\"" in FLAP_SYSTEM_PROMPT


def test_build_chat_messages_order() -> None:
    messages = build_chat_messages(
        "Plan next step",
        history=[
            {"role": "user", "content": "Earlier prompt"},
            {"role": "assistant", "content": "Earlier response"},
        ],
    )
    assert messages[0]["role"] == "system"
    assert messages[1]["content"] == "Earlier prompt"
    assert messages[2]["content"] == "Earlier response"
    assert messages[3]["role"] == "user"
    assert messages[3]["content"] == "Plan next step"


@pytest.mark.asyncio
async def test_chat_endpoint_success(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeRouter:
        async def generate(self, **kwargs):
            return SimpleNamespace(
                request_id="req-123",
                route_mode="fast_chat",
                model="groq/llama-3.3-70b",
                provider="groq",
                output_text="done",
                prompt_tokens=12,
                completion_tokens=18,
                total_tokens=30,
                cost_usd=0.0021,
                latency_ms=45,
            )

    monkeypatch.setattr("backend.main._get_router", lambda: FakeRouter())
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/chat",
            json={"user_input": "hello", "history": []},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["request_id"] == "req-123"
    assert payload["model"] == "groq/llama-3.3-70b"
    assert payload["message"] == "done"


@pytest.mark.asyncio
async def test_chat_endpoint_budget_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeRouter:
        async def generate(self, **kwargs):
            raise BudgetLimitExceededError("Daily budget exceeded")

    monkeypatch.setattr("backend.main._get_router", lambda: FakeRouter())
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/chat",
            json={"user_input": "hello", "history": []},
        )

    assert response.status_code == 429
    assert "Daily budget exceeded" in response.text
