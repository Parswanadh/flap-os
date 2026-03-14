"""FastAPI entrypoint for FLAP backend services."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from backend.agents.flap_core import build_chat_messages
from backend.router.llm_router import (
    BudgetLimitExceededError,
    LLMRouter,
    RoutingFailureError,
    build_router_from_env,
)


class ChatHistoryMessage(BaseModel):
    """Validated chat history message."""

    role: Literal["user", "assistant", "tool"] = Field(description="Message role in prior context.")
    content: str = Field(min_length=1, description="Message content.")


class ChatRequest(BaseModel):
    """Incoming chat payload for FLAP."""

    user_input: str = Field(min_length=1, description="Current user utterance.")
    mode: Literal["fast_chat", "code", "reasoning", "long_context", "offline"] | None = Field(
        default=None,
        description="Optional explicit router mode override.",
    )
    history: list[ChatHistoryMessage] = Field(default_factory=list, description="Prior turns.")
    estimated_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Optional caller-provided token estimate for routing.",
    )


class ChatResponse(BaseModel):
    """Outgoing chat response payload."""

    request_id: str
    mode: str
    model: str
    provider: str
    message: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    latency_ms: int
    created_at: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize shared app dependencies once."""
    app.state.router = build_router_from_env()
    yield


app = FastAPI(title="FLAP Backend", version="0.2.0", lifespan=lifespan)


def _get_router() -> LLMRouter:
    router = getattr(app.state, "router", None)
    if router is None:
        raise RuntimeError("Router has not been initialized.")
    if not isinstance(router, LLMRouter):
        raise RuntimeError("Router instance type is invalid.")
    return router


@app.get("/health")
async def health() -> dict[str, str]:
    """Health endpoint for docker, uptime checks, and supervisors."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest) -> ChatResponse:
    """Run FLAP personality + LLM routing for one chat request."""
    router = _get_router()
    try:
        history = [item.model_dump() for item in payload.history]
        messages = build_chat_messages(payload.user_input, history)
        result = await router.generate(
            messages=messages,
            requested_mode=payload.mode,
            estimated_tokens=payload.estimated_tokens,
            metadata={"surface": "fastapi", "endpoint": "/chat"},
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except BudgetLimitExceededError as error:
        raise HTTPException(status_code=429, detail=str(error)) from error
    except RoutingFailureError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    return ChatResponse(
        request_id=result.request_id,
        mode=result.route_mode,
        model=result.model,
        provider=result.provider,
        message=result.output_text,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        total_tokens=result.total_tokens,
        cost_usd=result.cost_usd,
        latency_ms=result.latency_ms,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
