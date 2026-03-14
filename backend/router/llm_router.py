"""LiteLLM routing and budget tracking for FLAP."""

from __future__ import annotations

import asyncio
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import aiosqlite
from dotenv import load_dotenv
from litellm import acompletion, completion_cost
from litellm.exceptions import (
    APIConnectionError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
)

load_dotenv()

RETRYABLE_ERRORS = (
    asyncio.TimeoutError,
    APIConnectionError,
    ServiceUnavailableError,
    InternalServerError,
    RateLimitError,
)

PROVIDER_ENV_KEYS: dict[str, str] = {
    "groq": "GROQ_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "openai": "OPENAI_API_KEY",
}


class RouterError(RuntimeError):
    """Base class for llm router failures."""


class BudgetLimitExceededError(RouterError):
    """Raised when budget thresholds are exceeded."""


class RoutingFailureError(RouterError):
    """Raised when all routing candidates fail."""


class RouteMode(str):
    """Supported route modes."""

    FAST_CHAT = "fast_chat"
    CODE = "code"
    REASONING = "reasoning"
    LONG_CONTEXT = "long_context"
    OFFLINE = "offline"


@dataclass(frozen=True)
class RouteConfig:
    """Configuration for each route mode."""

    timeout_s: float
    default_max_tokens: int
    models: tuple[str, ...]


ROUTE_CONFIGS: dict[str, RouteConfig] = {
    RouteMode.FAST_CHAT: RouteConfig(
        timeout_s=12.0,
        default_max_tokens=600,
        models=(
            "groq/llama-3.3-70b",
            "mistral/mistral-large-latest",
            "openrouter/anthropic/claude-sonnet-4-5",
            "openai/gpt-4o-mini",
            "ollama/qwen2.5:3b",
        ),
    ),
    RouteMode.CODE: RouteConfig(
        timeout_s=30.0,
        default_max_tokens=1800,
        models=(
            "ollama/deepseek-coder-v2",
            "groq/llama-3.3-70b",
            "mistral/mistral-large-latest",
            "openrouter/anthropic/claude-sonnet-4-5",
            "openai/gpt-4o-mini",
            "ollama/qwen2.5:3b",
        ),
    ),
    RouteMode.REASONING: RouteConfig(
        timeout_s=50.0,
        default_max_tokens=2800,
        models=(
            "openrouter/anthropic/claude-sonnet-4-5",
            "mistral/mistral-large-latest",
            "openai/gpt-4.1",
            "groq/llama-3.3-70b",
            "ollama/qwen2.5:3b",
        ),
    ),
    RouteMode.LONG_CONTEXT: RouteConfig(
        timeout_s=60.0,
        default_max_tokens=3200,
        models=(
            "mistral/mistral-large-latest",
            "openrouter/anthropic/claude-sonnet-4-5",
            "openai/gpt-4.1",
            "groq/llama-3.3-70b",
            "ollama/qwen2.5:3b",
        ),
    ),
    RouteMode.OFFLINE: RouteConfig(
        timeout_s=40.0,
        default_max_tokens=1200,
        models=("ollama/qwen2.5:3b",),
    ),
}


@dataclass(frozen=True)
class AttemptLog:
    """Per-attempt execution details."""

    model: str
    provider: str
    attempt: int
    status: str
    latency_ms: int = 0
    error_type: str | None = None
    error_message: str | None = None
    note: str | None = None


@dataclass(frozen=True)
class RouterResult:
    """Response shape returned by the router."""

    request_id: str
    route_mode: str
    model: str
    provider: str
    output_text: str
    finish_reason: str | None
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    latency_ms: int
    attempts: tuple[AttemptLog, ...]


class BudgetTracker:
    """Tracks and enforces daily/monthly LLM budget usage."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        daily_budget_usd: float | None = None,
        monthly_budget_usd: float | None = None,
    ) -> None:
        default_path = Path(os.getenv("FLAP_USAGE_DB_PATH", "backend/data/flap_usage.db"))
        self.db_path = Path(db_path) if db_path else default_path
        self.daily_budget_usd = (
            daily_budget_usd if daily_budget_usd is not None else float(os.getenv("FLAP_DAILY_BUDGET_USD", "3.0"))
        )
        self.monthly_budget_usd = (
            monthly_budget_usd
            if monthly_budget_usd is not None
            else float(os.getenv("FLAP_MONTHLY_BUDGET_USD", "50.0"))
        )
        self._initialized = False

    async def initialize(self) -> None:
        """Create sqlite schema if needed."""
        if self._initialized:
            return
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.db_path.as_posix()) as connection:
            await connection.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    route_mode TEXT NOT NULL,
                    model TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    prompt_tokens INTEGER NOT NULL,
                    completion_tokens INTEGER NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    cost_usd REAL NOT NULL,
                    latency_ms INTEGER NOT NULL,
                    success INTEGER NOT NULL,
                    error_type TEXT,
                    error_message TEXT
                )
                """
            )
            await connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_llm_usage_created_at ON llm_usage(created_at)"
            )
            await connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_llm_usage_route_mode ON llm_usage(route_mode)"
            )
            await connection.commit()
        self._initialized = True

    async def _sum_spend_between(self, start_iso: str, end_iso: str) -> float:
        await self.initialize()
        async with aiosqlite.connect(self.db_path.as_posix()) as connection:
            cursor = await connection.execute(
                """
                SELECT COALESCE(SUM(cost_usd), 0.0)
                FROM llm_usage
                WHERE success = 1 AND created_at >= ? AND created_at < ?
                """,
                (start_iso, end_iso),
            )
            row = await cursor.fetchone()
            await cursor.close()
        return float(row[0]) if row else 0.0

    async def current_daily_spend(self) -> float:
        """Return total successful cost since UTC day start."""
        now = datetime.now(timezone.utc)
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return await self._sum_spend_between(day_start.isoformat(), now.isoformat())

    async def current_monthly_spend(self) -> float:
        """Return total successful cost since UTC month start."""
        now = datetime.now(timezone.utc)
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        return await self._sum_spend_between(month_start.isoformat(), now.isoformat())

    async def assert_within_limits(self) -> None:
        """Raise when configured daily or monthly budgets are already exhausted."""
        daily_spend = await self.current_daily_spend()
        if daily_spend >= self.daily_budget_usd:
            raise BudgetLimitExceededError(
                f"Daily budget exceeded: ${daily_spend:.4f} / ${self.daily_budget_usd:.4f}"
            )

        monthly_spend = await self.current_monthly_spend()
        if monthly_spend >= self.monthly_budget_usd:
            raise BudgetLimitExceededError(
                f"Monthly budget exceeded: ${monthly_spend:.4f} / ${self.monthly_budget_usd:.4f}"
            )

    async def record_event(
        self,
        *,
        request_id: str,
        route_mode: str,
        model: str,
        provider: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        cost_usd: float,
        latency_ms: int,
        success: bool,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> None:
        """Persist one router attempt event."""
        await self.initialize()
        created_at = datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(self.db_path.as_posix()) as connection:
            await connection.execute(
                """
                INSERT INTO llm_usage (
                    created_at, request_id, route_mode, model, provider,
                    prompt_tokens, completion_tokens, total_tokens, cost_usd,
                    latency_ms, success, error_type, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at,
                    request_id,
                    route_mode,
                    model,
                    provider,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                    cost_usd,
                    latency_ms,
                    1 if success else 0,
                    error_type,
                    error_message,
                ),
            )
            await connection.commit()


class LLMRouter:
    """Route requests across providers with budget guards and fallback."""

    def __init__(
        self,
        *,
        budget_tracker: BudgetTracker | None = None,
        max_retries_per_model: int = 1,
        default_temperature: float = 0.2,
    ) -> None:
        self.budget_tracker = budget_tracker or BudgetTracker()
        self.max_retries_per_model = max_retries_per_model
        self.default_temperature = default_temperature

    @staticmethod
    def estimate_tokens(messages: list[dict[str, str]]) -> int:
        """Rough token estimate used for route selection."""
        total_chars = 0
        for message in messages:
            total_chars += len(message.get("content", ""))
        # Approximate English token count with a safety multiplier.
        return max(1, total_chars // 4)

    @staticmethod
    def _provider_for_model(model: str) -> str:
        if "/" not in model:
            return "unknown"
        return model.split("/", maxsplit=1)[0].strip().lower()

    @staticmethod
    def _provider_has_credentials(provider: str) -> bool:
        if provider == "ollama":
            return True
        env_key = PROVIDER_ENV_KEYS.get(provider)
        if env_key is None:
            return False
        return bool(os.getenv(env_key))

    @staticmethod
    def _looks_like_code(messages: list[dict[str, str]]) -> bool:
        code_pattern = re.compile(
            r"(```|^\s*def\s+|^\s*class\s+|^\s*import\s+|npm\s+install|pip\s+install|docker\s+compose)",
            re.IGNORECASE | re.MULTILINE,
        )
        return any(code_pattern.search(message.get("content", "")) for message in messages)

    def select_route_mode(
        self,
        *,
        messages: list[dict[str, str]],
        requested_mode: str | None = None,
        estimated_tokens: int | None = None,
    ) -> str:
        """Pick a route mode from explicit request or heuristics."""
        if os.getenv("FLAP_FORCE_OFFLINE", "0") == "1":
            return RouteMode.OFFLINE

        if requested_mode:
            if requested_mode not in ROUTE_CONFIGS:
                raise ValueError(f"Unsupported route mode: {requested_mode}")
            return requested_mode

        token_count = estimated_tokens if estimated_tokens is not None else self.estimate_tokens(messages)
        if self._looks_like_code(messages):
            return RouteMode.CODE
        if token_count < 500:
            return RouteMode.FAST_CHAT
        if token_count >= 6000:
            return RouteMode.LONG_CONTEXT
        return RouteMode.REASONING

    def _build_candidates(self, route_mode: str) -> tuple[list[str], list[AttemptLog]]:
        config = ROUTE_CONFIGS[route_mode]
        candidates: list[str] = []
        attempts: list[AttemptLog] = []
        for model in config.models:
            provider = self._provider_for_model(model)
            if self._provider_has_credentials(provider):
                candidates.append(model)
                continue
            attempts.append(
                AttemptLog(
                    model=model,
                    provider=provider,
                    attempt=0,
                    status="skipped",
                    note=f"Missing credentials: {PROVIDER_ENV_KEYS.get(provider, 'unknown provider')}",
                )
            )

        if not candidates and route_mode != RouteMode.OFFLINE:
            candidates.append(ROUTE_CONFIGS[RouteMode.OFFLINE].models[0])

        if not candidates:
            raise RoutingFailureError(f"No available models for route mode '{route_mode}'.")

        return candidates, attempts

    @staticmethod
    def _extract_text_and_reason(response: Any) -> tuple[str, str | None]:
        choices = getattr(response, "choices", None)
        if not choices:
            return "", None

        choice = choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason is None and isinstance(choice, dict):
            finish_reason = choice.get("finish_reason")

        message = getattr(choice, "message", None)
        if message is None and isinstance(choice, dict):
            message = choice.get("message")

        content = getattr(message, "content", None) if message is not None else None
        if isinstance(message, dict):
            content = message.get("content")

        if isinstance(content, list):
            chunks: list[str] = []
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    chunks.append(str(block["text"]))
                else:
                    chunks.append(str(block))
            return "".join(chunks).strip(), finish_reason

        if content is None:
            return "", finish_reason
        return str(content).strip(), finish_reason

    @staticmethod
    def _extract_usage(response: Any) -> tuple[int, int, int]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return (0, 0, 0)

        if isinstance(usage, dict):
            prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
            completion_tokens = int(usage.get("completion_tokens", 0) or 0)
            total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens) or 0)
            return (prompt_tokens, completion_tokens, total_tokens)

        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0)
        return (prompt_tokens, completion_tokens, total_tokens)

    @staticmethod
    def _safe_completion_cost(response: Any) -> float:
        try:
            return float(completion_cost(completion_response=response) or 0.0)
        except (ValueError, TypeError, KeyError):
            return 0.0

    async def generate(
        self,
        *,
        messages: list[dict[str, str]],
        requested_mode: str | None = None,
        estimated_tokens: int | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RouterResult:
        """Route and execute one completion request with provider fallback."""
        if not messages:
            raise ValueError("messages must not be empty")

        await self.budget_tracker.assert_within_limits()
        route_mode = self.select_route_mode(
            messages=messages,
            requested_mode=requested_mode,
            estimated_tokens=estimated_tokens,
        )
        route_config = ROUTE_CONFIGS[route_mode]
        candidates, attempts = self._build_candidates(route_mode)
        request_id = str(uuid.uuid4())
        overall_start = perf_counter()

        for model in candidates:
            provider = self._provider_for_model(model)
            for attempt_index in range(self.max_retries_per_model + 1):
                call_start = perf_counter()
                try:
                    response = await acompletion(
                        model=model,
                        messages=messages,
                        timeout=route_config.timeout_s,
                        temperature=temperature if temperature is not None else self.default_temperature,
                        max_tokens=max_tokens if max_tokens is not None else route_config.default_max_tokens,
                        metadata={
                            "request_id": request_id,
                            "route_mode": route_mode,
                            **(metadata or {}),
                        },
                    )
                    prompt_tokens, completion_tokens, total_tokens = self._extract_usage(response)
                    output_text, finish_reason = self._extract_text_and_reason(response)
                    latency_ms = int((perf_counter() - call_start) * 1000)
                    total_latency_ms = int((perf_counter() - overall_start) * 1000)
                    cost = self._safe_completion_cost(response)

                    await self.budget_tracker.record_event(
                        request_id=request_id,
                        route_mode=route_mode,
                        model=model,
                        provider=provider,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        cost_usd=cost,
                        latency_ms=latency_ms,
                        success=True,
                    )
                    attempts.append(
                        AttemptLog(
                            model=model,
                            provider=provider,
                            attempt=attempt_index,
                            status="success",
                            latency_ms=latency_ms,
                        )
                    )
                    return RouterResult(
                        request_id=request_id,
                        route_mode=route_mode,
                        model=model,
                        provider=provider,
                        output_text=output_text,
                        finish_reason=finish_reason,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        cost_usd=cost,
                        latency_ms=total_latency_ms,
                        attempts=tuple(attempts),
                    )
                except RETRYABLE_ERRORS as error:
                    latency_ms = int((perf_counter() - call_start) * 1000)
                    attempts.append(
                        AttemptLog(
                            model=model,
                            provider=provider,
                            attempt=attempt_index,
                            status="failed",
                            latency_ms=latency_ms,
                            error_type=type(error).__name__,
                            error_message=str(error),
                        )
                    )
                    await self.budget_tracker.record_event(
                        request_id=request_id,
                        route_mode=route_mode,
                        model=model,
                        provider=provider,
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                        cost_usd=0.0,
                        latency_ms=latency_ms,
                        success=False,
                        error_type=type(error).__name__,
                        error_message=str(error),
                    )
                    if attempt_index >= self.max_retries_per_model:
                        break
                except (AuthenticationError, BadRequestError, ValueError) as error:
                    latency_ms = int((perf_counter() - call_start) * 1000)
                    attempts.append(
                        AttemptLog(
                            model=model,
                            provider=provider,
                            attempt=attempt_index,
                            status="failed",
                            latency_ms=latency_ms,
                            error_type=type(error).__name__,
                            error_message=str(error),
                        )
                    )
                    await self.budget_tracker.record_event(
                        request_id=request_id,
                        route_mode=route_mode,
                        model=model,
                        provider=provider,
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                        cost_usd=0.0,
                        latency_ms=latency_ms,
                        success=False,
                        error_type=type(error).__name__,
                        error_message=str(error),
                    )
                    break

        raise RoutingFailureError(
            f"All model candidates failed for route '{route_mode}'. "
            f"Attempts={len(attempts)} request_id={request_id}"
        )


def build_router_from_env() -> LLMRouter:
    """Factory for application wiring."""
    db_path = os.getenv("FLAP_USAGE_DB_PATH")
    daily = float(os.getenv("FLAP_DAILY_BUDGET_USD", "3.0"))
    monthly = float(os.getenv("FLAP_MONTHLY_BUDGET_USD", "50.0"))
    tracker = BudgetTracker(db_path=db_path, daily_budget_usd=daily, monthly_budget_usd=monthly)
    return LLMRouter(budget_tracker=tracker)


__all__ = [
    "AttemptLog",
    "BudgetLimitExceededError",
    "BudgetTracker",
    "LLMRouter",
    "RouteMode",
    "RouterError",
    "RouterResult",
    "RoutingFailureError",
    "build_router_from_env",
]
