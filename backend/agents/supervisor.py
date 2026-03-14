"""LangGraph supervisor orchestrating FLAP sub-agents."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, is_dataclass
from typing import Any, Awaitable, Callable, TypedDict

from langgraph.graph import END, StateGraph

from backend.agents.browser_agent import BrowserAgent
from backend.agents.coder_agent import CoderAgent
from backend.agents.rag_agent import RagAgent
from backend.agents.reflection import ReflectionAttempt, run_with_reflection
from backend.agents.research_agent import ResearchAgent
from backend.agents.terminal_agent import TerminalAgent
from backend.memory.mem0_store import Mem0Store
from backend.router.llm_router import LLMRouter
from backend.tools.terminal_manager import TerminalManagerClient


class SupervisorState(TypedDict):
    """State shape for supervisor graph execution."""

    task: str
    selected_agents: list[str]
    agent_outputs: dict[str, Any]
    errors: list[str]
    final_response: str


NotifyCallback = Callable[[str], Awaitable[None]]


class FlapSupervisor:
    """Supervisor-agent pattern over research/code/browser/terminal/rag workers."""

    def __init__(
        self,
        *,
        router: LLMRouter,
        research_agent: ResearchAgent | None = None,
        coder_agent: CoderAgent | None = None,
        terminal_agent: TerminalAgent | None = None,
        browser_agent: BrowserAgent | None = None,
        rag_agent: RagAgent | None = None,
        notify_callback: NotifyCallback | None = None,
    ) -> None:
        self.router = router
        self.research_agent = research_agent or ResearchAgent(router=router)
        self.coder_agent = coder_agent or CoderAgent(router=router)
        self.terminal_agent = terminal_agent or TerminalAgent(manager=TerminalManagerClient())
        self.browser_agent = browser_agent or BrowserAgent()
        self.rag_agent = rag_agent or RagAgent(memory_store=Mem0Store())
        self.notify_callback = notify_callback
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(SupervisorState)
        graph.add_node("plan", self._plan_node)
        graph.add_node("execute", self._execute_node)
        graph.add_node("aggregate", self._aggregate_node)
        graph.set_entry_point("plan")
        graph.add_edge("plan", "execute")
        graph.add_edge("execute", "aggregate")
        graph.add_edge("aggregate", END)
        return graph.compile()

    @staticmethod
    def _select_agents(task: str) -> list[str]:
        lowered = task.lower()
        selected: list[str] = []
        if any(token in lowered for token in ("research", "search", "latest", "find", "investigate")):
            selected.append("research")
        if any(token in lowered for token in ("code", "bug", "fix", "implement", "refactor", "python", "node")):
            selected.append("coder")
        if any(token in lowered for token in ("terminal", "shell", "session", "stderr", "stdout", "log")):
            selected.append("terminal")
        if any(token in lowered for token in ("browser", "playwright", "website", "page", "click")):
            selected.append("browser")
        if any(token in lowered for token in ("memory", "remember", "history", "notes", "what was i doing")):
            selected.append("rag")
        if not selected:
            selected = ["research"]
        return selected

    async def _plan_node(self, state: SupervisorState) -> dict[str, Any]:
        task = state["task"]
        return {"selected_agents": self._select_agents(task)}

    async def _on_reflection_retry(self, record: ReflectionAttempt) -> None:
        if self.notify_callback is None:
            return
        await self.notify_callback(
            f"Agent retry {record.attempt}: {record.error_type} - {record.error_message}"
        )

    async def _on_reflection_escalate(self, agent_name: str, attempts: list[ReflectionAttempt]) -> None:
        if self.notify_callback is None:
            return
        last = attempts[-1]
        await self.notify_callback(
            f"Agent escalation: {agent_name} failed after {len(attempts)} retries "
            f"({last.error_type}: {last.error_message})"
        )

    async def _invoke_agent(self, agent_name: str, task: str) -> dict[str, Any]:
        if agent_name == "research":
            result = await self.research_agent.run(task=task)
        elif agent_name == "coder":
            result = await self.coder_agent.run(task=task)
        elif agent_name == "terminal":
            result = await self.terminal_agent.run(task=task)
        elif agent_name == "browser":
            result = await self.browser_agent.run(task=task)
        elif agent_name == "rag":
            result = await self.rag_agent.run(query=task)
        else:
            raise ValueError(f"Unsupported agent: {agent_name}")

        if is_dataclass(result):
            return asdict(result)
        if isinstance(result, dict):
            return result
        return {"result": result}

    async def _execute_node(self, state: SupervisorState) -> dict[str, Any]:
        task = state["task"]
        selected_agents = state.get("selected_agents", [])
        outputs: dict[str, Any] = {}
        errors: list[str] = []

        async def run_agent(agent_name: str) -> tuple[str, dict[str, Any] | None, str | None]:
            async def operation() -> dict[str, Any]:
                return await self._invoke_agent(agent_name, task)

            try:
                result = await run_with_reflection(
                    agent_name=agent_name,
                    operation=operation,
                    max_retries=3,
                    on_retry=self._on_reflection_retry,
                    on_escalate=self._on_reflection_escalate,
                )
                return (agent_name, result, None)
            except Exception as error:  # noqa: BLE001 - surfaced in state errors
                return (agent_name, None, f"{agent_name}: {type(error).__name__}: {error}")

        results = await asyncio.gather(*(run_agent(name) for name in selected_agents))
        for agent_name, payload, error in results:
            if payload is not None:
                outputs[agent_name] = payload
            if error is not None:
                errors.append(error)
        return {"agent_outputs": outputs, "errors": errors}

    async def _aggregate_node(self, state: SupervisorState) -> dict[str, Any]:
        outputs = state.get("agent_outputs", {})
        errors = state.get("errors", [])
        task = state["task"]
        prompt = (
            "Aggregate sub-agent outputs into one concise operator response.\n"
            f"Task: {task}\n"
            f"Agent outputs: {outputs}\n"
            f"Errors: {errors}\n"
            "Return a compact answer with clear next action."
        )
        result = await self.router.generate(
            messages=[{"role": "user", "content": prompt}],
            requested_mode="reasoning",
            metadata={"agent": "supervisor_aggregate"},
        )
        return {"final_response": result.output_text}

    async def run(self, *, task: str) -> SupervisorState:
        """Run full supervisor graph for a task."""
        if not task.strip():
            raise ValueError("task must not be empty")
        initial_state: SupervisorState = {
            "task": task.strip(),
            "selected_agents": [],
            "agent_outputs": {},
            "errors": [],
            "final_response": "",
        }
        result = await self.graph.ainvoke(initial_state)
        return result
