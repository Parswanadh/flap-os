"""Agent S2 bridge for planning and executing GUI actions."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from backend.tools.computer_control import ComputerControl, CommandTimeoutError


class AgentS2BridgeError(RuntimeError):
    """Raised when Agent S2 planning or execution fails."""


@dataclass(frozen=True)
class GuiAction:
    """One GUI action emitted by Agent S2."""

    action: str
    x: int | None = None
    y: int | None = None
    text: str | None = None
    amount: int | None = None


PlannerRunner = Callable[[str, str], Awaitable[str]]


class AgentS2Bridge:
    """Coordinates screenshot->plan->execute workflow using Agent S2."""

    def __init__(
        self,
        *,
        computer_control: ComputerControl,
        planner_command: str = "agent-s2",
        planner_timeout_s: float = 45.0,
        planner_runner: PlannerRunner | None = None,
    ) -> None:
        self.computer_control = computer_control
        self.planner_command = planner_command
        self.planner_timeout_s = planner_timeout_s
        self._planner_runner = planner_runner

    async def _default_planner_runner(self, goal: str, screenshot_path: str) -> str:
        process = await asyncio.create_subprocess_exec(
            self.planner_command,
            "plan",
            "--goal",
            goal,
            "--screenshot",
            screenshot_path,
            "--format",
            "json",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=self.planner_timeout_s)
        except asyncio.TimeoutError as error:
            process.kill()
            await process.communicate()
            raise CommandTimeoutError(f"Agent S2 planner timed out after {self.planner_timeout_s:.1f}s") from error

        if process.returncode != 0:
            raise AgentS2BridgeError(
                f"Agent S2 planner failed with code {process.returncode}: {stderr.decode(errors='replace')[:500]}"
            )
        return stdout.decode(errors="replace")

    @staticmethod
    def _parse_actions(raw_output: str) -> list[GuiAction]:
        try:
            payload = json.loads(raw_output)
        except json.JSONDecodeError as error:
            raise AgentS2BridgeError(f"Agent S2 returned invalid JSON: {error}") from error

        if not isinstance(payload, dict):
            raise AgentS2BridgeError("Agent S2 response must be a JSON object")
        raw_actions = payload.get("actions", [])
        if not isinstance(raw_actions, list):
            raise AgentS2BridgeError("Agent S2 response field 'actions' must be a list")

        parsed: list[GuiAction] = []
        for item in raw_actions:
            if not isinstance(item, dict):
                raise AgentS2BridgeError("Each action entry must be an object")
            action = str(item.get("action", "")).strip().lower()
            if action not in {"click", "type", "scroll"}:
                raise AgentS2BridgeError(f"Unsupported Agent S2 action: {action}")
            parsed.append(
                GuiAction(
                    action=action,
                    x=int(item["x"]) if "x" in item and item["x"] is not None else None,
                    y=int(item["y"]) if "y" in item and item["y"] is not None else None,
                    text=str(item["text"]) if "text" in item and item["text"] is not None else None,
                    amount=int(item["amount"]) if "amount" in item and item["amount"] is not None else None,
                )
            )
        return parsed

    async def plan_actions(self, *, goal: str, screenshot_path: str | None = None) -> list[GuiAction]:
        """Generate GUI action plan from Agent S2."""
        if not goal.strip():
            raise ValueError("goal must not be empty")
        screenshot = screenshot_path or await self.computer_control.screenshot()
        runner = self._planner_runner or self._default_planner_runner
        raw_output = await runner(goal, screenshot)
        return self._parse_actions(raw_output)

    async def execute_actions(self, actions: list[GuiAction]) -> None:
        """Execute Agent S2-produced GUI actions through ComputerControl."""
        for action in actions:
            if action.action == "click":
                if action.x is None or action.y is None:
                    raise AgentS2BridgeError("Click action requires x and y")
                await self.computer_control.click(x=action.x, y=action.y)
            elif action.action == "type":
                if action.text is None:
                    raise AgentS2BridgeError("Type action requires text")
                await self.computer_control.type_text(action.text)
            elif action.action == "scroll":
                if action.amount is None:
                    raise AgentS2BridgeError("Scroll action requires amount")
                await self.computer_control.scroll(action.amount)

    async def solve(self, *, goal: str, screenshot_path: str | None = None) -> list[GuiAction]:
        """Plan and execute one Agent S2 action sequence."""
        actions = await self.plan_actions(goal=goal, screenshot_path=screenshot_path)
        await self.execute_actions(actions)
        return actions
