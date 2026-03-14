from __future__ import annotations

import sys
from pathlib import Path

import pytest

from backend.tools.agent_s2_bridge import AgentS2Bridge, AgentS2BridgeError, GuiAction
from backend.tools.computer_control import CommandTimeoutError, ComputerControl


@pytest.mark.asyncio
async def test_execute_shell_success(tmp_path: Path) -> None:
    control = ComputerControl(default_cwd=tmp_path.as_posix())
    result = await control.execute_shell(f'"{sys.executable}" -c "print(123)"', timeout_s=10.0)
    assert result.exit_code == 0
    assert "123" in result.stdout


@pytest.mark.asyncio
async def test_execute_shell_timeout(tmp_path: Path) -> None:
    control = ComputerControl(default_cwd=tmp_path.as_posix())
    with pytest.raises(CommandTimeoutError):
        await control.execute_shell(f'"{sys.executable}" -c "import time; time.sleep(2)"', timeout_s=0.2)


@pytest.mark.asyncio
async def test_open_app_uses_xdg_open_for_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    control = ComputerControl(default_cwd=tmp_path.as_posix())
    seen: dict[str, str] = {}

    async def fake_execute_shell(command: str, **kwargs):
        seen["command"] = command
        class _Result:
            exit_code = 0
            stdout = ""
            stderr = ""
        return _Result()

    monkeypatch.setattr(control, "execute_shell", fake_execute_shell)
    await control.open_app("/home/parshu/notes.txt")
    assert seen["command"].startswith("xdg-open ")


@pytest.mark.asyncio
async def test_agent_s2_bridge_plan_and_execute() -> None:
    calls: list[tuple[str, object]] = []

    class FakeControl:
        async def screenshot(self, path: str | None = None) -> str:
            return "/tmp/fake.png"

        async def click(self, *, x: int, y: int, **kwargs) -> None:
            calls.append(("click", (x, y)))

        async def type_text(self, text: str, **kwargs) -> None:
            calls.append(("type", text))

        async def scroll(self, amount: int) -> None:
            calls.append(("scroll", amount))

    async def fake_runner(goal: str, screenshot_path: str) -> str:
        return """
        {
          "actions": [
            {"action": "click", "x": 120, "y": 240},
            {"action": "type", "text": "hello"},
            {"action": "scroll", "amount": -500}
          ]
        }
        """

    bridge = AgentS2Bridge(computer_control=FakeControl(), planner_runner=fake_runner)
    actions = await bridge.solve(goal="open terminal and type hello")
    assert len(actions) == 3
    assert calls == [("click", (120, 240)), ("type", "hello"), ("scroll", -500)]


@pytest.mark.asyncio
async def test_agent_s2_bridge_rejects_invalid_actions() -> None:
    class FakeControl:
        async def screenshot(self, path: str | None = None) -> str:
            return "/tmp/fake.png"

        async def click(self, *, x: int, y: int, **kwargs) -> None:
            return None

        async def type_text(self, text: str, **kwargs) -> None:
            return None

        async def scroll(self, amount: int) -> None:
            return None

    async def fake_runner(goal: str, screenshot_path: str) -> str:
        return '{"actions":[{"action":"drag","x":1,"y":2}]}'

    bridge = AgentS2Bridge(computer_control=FakeControl(), planner_runner=fake_runner)
    with pytest.raises(AgentS2BridgeError):
        await bridge.plan_actions(goal="invalid")


def test_gui_action_dataclass() -> None:
    action = GuiAction(action="click", x=1, y=2)
    assert action.action == "click"
    assert action.x == 1
