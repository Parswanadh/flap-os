from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

from backend.mcp_config import as_process_map


def _load_flap_mcp_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "mcp-servers" / "flap_mcp_server.py"
    spec = importlib.util.spec_from_file_location("flap_mcp_server_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load flap_mcp_server module spec.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_flap_mcp_filesystem_tools(tmp_path: Path) -> None:
    module = _load_flap_mcp_module()
    target = tmp_path / "sample.txt"

    write_result = await module.filesystem_write(target.as_posix(), "hello")
    assert write_result["bytes_written"] == 5

    read_result = await module.filesystem_read(target.as_posix())
    assert read_result["content"] == "hello"

    search_result = await module.filesystem_search(tmp_path.as_posix(), "*.txt")
    assert target.as_posix() in search_result["matches"]


@pytest.mark.asyncio
async def test_flap_mcp_shell_execute(tmp_path: Path) -> None:
    module = _load_flap_mcp_module()
    command = f'"{sys.executable}" -c "print(456)"'
    result = await module.shell_execute(command, cwd=tmp_path.as_posix())
    assert result["exit_code"] == 0
    assert "456" in result["stdout"]


def test_mcp_config_contains_required_servers() -> None:
    config = as_process_map()
    assert "github" in config
    assert "playwright" in config
    assert "brave-search" in config
    assert "sqlite" in config
    assert "flap-custom" in config
