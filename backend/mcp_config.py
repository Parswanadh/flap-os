"""MCP server configuration registry for FLAP backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MCPServerConfig:
    """Config entry for one MCP server process."""

    command: str
    args: tuple[str, ...]
    env: dict[str, str] | None = None


COMMUNITY_MCP_SERVERS: dict[str, MCPServerConfig] = {
    "github": MCPServerConfig(
        command="npx",
        args=("-y", "@modelcontextprotocol/server-github"),
        env={"GITHUB_TOKEN": "${GITHUB_TOKEN}"},
    ),
    "playwright": MCPServerConfig(
        command="npx",
        args=("-y", "@playwright/mcp@latest"),
    ),
    "brave-search": MCPServerConfig(
        command="npx",
        args=("-y", "@modelcontextprotocol/server-brave-search"),
        env={"BRAVE_API_KEY": "${BRAVE_API_KEY}"},
    ),
    "sqlite": MCPServerConfig(
        command="npx",
        args=("-y", "@modelcontextprotocol/server-sqlite", "--db-path", "/home/parshu/flap-os/backend/data/flap.db"),
    ),
    "flap-custom": MCPServerConfig(
        command="python",
        args=("mcp-servers/flap_mcp_server.py",),
    ),
}


def as_process_map() -> dict[str, dict[str, Any]]:
    """Convert typed configs to plain dictionaries for runtime launchers."""
    output: dict[str, dict[str, Any]] = {}
    for name, config in COMMUNITY_MCP_SERVERS.items():
        item: dict[str, Any] = {
            "command": config.command,
            "args": list(config.args),
        }
        if config.env:
            item["env"] = dict(config.env)
        output[name] = item
    return output
