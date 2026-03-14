"""Custom MCP server for FLAP local tooling."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from backend.memory.mem0_store import Mem0Store
from backend.memory.screenpipe_client import ScreenpipeClient
from backend.tools.computer_control import ComputerControl

load_dotenv()

mcp = FastMCP("flap-mcp")


def _safe_path(path: str) -> Path:
    candidate = Path(path).expanduser().resolve()
    return candidate


@mcp.tool()
async def filesystem_read(path: str) -> dict[str, Any]:
    """Read UTF-8 text file content from disk."""
    target = _safe_path(path)
    if not target.exists():
        raise FileNotFoundError(f"Path does not exist: {target}")
    if target.is_dir():
        raise IsADirectoryError(f"Path is a directory: {target}")
    return {"path": target.as_posix(), "content": target.read_text(encoding="utf-8")}


@mcp.tool()
async def filesystem_write(path: str, content: str) -> dict[str, Any]:
    """Write UTF-8 text content to disk."""
    target = _safe_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return {"path": target.as_posix(), "bytes_written": len(content.encode("utf-8"))}


@mcp.tool()
async def filesystem_search(root: str, pattern: str, limit: int = 50) -> dict[str, Any]:
    """Search file names under root with glob pattern."""
    if limit < 1:
        raise ValueError("limit must be >= 1")
    root_path = _safe_path(root)
    if not root_path.exists() or not root_path.is_dir():
        raise NotADirectoryError(f"Invalid search root: {root_path}")
    matches = [path.as_posix() for path in root_path.rglob(pattern)][:limit]
    return {"root": root_path.as_posix(), "pattern": pattern, "matches": matches}


@mcp.tool()
async def shell_execute(command: str, timeout_s: float = 60.0, cwd: str | None = None) -> dict[str, Any]:
    """Execute shell command asynchronously and return captured output."""
    control = ComputerControl(default_cwd=cwd or os.getcwd())
    result = await control.execute_shell(command, timeout_s=timeout_s, check=False)
    return {
        "command": result.command,
        "exit_code": result.exit_code,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "duration_ms": result.duration_ms,
    }


@mcp.tool()
async def comfyui_generate(prompt: str, workflow: dict[str, Any] | None = None) -> dict[str, Any]:
    """Trigger ComfyUI prompt generation endpoint."""
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    base_url = os.getenv("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")
    request_payload = {
        "prompt": workflow or {
            "1": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt}},
        }
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(f"{base_url}/prompt", json=request_payload)
    if response.status_code != 200:
        raise RuntimeError(f"ComfyUI request failed ({response.status_code}): {response.text[:300]}")
    payload = response.json()
    return {"status": "submitted", "response": payload}


@mcp.tool()
async def n8n_trigger(workflow_path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Trigger n8n webhook endpoint."""
    if not workflow_path.strip():
        raise ValueError("workflow_path must not be empty")
    base_url = os.getenv("N8N_WEBHOOK_BASE_URL", "http://127.0.0.1:5678/webhook").rstrip("/")
    url = f"{base_url}/{workflow_path.strip().lstrip('/')}"
    async with httpx.AsyncClient(timeout=45.0) as client:
        response = await client.post(url, json=payload or {})
    if response.status_code >= 400:
        raise RuntimeError(f"n8n trigger failed ({response.status_code}): {response.text[:300]}")
    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type:
        data = response.json()
    else:
        data = {"text": response.text}
    return {"status": "ok", "response": data}


@mcp.tool()
async def memory_search(query: str, limit: int = 5) -> dict[str, Any]:
    """Query FLAP memory store by semantic similarity."""
    store = Mem0Store()
    hits = await store.search_memories(query=query, limit=limit)
    return {
        "query": query,
        "results": [
            {
                "memory_id": hit.memory_id,
                "text": hit.text,
                "source": hit.source,
                "distance": hit.distance,
                "created_at": hit.created_at,
            }
            for hit in hits
        ],
    }


@mcp.tool()
async def screenpipe_query(query: str, limit: int = 20) -> dict[str, Any]:
    """Query Screenpipe indexed screen/audio history."""
    client = ScreenpipeClient()
    results = await client.query(query=query, limit=limit)
    return {
        "query": query,
        "results": [
            {
                "event_id": item.event_id,
                "content": item.content,
                "source": item.source,
                "timestamp": item.timestamp,
            }
            for item in results
        ],
    }


def main() -> None:
    """Run MCP server over stdio transport."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
