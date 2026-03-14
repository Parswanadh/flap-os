import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_required_paths_exist() -> None:
    required_paths = [
        "backend/main.py",
        "backend/agents/flap_core.py",
        "backend/agents/supervisor.py",
        "backend/agents/research_agent.py",
        "backend/agents/coder_agent.py",
        "backend/agents/terminal_agent.py",
        "backend/agents/browser_agent.py",
        "backend/agents/rag_agent.py",
        "backend/agents/reflection.py",
        "backend/memory/mem0_store.py",
        "backend/memory/screenpipe_client.py",
        "backend/memory/clipboard_watcher.py",
        "backend/tools/computer_control.py",
        "backend/tools/agent_s2_bridge.py",
        "backend/voice/stt.py",
        "backend/voice/tts.py",
        "backend/voice/wake_word.py",
        "backend/telegram/bot.py",
        "backend/router/llm_router.py",
        "backend/requirements.txt",
        "frontend/src/lib",
        "frontend/src/routes",
        "frontend/src/stores",
        "frontend/src-tauri",
        "terminal-server/index.js",
        "terminal-server/package.json",
        "mcp-servers/flap_mcp_server.py",
        "tests",
        "scripts/setup_ubuntu.sh",
        "scripts/tailscale_setup.sh",
        "docker-compose.yml",
        ".env.example",
        "README.md",
    ]
    missing = [path for path in required_paths if not (ROOT / path).exists()]
    assert not missing, f"Missing required paths: {missing}"


def test_env_example_contains_all_required_keys() -> None:
    env_text = (ROOT / ".env.example").read_text(encoding="utf-8")
    expected_keys = [
        "GROQ_API_KEY=",
        "MISTRAL_API_KEY=",
        "OPENROUTER_API_KEY=",
        "OPENAI_API_KEY=",
        "DEEPGRAM_API_KEY=",
        "TELEGRAM_BOT_TOKEN=",
        "PICOVOICE_ACCESS_KEY=",
        "LANGSMITH_API_KEY=",
        "GITHUB_TOKEN=",
    ]
    missing = [key for key in expected_keys if key not in env_text]
    assert not missing, f"Missing keys in .env.example: {missing}"


def test_compose_has_required_services_and_resource_limits() -> None:
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    expected_services = ["flap-backend:", "flap-terminal:", "chromadb:", "ollama:", "n8n:"]
    for service in expected_services:
        assert service in compose, f"{service} missing in docker-compose.yml"

    for service in ["flap-backend", "flap-terminal", "chromadb", "ollama", "n8n"]:
        match = re.search(
            rf"(?ms)^  {re.escape(service)}:\n(.*?)(?=^  [a-z0-9-]+:|^volumes:|^networks:|\Z)",
            compose,
        )
        assert match is not None, f"{service} section missing"
        section = match.group(1)
        assert "restart: always" in section, f"{service} must set restart: always"
        assert "mem_limit:" in section, f"{service} must set mem_limit"
