# FLAP OS

FLAP is Parshu's sovereign personal AI stack: FastAPI brain, LangGraph supervisor, Telegram/voice interfaces, memory layer, terminal orchestration, MCP tools, and a Tauri + Svelte desktop UI.

## Architecture

- `backend/` — FastAPI app, LiteLLM routing, agents, memory, tools, voice, Telegram.
- `terminal-server/` — Node PTY server (`bash`, `claude`, `gemini`, `copilot`, `ollama`, `docker`) with WebSocket streaming + alert scanner.
- `mcp-servers/` — custom MCP server with filesystem/shell/memory/screenpipe/comfyui/n8n tools.
- `frontend/` — SvelteKit + Tailwind glassmorphism UI, plus Tauri shell in `src-tauri/`.
- `docker-compose.yml` — `flap-backend`, `flap-terminal`, `chromadb`, `ollama`, `n8n` with resource limits.

## Routing Profiles (LiteLLM)

- `fast_chat` → `groq/llama-3.3-70b`
- `code` → `ollama/deepseek-coder-v2`
- `reasoning` → `openrouter/anthropic/claude-sonnet-4-5`
- `long_context` → `mistral/mistral-large-latest`
- `offline` → `ollama/qwen2.5:3b`

Fallback chain: Groq → Mistral → OpenRouter → OpenAI → Ollama.

## Local Development (Windows + Conda FLAP)

```powershell
conda activate FLAP
cd D:\Projects\FLAP\flap-os
python -m pip install -r backend\requirements.txt
npm --prefix terminal-server install
npm --prefix frontend install
```

Run backend tests:

```powershell
conda activate FLAP
cd D:\Projects\FLAP\flap-os
python -m pytest -q tests
```

Run frontend build:

```powershell
conda activate FLAP
cd D:\Projects\FLAP\flap-os\frontend
npm run build
```

## Ubuntu Server Deployment

On the Dell server:

```bash
cd /home/parshu/flap-os
cp .env.example .env
# fill secrets in .env
bash scripts/setup_ubuntu.sh
bash scripts/tailscale_setup.sh
```

Systemd units shipped:

- `scripts/systemd/flap-compose.service`
- `scripts/systemd/screenpipe.service`

## Required Secrets

Copy `.env.example` to `.env` and set:

- `GROQ_API_KEY`
- `MISTRAL_API_KEY`
- `OPENROUTER_API_KEY`
- `OPENAI_API_KEY`
- `DEEPGRAM_API_KEY`
- `TELEGRAM_BOT_TOKEN`
- `PICOVOICE_ACCESS_KEY`
- `LANGSMITH_API_KEY`
- `GITHUB_TOKEN`

## Notes

- The backend is async-first (`httpx`, `aiofiles`, `aiosqlite`, `asyncio`).
- Memory stack uses ChromaDB + SQLite with semantic retrieval.
- Telegram `/run` blocks destructive commands unless explicitly forced.
