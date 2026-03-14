import cors from "cors";
import express from "express";
import http from "node:http";
import pty from "node-pty";
import { WebSocketServer } from "ws";

const PORT = Number(process.env.TERMINAL_SERVER_PORT ?? 3001);
const BUFFER_LIMIT = 10_000;
const STARTUP_DELAY_MS = 1_000;

const SESSION_NAMES = ["bash", "claude", "gemini", "copilot", "ollama", "docker"];

const app = express();
app.use(cors());
app.use(express.json({ limit: "1mb" }));

const server = http.createServer(app);
const wss = new WebSocketServer({ server, path: "/ws" });

const sessions = new Map();

function getShell() {
  return process.env.TERMINAL_SHELL ?? "/bin/bash";
}

function safeSessionPrompt(name) {
  return `export PS1='(${name}) \\u@\\h:\\w$ ';`;
}

function startupCommand(name) {
  const fallback = `${safeSessionPrompt(name)} exec bash`;
  switch (name) {
    case "claude":
      return `if command -v claude >/dev/null 2>&1; then exec claude; else echo '[FLAP] claude CLI not found'; ${fallback}; fi`;
    case "gemini":
      return `if command -v gemini >/dev/null 2>&1; then exec gemini; else echo '[FLAP] gemini CLI not found'; ${fallback}; fi`;
    case "copilot":
      return `if command -v copilot >/dev/null 2>&1; then exec copilot; else echo '[FLAP] copilot CLI not found'; ${fallback}; fi`;
    case "ollama":
      return `if command -v ollama >/dev/null 2>&1; then echo '[FLAP] ollama session ready'; ${fallback}; else echo '[FLAP] ollama not found'; ${fallback}; fi`;
    case "docker":
      return `if command -v docker >/dev/null 2>&1; then echo '[FLAP] docker session ready'; ${fallback}; else echo '[FLAP] docker not found'; ${fallback}; fi`;
    case "bash":
    default:
      return `${safeSessionPrompt(name)} exec bash`;
  }
}

function truncateBuffer(text) {
  if (text.length <= BUFFER_LIMIT) {
    return text;
  }
  return text.slice(text.length - BUFFER_LIMIT);
}

function nowIso() {
  return new Date().toISOString();
}

function looksLikeError(line) {
  return /\b(error|exception|failed)\b/i.test(line);
}

async function emitAlert(alertPayload) {
  const webhook = process.env.TERMINAL_ALERT_WEBHOOK_URL;
  if (!webhook) {
    return;
  }
  const response = await fetch(webhook, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(alertPayload),
  });
  if (!response.ok) {
    throw new Error(`Alert webhook failed (${response.status})`);
  }
}

function broadcast(event) {
  const payload = JSON.stringify(event);
  for (const client of wss.clients) {
    if (client.readyState === 1) {
      client.send(payload);
    }
  }
}

function createSession(name) {
  const shell = getShell();
  const ptyProcess = pty.spawn(shell, ["-lc", startupCommand(name)], {
    name: "xterm-256color",
    cols: 120,
    rows: 40,
    cwd: process.env.TERMINAL_CWD ?? process.env.HOME ?? "/home/parshu",
    env: process.env,
  });

  const state = {
    name,
    process: ptyProcess,
    buffer: "",
    startedAt: nowIso(),
    alive: true,
    lastErrorLine: null,
  };
  sessions.set(name, state);

  ptyProcess.onData((chunk) => {
    state.buffer = truncateBuffer(`${state.buffer}${chunk}`);
    broadcast({ type: "output", session: name, chunk, ts: nowIso() });

    const lines = chunk.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
    for (const line of lines) {
      if (!looksLikeError(line)) {
        continue;
      }
      state.lastErrorLine = line;
      const alert = {
        type: "terminal_alert",
        session: name,
        line,
        ts: nowIso(),
      };
      broadcast(alert);
      emitAlert(alert).catch((error) => {
        broadcast({
          type: "terminal_alert_failure",
          session: name,
          message: String(error),
          ts: nowIso(),
        });
      });
    }
  });

  ptyProcess.onExit(({ exitCode, signal }) => {
    state.alive = false;
    broadcast({
      type: "exit",
      session: name,
      exitCode,
      signal,
      ts: nowIso(),
    });
    setTimeout(() => {
      createSession(name);
      broadcast({ type: "restart", session: name, ts: nowIso() });
    }, STARTUP_DELAY_MS);
  });
}

for (const name of SESSION_NAMES) {
  createSession(name);
}

app.get("/health", (_req, res) => {
  res.json({
    status: "ok",
    sessions: SESSION_NAMES.length,
    active: Array.from(sessions.values()).filter((state) => state.alive).length,
  });
});

app.get("/sessions", (_req, res) => {
  const payload = Array.from(sessions.values()).map((state) => ({
    name: state.name,
    pid: state.process.pid,
    alive: state.alive,
    bufferLength: state.buffer.length,
    startedAt: state.startedAt,
    lastErrorLine: state.lastErrorLine,
  }));
  res.json({ sessions: payload });
});

app.get("/sessions/:name/buffer", (req, res) => {
  const session = sessions.get(req.params.name);
  if (!session) {
    res.status(404).json({ error: `Unknown session: ${req.params.name}` });
    return;
  }
  res.json({
    session: session.name,
    buffer: session.buffer,
    bufferLength: session.buffer.length,
  });
});

app.post("/sessions/:name/input", (req, res) => {
  const session = sessions.get(req.params.name);
  if (!session) {
    res.status(404).json({ error: `Unknown session: ${req.params.name}` });
    return;
  }
  const input = typeof req.body?.input === "string" ? req.body.input : "";
  if (!input) {
    res.status(400).json({ error: "Body must include non-empty string field 'input'" });
    return;
  }
  session.process.write(input);
  res.json({ ok: true, session: session.name, written: input.length });
});

app.post("/sessions/:name/resize", (req, res) => {
  const session = sessions.get(req.params.name);
  if (!session) {
    res.status(404).json({ error: `Unknown session: ${req.params.name}` });
    return;
  }
  const cols = Number(req.body?.cols);
  const rows = Number(req.body?.rows);
  if (!Number.isInteger(cols) || !Number.isInteger(rows) || cols < 20 || rows < 5) {
    res.status(400).json({ error: "cols and rows must be integers (cols>=20, rows>=5)" });
    return;
  }
  session.process.resize(cols, rows);
  res.json({ ok: true, session: session.name, cols, rows });
});

wss.on("connection", (socket) => {
  socket.send(
    JSON.stringify({
      type: "hello",
      sessions: SESSION_NAMES,
      ts: nowIso(),
    }),
  );

  socket.on("message", (rawMessage) => {
    const payload = String(rawMessage);
    let parsed;
    try {
      parsed = JSON.parse(payload);
    } catch (_error) {
      socket.send(JSON.stringify({ type: "error", message: "Invalid JSON payload" }));
      return;
    }

    if (parsed.type === "input") {
      const session = sessions.get(parsed.session);
      if (!session) {
        socket.send(JSON.stringify({ type: "error", message: `Unknown session: ${parsed.session}` }));
        return;
      }
      const input = typeof parsed.input === "string" ? parsed.input : "";
      if (!input) {
        socket.send(JSON.stringify({ type: "error", message: "input must be a non-empty string" }));
        return;
      }
      session.process.write(input);
      return;
    }

    if (parsed.type === "read_buffer") {
      const session = sessions.get(parsed.session);
      if (!session) {
        socket.send(JSON.stringify({ type: "error", message: `Unknown session: ${parsed.session}` }));
        return;
      }
      socket.send(
        JSON.stringify({
          type: "buffer",
          session: session.name,
          buffer: session.buffer,
          ts: nowIso(),
        }),
      );
      return;
    }

    socket.send(JSON.stringify({ type: "error", message: `Unsupported message type: ${parsed.type}` }));
  });
});

server.listen(PORT, "0.0.0.0", () => {
  console.log(`[FLAP terminal-server] listening on 0.0.0.0:${PORT}`);
  console.log(`[FLAP terminal-server] sessions: ${SESSION_NAMES.join(", ")}`);
});
