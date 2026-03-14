import { writable } from "svelte/store";

export type ViewName = "chat" | "terminal" | "memory" | "agents" | "settings";

export const activeView = writable<ViewName>("chat");
export const selectedModel = writable("fast_chat");
export const voiceActive = writable(false);
export const terminalAlerts = writable<string[]>([]);

export const terminalSessions = writable([
  "bash",
  "claude",
  "gemini",
  "copilot",
  "ollama",
  "docker",
]);
