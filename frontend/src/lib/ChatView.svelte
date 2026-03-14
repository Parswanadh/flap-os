<script lang="ts">
  import { selectedModel, voiceActive } from "../stores/app";

  let input = "";
  let messages: { role: "user" | "assistant"; content: string }[] = [
    { role: "assistant", content: "🔔 FLAP online. Ask me anything." },
  ];

  function sendMessage() {
    const trimmed = input.trim();
    if (!trimmed) return;
    messages = [...messages, { role: "user", content: trimmed }];
    messages = [...messages, { role: "assistant", content: "✅ Request queued to backend /chat." }];
    input = "";
  }
</script>

<section class="glass rounded-2xl p-4 h-full flex flex-col gap-4">
  <header class="flex items-center justify-between">
    <div>
      <h2 class="text-xl font-semibold">Chat</h2>
      <p class="text-xs text-slate-300">FastAPI + FLAP personality + model routing</p>
    </div>
    <div class="flex items-center gap-3">
      <div class="h-3 w-3 rounded-full bg-violet-400 { $voiceActive ? 'animate-pulse' : '' }"></div>
      <select class="bg-slate-900/70 border border-slate-600 rounded-lg p-2 text-sm" bind:value={$selectedModel}>
        <option value="fast_chat">fast_chat</option>
        <option value="code">code</option>
        <option value="reasoning">reasoning</option>
        <option value="long_context">long_context</option>
        <option value="offline">offline</option>
      </select>
    </div>
  </header>

  <div class="flex-1 overflow-auto rounded-xl border border-slate-700/60 p-3 bg-slate-950/50 space-y-3">
    {#each messages as message, idx (idx)}
      <div class="text-sm leading-relaxed">
        <span class="font-semibold {message.role === 'user' ? 'text-cyan-300' : 'text-violet-300'}">
          {message.role === "user" ? "You" : "FLAP"}:
        </span>
        <span class="text-slate-100 ml-1">{message.content}</span>
      </div>
    {/each}
  </div>

  <div class="flex gap-2">
    <textarea
      class="flex-1 bg-slate-900/80 border border-slate-600 rounded-xl p-3 text-sm"
      bind:value={input}
      placeholder="Ask FLAP..."
      rows="2"
    ></textarea>
    <button class="bg-violet-600 hover:bg-violet-500 px-4 rounded-xl text-sm font-medium" on:click={sendMessage}>
      Send
    </button>
  </div>
</section>
