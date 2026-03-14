<script lang="ts">
  let query = "";
  let timeline = [
    { ts: "10:30", source: "clipboard", text: "Docker compose limit tuning notes" },
    { ts: "10:12", source: "telegram", text: "Asked FLAP to run status command" },
    { ts: "09:55", source: "screenpipe", text: "Debugging llm_router retry path" },
  ];
</script>

<section class="glass rounded-2xl p-4 h-full flex flex-col gap-4">
  <header>
    <h2 class="text-xl font-semibold">Memory Timeline</h2>
    <p class="text-xs text-slate-300">mem0 + Chroma + Screenpipe context retrieval</p>
  </header>

  <input
    class="bg-slate-900/80 border border-slate-600 rounded-xl p-3 text-sm"
    bind:value={query}
    placeholder="Semantic memory search..."
  />

  <div class="flex-1 overflow-auto space-y-2">
    {#each timeline as item}
      <div class="rounded-xl border border-slate-700/80 bg-slate-900/50 p-3">
        <div class="text-xs text-slate-400">{item.ts} • {item.source}</div>
        <div class="text-sm mt-1">{item.text}</div>
      </div>
    {/each}
  </div>
</section>
