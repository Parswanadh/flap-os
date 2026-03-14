<script lang="ts">
  const agents = [
    { name: "supervisor", status: "running", detail: "Planning task graph" },
    { name: "research", status: "idle", detail: "Awaiting web query" },
    { name: "coder", status: "running", detail: "Preparing patch draft" },
    { name: "terminal", status: "running", detail: "Scanning PTY alerts" },
    { name: "rag", status: "idle", detail: "No retrieval request" },
  ];
</script>

<section class="glass rounded-2xl p-4 h-full flex flex-col gap-4">
  <header>
    <h2 class="text-xl font-semibold">Agent Graph</h2>
    <p class="text-xs text-slate-300">LangGraph execution state and reflection retries</p>
  </header>

  <div class="grid gap-3 md:grid-cols-2">
    {#each agents as agent}
      <article class="rounded-xl border border-slate-700 bg-slate-900/60 p-3">
        <div class="flex items-center justify-between">
          <h3 class="font-medium capitalize">{agent.name}</h3>
          <span class="text-xs px-2 py-1 rounded-lg {agent.status === 'running' ? 'bg-emerald-700/70' : 'bg-slate-700'}">
            {agent.status}
          </span>
        </div>
        <p class="text-sm text-slate-300 mt-2">{agent.detail}</p>
      </article>
    {/each}
  </div>
</section>
