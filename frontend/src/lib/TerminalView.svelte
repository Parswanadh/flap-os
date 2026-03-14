<script lang="ts">
  import { terminalSessions } from "../stores/app";

  let selected = "bash";
</script>

<section class="glass rounded-2xl p-4 h-full flex flex-col gap-4">
  <header>
    <h2 class="text-xl font-semibold">Terminal Manager</h2>
    <p class="text-xs text-slate-300">6 PTY sessions with live monitoring and alerts</p>
  </header>

  <div class="flex flex-wrap gap-2">
    {#each $terminalSessions as session}
      <button
        class="px-3 py-1 rounded-lg text-xs border {selected === session ? 'bg-violet-600 border-violet-400' : 'bg-slate-900/70 border-slate-700'}"
        on:click={() => (selected = session)}
      >
        {session}
      </button>
    {/each}
  </div>

  <div class="flex-1 rounded-xl border border-slate-700 bg-black/70 p-3 overflow-auto font-mono text-xs">
    <p class="text-emerald-300">[{selected}] terminal output stream</p>
    <p class="text-slate-300 mt-2">xterm.js panel binds here in runtime build.</p>
    <p class="text-rose-300 mt-4">Error scanner status: active</p>
  </div>
</section>
