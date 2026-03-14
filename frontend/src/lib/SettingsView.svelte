<script lang="ts">
  let backendUrl = "http://127.0.0.1:8000";
  let terminalUrl = "http://127.0.0.1:3001";
  let mcpGithub = true;
  let mcpPlaywright = true;
  let mcpBrave = true;
  let mcpSqlite = true;
</script>

<section class="glass rounded-2xl p-4 h-full flex flex-col gap-4">
  <header>
    <h2 class="text-xl font-semibold">Settings</h2>
    <p class="text-xs text-slate-300">Model routing, API endpoints, and MCP toggles</p>
  </header>

  <div class="grid gap-3 md:grid-cols-2">
    <label class="text-sm">
      <span class="block text-slate-300 mb-1">Backend URL</span>
      <input class="w-full bg-slate-900/80 border border-slate-600 rounded-lg p-2" bind:value={backendUrl} />
    </label>
    <label class="text-sm">
      <span class="block text-slate-300 mb-1">Terminal URL</span>
      <input class="w-full bg-slate-900/80 border border-slate-600 rounded-lg p-2" bind:value={terminalUrl} />
    </label>
  </div>

  <div class="space-y-2 text-sm">
    <label class="flex items-center gap-2"><input type="checkbox" bind:checked={mcpGithub} /> MCP: GitHub</label>
    <label class="flex items-center gap-2"><input type="checkbox" bind:checked={mcpPlaywright} /> MCP: Playwright</label>
    <label class="flex items-center gap-2"><input type="checkbox" bind:checked={mcpBrave} /> MCP: Brave Search</label>
    <label class="flex items-center gap-2"><input type="checkbox" bind:checked={mcpSqlite} /> MCP: SQLite</label>
  </div>
</section>
