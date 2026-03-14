<script lang="ts">
import AgentView from "$lib/AgentView.svelte";
import ChatView from "$lib/ChatView.svelte";
import MemoryView from "$lib/MemoryView.svelte";
import SettingsView from "$lib/SettingsView.svelte";
import TerminalView from "$lib/TerminalView.svelte";
import { activeView, type ViewName } from "../stores/app";

  const navItems: { id: ViewName; label: string }[] = [
    { id: "chat", label: "Chat" },
    { id: "terminal", label: "Terminal" },
    { id: "memory", label: "Memory" },
    { id: "agents", label: "Agents" },
    { id: "settings", label: "Settings" },
  ];
</script>

<main class="min-h-screen p-5 md:p-8">
  <header class="mb-5 flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
    <div>
      <h1 class="text-2xl md:text-3xl font-bold tracking-tight">FLAP OS</h1>
      <p class="text-sm text-slate-300">Sovereign personal AI control center</p>
    </div>
    <div class="glass rounded-xl p-1 flex gap-1 w-fit">
      {#each navItems as item}
        <button
          class="px-3 py-2 rounded-lg text-xs md:text-sm transition { $activeView === item.id ? 'bg-violet-600 text-white' : 'text-slate-300 hover:bg-slate-700/50'}"
          on:click={() => ($activeView = item.id)}
        >
          {item.label}
        </button>
      {/each}
    </div>
  </header>

  <section class="h-[calc(100vh-140px)]">
    {#if $activeView === "chat"}
      <ChatView />
    {:else if $activeView === "terminal"}
      <TerminalView />
    {:else if $activeView === "memory"}
      <MemoryView />
    {:else if $activeView === "agents"}
      <AgentView />
    {:else}
      <SettingsView />
    {/if}
  </section>
</main>
