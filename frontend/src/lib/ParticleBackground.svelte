<script lang="ts">
  const particles = Array.from({ length: 24 }, (_, i) => ({
    id: i,
    left: `${Math.random() * 100}%`,
    top: `${Math.random() * 100}%`,
    size: `${4 + Math.random() * 10}px`,
    duration: `${10 + Math.random() * 12}s`,
    delay: `${Math.random() * 8}s`,
  }));
</script>

<div class="pointer-events-none fixed inset-0 -z-10 overflow-hidden">
  {#each particles as particle (particle.id)}
    <span
      class="absolute rounded-full bg-violet-300/40 blur-[1px] animate-float"
      style={`left:${particle.left};top:${particle.top};width:${particle.size};height:${particle.size};animation-duration:${particle.duration};animation-delay:${particle.delay};`}
    ></span>
  {/each}
</div>

<style>
  @keyframes float {
    0%,
    100% {
      transform: translateY(0) scale(1);
      opacity: 0.35;
    }
    50% {
      transform: translateY(-22px) scale(1.15);
      opacity: 0.75;
    }
  }

  .animate-float {
    animation-name: float;
    animation-iteration-count: infinite;
    animation-timing-function: ease-in-out;
  }
</style>
