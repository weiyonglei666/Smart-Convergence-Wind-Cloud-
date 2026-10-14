<template>
  <section class="panel-frame" :class="`panel-${tone || 'cyan'}`">
    <header class="panel-title"><span class="title-mark"></span><h2>{{ title }}</h2><span v-if="meta" class="panel-meta">{{ meta }}</span></header>
    <div class="panel-body"><slot /></div>
  </section>
</template>
<script setup lang="ts">
defineProps<{ title: string; meta?: string; tone?: 'cyan' | 'green' | 'orange' }>()
</script>
<style scoped>
.panel-frame{position:relative;min-height:100px;border:1px solid var(--line);background:linear-gradient(145deg,rgba(10,35,60,.78),rgba(4,16,30,.9));box-shadow:var(--glow),inset 0 1px rgba(130,225,255,.08);clip-path:polygon(0 0,calc(100% - 12px) 0,100% 12px,100% 100%,0 100%)}
.panel-title{display:flex;align-items:center;gap:9px;height:40px;padding:0 15px;border-bottom:1px solid rgba(70,188,230,.16)}
.panel-title h2{margin:0;font-size:13px;font-weight:650;letter-spacing:.12em;color:#dff8ff}.title-mark{width:4px;height:16px;background:var(--cyan);box-shadow:0 0 10px var(--cyan)}.panel-meta{margin-left:auto;color:var(--muted);font-size:10px;letter-spacing:.08em}.panel-body{padding:14px}.panel-green .title-mark{background:var(--green);box-shadow:0 0 10px var(--green)}.panel-orange .title-mark{background:var(--orange);box-shadow:0 0 10px var(--orange)}
</style>
