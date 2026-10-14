<template><span class="freshness" :class="`freshness-${freshness}`"><i></i>{{ labels[freshness] }}</span></template>
<script setup lang="ts">
import type { Freshness } from '../../api/types'
defineProps<{ freshness: Freshness }>()
const labels: Record<Freshness, string> = { realtime: '实时', near_realtime: '准实时', delayed: '延迟', offline: '离线', invalid: '异常', unavailable: '连接失败' }
</script>
<style scoped>.freshness{display:inline-flex;align-items:center;gap:6px;font-size:11px;color:#9db6c9}.freshness i{width:6px;height:6px;border-radius:50%;background:currentColor;box-shadow:0 0 8px currentColor}.freshness-realtime{color:var(--green)}.freshness-near_realtime{color:var(--cyan)}.freshness-delayed{color:var(--orange)}.freshness-offline,.freshness-invalid{color:var(--red)}.freshness-unavailable{color:var(--muted)}</style>
