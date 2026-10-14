<template>
  <div class="map-wrap">
    <div ref="mapEl" class="map-canvas" aria-label="中国新能源场站地图"></div>
    <div class="map-label">中国新能源场站分布 · {{ selectedTypeLabel }}</div>
    <select v-model="selectedStationId" class="station-select" aria-label="选择场站"><option value="">全国场站</option><option v-for="station in stations" :key="station.station_id" :value="station.station_id">{{ station.station_id }}</option></select>
    <div v-if="selectedStation" class="city-card"><strong>{{ selectedStation.station_id }}</strong><span>{{ selectedStation.station_type === 'wind' ? '风电场站' : '光伏场站' }}</span><small>经度 {{ selectedStation.longitude.toFixed(3) }} · 纬度 {{ selectedStation.latitude.toFixed(3) }}</small><button @click="resetView">返回全国</button></div>
    <div class="map-legend"><span><i class="dot wind"></i>风电</span><span><i class="dot pv"></i>光伏</span></div>
  </div>
</template>
<style scoped>.station-select{appearance:none;position:absolute;right:14px;top:11px;z-index:4;min-width:150px;height:34px;padding:0 34px 0 13px;border:1px solid rgba(69,203,236,.5);border-radius:4px;background-color:rgba(5,26,45,.96);background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath d='m2 4 4 4 4-4' fill='none' stroke='%237fd9ed' stroke-width='1.4'/%3E%3C/svg%3E");background-repeat:no-repeat;background-position:right 12px center;box-shadow:0 0 0 1px rgba(49,182,220,.08),0 5px 18px rgba(0,0,0,.2);color:#d8f8ff;font-size:11px;letter-spacing:.04em;outline:none;cursor:pointer;transition:border-color .2s,box-shadow .2s}.station-select:hover,.station-select:focus{border-color:#55dcff;box-shadow:0 0 0 3px rgba(55,211,246,.12),0 6px 20px rgba(0,0,0,.25)}.station-select option{background:#071b2d;color:#d8f8ff}</style>
<script setup lang="ts">
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from 'vue'
import * as echarts from 'echarts'
import type { Station } from '../../api/types'
const props = defineProps<{ stations: Station[]; selectedId: string | null; selectedType: 'all' | 'wind' | 'pv' }>()
const emit = defineEmits<{ (event: 'select-station', stationId: string): void }>()
const selectedStationId = computed({ get: () => props.selectedId || '', set: (value: string) => emit('select-station', value) })
const mapEl = ref<HTMLElement | null>(null)
let chart: echarts.ECharts | null = null
const selectedTypeLabel = computed(() => props.selectedType === 'all' ? '全部类型' : props.selectedType === 'wind' ? '风电场站' : '光伏场站')
const mappedStations = computed(() => props.stations.filter((station) => props.selectedType === 'all' || station.station_type === props.selectedType))
const selectedStation = computed(() => props.stations.find((station) => station.station_id === props.selectedId) || null)
function option() {
  return { backgroundColor: 'transparent', tooltip: { trigger: 'item', formatter: (params: any) => params.data?.station_id || params.name || '' }, geo: { map: 'china', roam: true, zoom: selectedStation.value ? 3.2 : 1.05, center: selectedStation.value ? [selectedStation.value.longitude, selectedStation.value.latitude] : [105, 36], label: { show: false }, itemStyle: { areaColor: 'rgba(16,74,101,.48)', borderColor: '#3dbbdc', borderWidth: 0.7 }, emphasis: { itemStyle: { areaColor: 'rgba(43,151,178,.7)' } } }, series: [{ name: '场站', type: 'scatter', coordinateSystem: 'geo', symbolSize: (value: (number | string)[]) => selectedStation.value?.station_id === value[2] ? 16 : 10, data: mappedStations.value.map((station) => ({ name: station.station_id, station_id: station.station_id, value: [station.longitude, station.latitude, station.station_id], itemStyle: { color: station.station_type === 'wind' ? '#39d9ff' : '#3ee5b1', shadowBlur: 14, shadowColor: station.station_type === 'wind' ? '#39d9ff' : '#3ee5b1' } })) }] }
}
function render() { chart?.setOption(option(), true) }
function resetView() { emit('select-station', ''); nextTick(render) }
onMounted(async () => { if (!mapEl.value) return; chart = echarts.init(mapEl.value); try { const geo = await fetch('/geo/china.json').then((response) => response.json()); echarts.registerMap('china', geo); render(); chart.on('click', (params: any) => { if (params.data?.station_id) emit('select-station', params.data.station_id) }) } catch { /* retain the surrounding dashboard if the map asset is unavailable */ } window.addEventListener('resize', render) })
watch(() => [props.stations, props.selectedId, props.selectedType], render, { deep: true })
onUnmounted(() => { window.removeEventListener('resize', render); chart?.dispose(); chart = null })
</script>
<style scoped>.map-wrap{position:relative;height:100%;min-height:390px;overflow:hidden;background:radial-gradient(ellipse at 50% 48%,rgba(19,93,126,.28),transparent 60%),linear-gradient(135deg,rgba(5,28,50,.95),rgba(2,14,27,.98));border:1px solid rgba(77,202,238,.2)}.map-canvas{position:absolute;inset:0}.map-label{position:absolute;left:16px;top:14px;color:#a4cbe0;font-size:10px;letter-spacing:.1em;z-index:2}.city-card{position:absolute;left:16px;bottom:16px;z-index:3;display:flex;flex-direction:column;gap:5px;padding:12px 14px;min-width:180px;background:rgba(4,23,39,.88);border:1px solid rgba(255,180,91,.5);box-shadow:0 0 20px rgba(255,180,91,.12)}.city-card strong{color:var(--orange);font:15px ui-monospace,monospace}.city-card span{color:#c5e7ef;font-size:11px}.city-card small{color:#7696a6;font-size:9px}.city-card button{align-self:flex-start;margin-top:4px;color:#83d8eb;background:transparent;border:0;padding:0;font-size:10px;cursor:pointer}.map-legend{position:absolute;right:14px;bottom:12px;z-index:2;display:flex;gap:14px;color:#9ab7c9;font-size:10px}.dot{display:inline-block;width:7px;height:7px;margin-right:5px;border-radius:50%;background:#39d9ff;box-shadow:0 0 7px currentColor}.dot.pv{background:#3ee5b1}</style>
