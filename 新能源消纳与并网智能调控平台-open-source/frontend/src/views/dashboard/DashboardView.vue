<template>
  <main class="screen-shell">
    <header class="screen-header"><div class="brand-lockup"><span class="brand-icon">⚡</span><div><h1>新能源消纳与并网智能调控平台</h1><p>NEW ENERGY CONSUMPTION &amp; GRID CONTROL</p></div></div><nav class="view-tabs"><button v-for="tab in tabs" :key="tab.key" :class="{active: activeTab === tab.key}" @click="activeTab = tab.key">{{ tab.label }}</button></nav><div class="header-actions"><div class="clock mono">{{ nowText }}</div><div data-testid="realtime-status" class="connection"><span v-if="!store.overview.updated_at" class="waiting-label">等待数据 · </span><DataFreshnessBadge :freshness="connectionFreshness" /><span class="update-time">{{ lastUpdate }}</span></div><RouterLink class="control-link" to="/assistant">智能调控中心 <span>↗</span></RouterLink><button class="admin-link">后台管理 <span>↗</span></button></div></header>

    <section class="screen-content">
      <MetricStrip :items="metrics" />
      <div class="main-grid">
        <aside class="left-rail">
          <PanelFrame title="运行概况" meta="11 个目标场站" tone="green"><div class="overview-ring"><div class="ring"><strong>{{ format(totalPower) }}</strong><small>MW 实时功率</small></div><div class="overview-list"><div><span>在线场站</span><b data-testid="station-count" class="green mono">{{ onlineCount }}/{{ stationTotal }}</b></div><div><span>风电出力</span><b class="mono">{{ format(windPower) }} <small>MW</small></b></div><div><span>光伏出力</span><b class="mono">{{ format(pvPower) }} <small>MW</small></b></div></div></div></PanelFrame>
          <PanelFrame title="未来气象预报" meta="远程节点最新 NWP / UTC+8"><div class="weather-grid"><div><span>🌡 温度</span><b>{{ nwpTemperature }}<small>°C</small></b></div><div><span>💨 风速</span><b>{{ nwpWind }}<small>m/s</small></b></div><div><span>☀ 辐照</span><b>{{ nwpRadiation }}<small>W/m²</small></b></div><div><span>☁ 云量</span><b>{{ nwpCloud }}<small>%</small></b></div></div><div class="weather-source">{{ selectedNwp?.station_id || 'FD0001' }} · 未来时刻 {{ nwpValidTime }}</div></PanelFrame>
          <PanelFrame :title="`极端天气告警 · ${selectedAlert.station_id}`" meta="当前场站 · 未来窗口" tone="orange"><div class="alarm-list"><div class="alarm-row" :class="`alarm-${selectedAlert.level}`"><span class="alarm-icon">{{ selectedAlert.level === 'normal' ? '✓' : '!' }}</span><div><strong>{{ selectedAlert.title }}</strong><small>{{ selectedAlert.detail }}</small></div></div></div></PanelFrame>
        </aside>

        <section class="center-stage"><div class="stage-toolbar"><div class="scope-select"><span class="scope-dot"></span>全国新能源资产 <span class="chevron">⌄</span></div><div class="map-tabs"><button v-for="type in mapTypes" :key="type.key" :class="{active: selectedType === type.key}" @click="selectedType = type.key">{{ type.label }}</button></div><div class="stage-meta">数据口径：15 min · <DataFreshnessBadge :freshness="connectionFreshness" /></div></div><StationMap :stations="displayStations" :selected-id="store.selectedStationId" :selected-type="selectedType" @select-station="selectStation" /><div class="stage-footer"><div><span>今日实际发电量</span><strong>— <small>GWh</small></strong></div><div><span>理论可发电量</span><strong>— <small>GWh</small></strong></div><div><span>当前功率缺口</span><strong>— <small>MW</small></strong></div><div class="footer-source"><span>实时更新</span><DataFreshnessBadge :freshness="connectionFreshness" /></div></div></section>

        <aside class="right-rail"><PanelFrame title="预测与实际功率" meta="今日 · 15 min"><div class="chart-head"><div><span>预测准确率</span><strong>—</strong></div><div class="chart-switch"><span class="active">功率</span><span>误差</span></div></div><PowerTrendChart :empty="!hasData" /></PanelFrame><PanelFrame title="风电 Top5" meta="按实时功率"><div class="ranking-list"><div v-for="(item, index) in rankingWind" :key="item.id" class="rank-row"><i>{{ `0${index + 1}` }}</i><span>{{ item.id }}</span><b>{{ item.value }}</b><em>MW</em></div><div v-if="!rankingWind.length" class="empty-note">等待真实功率数据</div></div></PanelFrame><PanelFrame title="光伏 Top5" meta="按实时功率" tone="green"><div class="ranking-list"><div v-for="(item, index) in rankingPv" :key="item.id" class="rank-row"><i class="pv-rank">{{ `0${index + 1}` }}</i><span>{{ item.id }}</span><b>{{ item.value }}</b><em>MW</em></div><div v-if="!rankingPv.length" class="empty-note">等待真实功率数据</div></div></PanelFrame></aside>
      </div>
      <section class="bottom-trend"><PanelFrame title="新能源功率趋势" meta="实时曲线"><PowerTrendChart :empty="!hasData" /></PanelFrame><PanelFrame title="消纳状态" :meta="selectedConsumption.station_id==='ALL'?'全场汇总':selectedConsumption.station_id" tone="green"><div class="liquid-pair"><div class="liquid-item"><div class="liquid-ball green" :class="{missing:selectedConsumption.rate==null}" :style="{'--level':`${100-Math.max(0,Math.min(100,selectedConsumption.rate||0))}%`}"><div class="liquid-wave wave-one"></div><div class="liquid-wave wave-two"></div><strong>{{selectedConsumption.rate==null?'—':selectedConsumption.rate.toFixed(1)+'%'}}</strong></div><span>消纳率</span></div><div class="liquid-item"><div class="liquid-ball cyan" :class="{missing:selectedConsumption.curtailment==null}" :style="{'--level':`${100-Math.max(0,Math.min(100,selectedConsumption.curtailment||0))}%`}"><div class="liquid-wave wave-one"></div><div class="liquid-wave wave-two"></div><strong>{{selectedConsumption.curtailment==null?'—':selectedConsumption.curtailment.toFixed(1)+'%'}}</strong></div><span>弃电率</span></div></div></PanelFrame></section>
    <div class="assistant-dock" :class="{ open: assistantOpen, dragging: assistantDragging }" :style="assistantDockStyle"><button class="assistant-float" aria-label="打开智能调控助手" @pointerdown="startAssistantDrag" @click="toggleAssistant"><span class="assistant-pulse"></span><img src="/assets/dispatch-assistant.png" alt="智能调控助手" draggable="false" /><span class="assistant-label">智能调控助手</span></button><div v-if="assistantOpen" class="assistant-popover"><div class="assistant-popover-head"><div><strong>智能调控助手</strong><small>实时读取功率 · 气象 · 限电状态</small></div><span class="assistant-live"><i></i>在线</span></div><div class="assistant-signal"><span>当前场站</span><b>{{ store.selectedStationId || "全场场站" }}</b><span>消纳率</span><b>{{ consumption?.consumption_rate == null ? "—" : consumption.consumption_rate.toFixed(1) + "%" }}</b></div><p>我可以结合实时数据和调控知识，分析消纳、限电、并网风险与预测偏差。</p><RouterLink to="/assistant" class="assistant-enter">进入智能调控中心 <span>↗</span></RouterLink></div></div>
    </section>
  </main>
</template>

<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { RouterLink } from 'vue-router'
import { useRouter } from 'vue-router'
import PanelFrame from '../../components/dashboard/PanelFrame.vue'
import DataFreshnessBadge from '../../components/dashboard/DataFreshnessBadge.vue'
import MetricStrip from '../../components/dashboard/MetricStrip.vue'
import StationMap from '../../components/dashboard/StationMap.vue'
import PowerTrendChart from '../../components/dashboard/PowerTrendChart.vue'
import { platformApi } from '../../api/platform'
import { usePlatformStore } from '../../stores/platform'
import type { ConsumptionOverview, Freshness, Station, NwpLatest } from '../../api/types'

const store = usePlatformStore()
const assistantOpen = ref(false)
const assistantDragging = ref(false)
const assistantPosition = ref<{ x: number; y: number } | null>(null)
const assistantDockStyle = computed(() => assistantPosition.value ? { left: `${assistantPosition.value.x}px`, top: `${assistantPosition.value.y}px`, right: 'auto', bottom: 'auto' } : undefined)
let assistantPointerOffset = { x: 0, y: 0 }
let assistantDragOrigin = { x: 0, y: 0 }
let assistantSuppressClick = false

function clampAssistantPosition(x: number, y: number) {
  return {
    x: Math.min(Math.max(12, x), Math.max(12, window.innerWidth - 116)),
    y: Math.min(Math.max(12, y), Math.max(12, window.innerHeight - 132)),
  }
}

function startAssistantDrag(event: PointerEvent) {
  if (event.button !== 0) return
  const button = event.currentTarget as HTMLElement
  const dock = button.closest('.assistant-dock') as HTMLElement
  const rect = dock.getBoundingClientRect()
  assistantPosition.value = { x: rect.left, y: rect.top }
  assistantPointerOffset = { x: event.clientX - rect.left, y: event.clientY - rect.top }
  assistantDragOrigin = { x: event.clientX, y: event.clientY }
  assistantDragging.value = true
  window.addEventListener('pointermove', moveAssistant)
  window.addEventListener('pointerup', finishAssistantDrag, { once: true })
  event.preventDefault()
}

function moveAssistant(event: PointerEvent) {
  if (!assistantDragging.value) return
  const distance = Math.hypot(event.clientX - assistantDragOrigin.x, event.clientY - assistantDragOrigin.y)
  if (distance > 4) assistantSuppressClick = true
  assistantPosition.value = clampAssistantPosition(event.clientX - assistantPointerOffset.x, event.clientY - assistantPointerOffset.y)
}

function finishAssistantDrag() {
  assistantDragging.value = false
  window.removeEventListener('pointermove', moveAssistant)
  if (assistantPosition.value) localStorage.setItem('assistant-dock-position', JSON.stringify(assistantPosition.value))
  window.setTimeout(() => { assistantSuppressClick = false }, 0)
}

function toggleAssistant() {
  if (!assistantSuppressClick) assistantOpen.value = !assistantOpen.value
}

function restoreAssistantPosition() {
  try {
    const saved = JSON.parse(localStorage.getItem('assistant-dock-position') || 'null')
    if (Number.isFinite(saved?.x) && Number.isFinite(saved?.y)) assistantPosition.value = clampAssistantPosition(saved.x, saved.y)
  } catch { /* ignore invalid saved position */ }
}

function keepAssistantInViewport() {
  if (assistantPosition.value) assistantPosition.value = clampAssistantPosition(assistantPosition.value.x, assistantPosition.value.y)
}
const nwp = ref<NwpLatest | null>(null)
const consumption = ref<ConsumptionOverview | null>(null)
const selectedConsumption = computed(() => { const id = store.selectedStationId; const item = id ? consumption.value?.station_breakdown?.find((row) => row.station_id === id) : null; if (item) { const anomaly = item.status === 'anomaly'; const missing = item.status === 'missing_fields'; return { station_id: item.station_id, rate: item.consumption_rate, curtailment: item.curtailment_rate, coverage: missing ? '字段缺失' : anomaly ? '口径异常' : '场站独立计算', note: missing ? '该场站可利用功率或实际功率缺失，暂不计算场站消纳率' : anomaly ? '实际功率高于可利用功率，暂不计算消纳率' : '数据来自该场站最新 15 min 实时功率口径' } } return { station_id: 'ALL', rate: consumption.value?.consumption_rate ?? null, curtailment: consumption.value?.curtailment_rate ?? null, coverage: `${consumption.value?.comparable_station_count || 0}/${consumption.value?.station_count || 0} 场站`, note: consumption.value?.warning || '正在读取实际 / 理论 / 可利用功率' } })
const selectedNwp = computed(() => nwp.value?.stations.find((item) => item.station_id === (store.selectedStationId || 'FD0001')) || nwp.value?.stations[0])
const nwpForecastWindow = computed(() => { const all = selectedNwp.value?.forecast || []; const future = all.filter((item) => Date.parse(item.valid_time_utc) >= Date.now() - 15 * 60 * 1000); return future.length ? future.slice(0, 16) : all.slice(-16) })
const nwpVariables = computed(() => nwpForecastWindow.value[0]?.variables || selectedNwp.value?.variables || {})
const nwpTemperature = computed(() => { const value = Number(nwpVariables.value.t2m); return Number.isFinite(value) ? (value > 150 ? (value - 273.15).toFixed(1) : value.toFixed(1)) : '—' })
const nwpWind = computed(() => { const u = Number(nwpVariables.value.u10), v = Number(nwpVariables.value.v10); return Number.isFinite(u) && Number.isFinite(v) ? Math.hypot(u, v).toFixed(1) : '—' })
const nwpRadiation = computed(() => { const value = Number(nwpVariables.value.ssrd); return Number.isFinite(value) ? (value > 10000 ? (value / 3600).toFixed(1) : value.toFixed(1)) : '—' })
const nwpCloud = computed(() => { const value = Number(nwpVariables.value.tcc); return Number.isFinite(value) ? (value * 100).toFixed(0) : '—' })
const nwpValidTime = computed(() => { const value = nwpForecastWindow.value[0]?.valid_time_utc; return value ? new Date(value).toLocaleString('zh-CN',{month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',hour12:false}) : '加载中' })
const weatherAlerts = computed(() => {
  if (!nwp.value?.stations?.length) return displayStations.value.map((station) => ({ key: station.station_id, station_id: station.station_id, level: 'normal' as const, title: '等待最新 NWP', detail: '正在读取该场站的最新预报时间轴' }))
  return nwp.value.stations.map((station) => {
  const all = station.forecast || []
  const future = all.filter((item) => Date.parse(item.valid_time_utc) >= Date.now() - 15 * 60 * 1000).slice(0, 16)
  const forecast = future.length ? future : all.slice(-16)
  const values = forecast.map((item) => {
    const vars = item.variables || {}, u = Number(vars.u10), v = Number(vars.v10), t = Number(vars.t2m), cloud = Number(vars.tcc)
    return { wind: Number.isFinite(u) && Number.isFinite(v) ? Math.hypot(u, v) : null, temp: Number.isFinite(t) ? (t > 150 ? t - 273.15 : t) : null, cloud: Number.isFinite(cloud) ? cloud * 100 : null }
  })
  // Extreme-weather status is near-term operational risk, not a warning for
  // any cloud value somewhere in the full 48h file. Use the next four NWP
  // slots (the visible short-term window) and require sustained cloud cover.
  const nearTerm = values.slice(0, 4)
  const maxWind = Math.max(...nearTerm.map((v) => v.wind ?? -Infinity)), maxTemp = Math.max(...nearTerm.map((v) => v.temp ?? -Infinity)), minTemp = Math.min(...nearTerm.map((v) => v.temp ?? Infinity)), maxCloud = Math.max(...nearTerm.map((v) => v.cloud ?? -Infinity)), sustainedCloud = nearTerm.filter((v) => (v.cloud ?? -Infinity) >= 95).length
  let level: 'warning' | 'danger' | 'normal' = 'normal', title = '正常', detail = forecast.length ? `未来 ${forecast.length} 个时次未超过风温云量告警阈值` : '暂无可用 NWP 未来时次'
  if (maxWind >= 17.2) { level = 'danger'; title = '大风风险'; detail = `未来最大风速 ${maxWind.toFixed(1)} m/s，超过 8 级风阈值` }
  else if (maxWind >= 13.9) { level = 'warning'; title = '强风提醒'; detail = `未来最大风速 ${maxWind.toFixed(1)} m/s，请关注风机运行` }
  else if (maxTemp >= 35) { level = 'warning'; title = '高温提醒'; detail = `未来最高气温 ${maxTemp.toFixed(1)} °C` }
  else if (minTemp <= -10) { level = 'warning'; title = '低温提醒'; detail = `未来最低气温 ${minTemp.toFixed(1)} °C` }
  else if (sustainedCloud >= 2) { level = 'warning'; title = '持续高云量提醒'; detail = `未来近端 ${sustainedCloud} 个时次云量≥95%，建议关注辐照波动` }
  return { key: station.station_id, station_id: station.station_id, level, title, detail }
  })
})
const selectedAlert = computed(() => weatherAlerts.value.find((item) => item.station_id === (store.selectedStationId || 'FD0001')) || weatherAlerts.value[0] || { station_id: store.selectedStationId || 'FD0001', level: 'normal' as const, title: '等待最新 NWP', detail: '正在读取该场站的最新预报时间轴' })
const router = useRouter()
const activeTab = ref('power')
const selectedType = ref<'all' | 'wind' | 'pv'>('all')
const tabs = [{ key: 'power', label: '功率监测' }, { key: 'weather', label: '气象观测' }, { key: 'consumption', label: '消纳分析' }]
const mapTypes = [{ key: 'all' as const, label: '风光资产' }, { key: 'wind' as const, label: '🌀 风电场站' }, { key: 'pv' as const, label: '☀ 光伏场站' }]
const fallbackStations: Station[] = [
  ['FD0001','wind',117.359,39.2483,30],['FD0002','wind',117.612,39.5231,12],['FD0003','wind',117.822,39.5586,20],['FD0061','wind',117.53,39.4514,50],['FD0063','wind',115.835,37.1339,100],['FD0082','wind',122.19,40.65,300],['FD0083','wind',122.268,41.7631,600],['FD0084','wind',122.725,41.8444,300],['FD0085','wind',122.531,41.5105,50],['GF0050','pv',100.296829,36.057195,1000],['GF0051','pv',117.529761,39.451369,49.8],
].map(([station_id, station_type, longitude, latitude, capacity_mw]) => ({ station_id: String(station_id), station_type: station_type as 'wind' | 'pv', name: String(station_id), longitude: Number(longitude), latitude: Number(latitude), capacity_mw: Number(capacity_mw), enabled: true }))
const displayStations = computed(() => store.stations.length ? store.stations : fallbackStations)
const stationTotal = computed(() => displayStations.value.length)
const snapshots = computed(() => store.overview.stations)
const hasData = computed(() => snapshots.value.some(s => s.active_power_mw !== null))
const onlineCount = computed(() => snapshots.value.filter(s => s.observed_at && s.freshness !== 'offline' && s.freshness !== 'invalid').length)
const totalPower = computed(() => sumType('all'))
const windPower = computed(() => sumType('wind'))
const pvPower = computed(() => sumType('pv'))
function sumType(type: 'all' | 'wind' | 'pv') { return snapshots.value.filter(s => type === 'all' || displayStations.value.find(st => st.station_id === s.station_id)?.station_type === type).reduce((sum, s) => sum + (s.active_power_mw || 0), 0) || null }
const rankingWind = computed(() => ranking('wind'))
const rankingPv = computed(() => ranking('pv'))
function ranking(type: 'wind' | 'pv') { return snapshots.value.filter(s => displayStations.value.find(st => st.station_id === s.station_id)?.station_type === type && s.active_power_mw !== null).sort((a,b) => (b.active_power_mw || 0) - (a.active_power_mw || 0)).slice(0,5).map(s => ({ id: s.station_id, value: (s.active_power_mw || 0).toFixed(2) })) }
const connectionFreshness = computed<Freshness>(() => store.connected ? (store.isRealtime ? 'realtime' : 'near_realtime') : (hasData.value ? 'delayed' : 'unavailable'))
const now = ref(new Date())
const nowText = computed(() => now.value.toLocaleTimeString('zh-CN', { hour12: false }))
const lastUpdate = computed(() => store.overview.updated_at ? new Date(store.overview.updated_at).toLocaleTimeString('zh-CN', { hour12: false }) : '等待数据')
const metrics = computed(() => [{ label: '新能源总功率', value: format(totalPower.value), unit: 'MW', foot: hasData.value ? '实时汇总' : '等待实时数据', state: hasData.value ? 'good' : '' }, { label: '风电实时功率', value: format(windPower.value), unit: 'MW', foot: 'FD 系列场站', state: '' }, { label: '光伏实时功率', value: format(pvPower.value), unit: 'MW', foot: 'GF 系列场站', state: '' }, { label: '新能源消纳率', value: consumption.value?.consumption_rate == null ? '—' : consumption.value.consumption_rate.toFixed(2), unit: '%', foot: consumption.value?.consumption_rate == null ? '等待消纳数据' : '实时消纳统计', state: consumption.value?.consumption_rate != null ? 'good' : 'warn' }, { label: '数据源状态', value: String(onlineCount.value), unit: '在线', foot: `${stationTotal.value} 个目标场站`, state: onlineCount.value === stationTotal.value ? 'good' : 'warn' }])

function format(value: number | null) { return value === null ? '—' : value.toFixed(2) }
function selectStation(id: string) { store.selectStation(id) }
function goAdmin() { router.push('/admin') }
function handleViewTab(event: Event) { const label = (event.currentTarget as HTMLElement).textContent || ''; if (label.includes('气象')) router.push('/weather'); else if (label.includes('消纳')) router.push('/analysis'); else router.push('/dashboard') }
function relabelObservation() {
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT)
  const nodes: Text[] = []
  while (walker.nextNode()) nodes.push(walker.currentNode as Text)
  nodes.forEach((node) => { node.nodeValue = node.nodeValue?.replaceAll('实时天气预报', '实时气象观测').replaceAll('选择场站后展示最新 NWP 网格结果', '选择场站后展示最新观测数据').replaceAll('等待数据 ·', '正在连接 ·') || '' })
}
let clockTimer: number | undefined
let nwpTimer: number | undefined
let consumptionTimer: number | undefined
async function loadNwp() { try { nwp.value = await platformApi.nwpLatest() } catch { /* keep last successful NWP */ } }
async function loadConsumption() { try { consumption.value = await platformApi.consumption() } catch { /* keep last successful aggregate */ } }
onMounted(() => { document.querySelector('.admin-link')?.addEventListener('click', goAdmin); document.querySelectorAll('.view-tabs button').forEach((button) => button.addEventListener('click', handleViewTab)); restoreAssistantPosition(); window.addEventListener('resize', keepAssistantInViewport); relabelObservation(); if (import.meta.env.MODE !== 'test') { store.loadOverview().catch(() => undefined); store.connectRealtime(); loadNwp(); loadConsumption(); nwpTimer = window.setInterval(loadNwp, 5000); consumptionTimer = window.setInterval(loadConsumption, 15000) } clockTimer = window.setInterval(() => { now.value = new Date() }, 1000) })
onUnmounted(() => { document.querySelector('.admin-link')?.removeEventListener('click', goAdmin); document.querySelectorAll('.view-tabs button').forEach((button) => button.removeEventListener('click', handleViewTab)); window.removeEventListener('resize', keepAssistantInViewport); window.removeEventListener('pointermove', moveAssistant); window.removeEventListener('pointerup', finishAssistantDrag); store.disconnectRealtime(); if (clockTimer) window.clearInterval(clockTimer); if (nwpTimer) window.clearInterval(nwpTimer); if (consumptionTimer) window.clearInterval(consumptionTimer) })
</script>

<style scoped>
.assistant-dock{position:fixed;z-index:30;right:24px;bottom:24px}.assistant-float{position:relative;width:104px;height:104px;border-radius:50%;display:grid;place-items:center;background:radial-gradient(circle,rgba(48,211,188,.2),rgba(5,28,45,.97) 66%);border:1px solid rgba(67,231,192,.72);box-shadow:0 0 28px rgba(46,222,191,.28),inset 0 0 18px rgba(54,190,211,.16);cursor:pointer;animation:floatPulse 3.4s ease-in-out infinite}.assistant-float img{width:92px;height:92px;object-fit:contain;filter:drop-shadow(0 0 12px rgba(75,241,217,.55));transition:transform .28s ease}.assistant-float:hover img,.assistant-dock.open .assistant-float img{transform:scale(1.08) rotate(-4deg)}.assistant-label{position:absolute;bottom:-20px;white-space:nowrap;color:#74e9cc;font-size:10px;letter-spacing:.08em}.assistant-pulse{position:absolute;inset:-8px;border:1px solid rgba(71,219,194,.25);border-radius:50%;animation:assistantRing 2.4s ease-out infinite;pointer-events:none}.assistant-popover{position:absolute;right:0;bottom:132px;width:286px;padding:15px;background:rgba(5,24,39,.97);border:1px solid rgba(72,211,193,.4);box-shadow:0 14px 40px rgba(0,0,0,.38),0 0 24px rgba(48,208,191,.13);backdrop-filter:blur(12px)}.assistant-popover-head{display:flex;justify-content:space-between;align-items:start;padding-bottom:11px;border-bottom:1px solid rgba(78,177,195,.16)}.assistant-popover-head strong,.assistant-popover-head small{display:block}.assistant-popover-head strong{font-size:13px;color:#e0fff8}.assistant-popover-head small{margin-top:5px;color:#7899a8;font-size:9px}.assistant-live{color:#59dfb6;font-size:10px}.assistant-live i{display:inline-block;width:6px;height:6px;margin-right:5px;border-radius:50%;background:#59dfb6;box-shadow:0 0 8px #59dfb6}.assistant-signal{display:grid;grid-template-columns:1fr auto;gap:8px 12px;padding:13px 0;color:#7899a8;font-size:10px}.assistant-signal b{color:#dffbff;font:12px ui-monospace,monospace}.assistant-popover p{margin:0;padding:10px;background:rgba(45,145,152,.12);color:#9bc4c8;font-size:10px;line-height:1.6}.assistant-enter{display:flex;justify-content:space-between;margin-top:12px;padding:10px 11px;background:#3eddb5;color:#062019;text-decoration:none;font-size:10px;font-weight:700}.assistant-enter span{font-size:14px}@keyframes assistantRing{0%{transform:scale(.8);opacity:.8}100%{transform:scale(1.28);opacity:0}}@keyframes floatPulse{0%,100%{transform:translateY(0)}50%{transform:translateY(-5px)}}.screen-shell{min-height:100vh;background:transparent;color:#e6f7ff}.screen-header{height:82px;display:flex;align-items:center;padding:0 28px;border-bottom:1px solid rgba(70,188,230,.22);background:linear-gradient(180deg,rgba(4,19,35,.96),rgba(4,19,35,.72));position:relative}.screen-header:after{content:'';position:absolute;left:8%;right:8%;bottom:-1px;height:1px;background:linear-gradient(90deg,transparent,var(--cyan),transparent);opacity:.7}.brand-lockup{display:flex;align-items:center;gap:12px;min-width:390px}.brand-icon{font-size:29px;color:#ffe56b;text-shadow:0 0 16px #ffbb38}.brand-lockup h1{margin:0;font-size:21px;letter-spacing:.1em;font-weight:700}.brand-lockup p{margin:7px 0 0;color:#5e89a4;font-size:9px;letter-spacing:.28em}.view-tabs{display:flex;align-self:stretch;align-items:center;gap:4px;margin-left:22px}.view-tabs button,.map-tabs button{height:34px;padding:0 16px;border:0;background:transparent;color:#7d9aae;font-size:12px;cursor:pointer}.view-tabs button.active{color:#ddf9ff;background:rgba(50,178,224,.13);box-shadow:inset 0 -2px var(--cyan)}.header-actions{margin-left:auto;display:flex;align-items:center;gap:18px}.clock{color:#a9d9e9;font-size:15px}.connection{display:flex;flex-direction:column;align-items:flex-end;gap:4px}.update-time{color:#5d8195;font-size:9px}.admin-link{border:1px solid rgba(85,194,230,.28);background:rgba(30,116,154,.1);color:#9ed7e8;padding:8px 12px;font-size:11px;cursor:pointer}.screen-content{padding:18px 24px 24px;max-width:1920px;margin:auto}.main-grid{display:grid;grid-template-columns:minmax(250px,.85fr) minmax(520px,2.1fr) minmax(250px,.85fr);gap:12px;margin-top:12px}.left-rail,.right-rail{display:grid;align-content:start;gap:12px}.center-stage{min-width:0}.stage-toolbar{display:flex;align-items:center;gap:14px;height:45px;padding:0 12px;border:1px solid rgba(70,188,230,.2);border-bottom:0;background:rgba(5,22,39,.78)}.scope-select{color:#bfe8f2;font-size:12px}.scope-dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--cyan);box-shadow:0 0 8px var(--cyan);margin-right:7px}.chevron{margin-left:8px;color:var(--muted)}.map-tabs{display:flex;margin-left:auto}.map-tabs button{height:28px;padding:0 10px;font-size:10px}.map-tabs button.active{color:var(--cyan);border-bottom:1px solid var(--cyan)}.stage-meta{color:#6f91a3;font-size:10px;display:flex;align-items:center;gap:7px}.stage-footer{display:grid;grid-template-columns:repeat(3,1fr) 1.2fr;gap:1px;border:1px solid rgba(70,188,230,.2);border-top:0;background:rgba(48,157,193,.08)}.stage-footer>div{padding:13px 14px;background:rgba(5,24,41,.75)}.stage-footer span{display:block;color:#7697a9;font-size:10px}.stage-footer strong{display:block;margin-top:6px;color:#dffaff;font:18px ui-monospace,monospace}.stage-footer small{font:10px inherit;color:var(--cyan-soft)}.footer-source{display:flex;flex-direction:column;justify-content:center;gap:8px}.overview-ring{display:flex;align-items:center;gap:18px}.ring{width:116px;height:116px;border-radius:50%;display:flex;flex-direction:column;align-items:center;justify-content:center;border:1px solid rgba(54,226,181,.42);box-shadow:inset 0 0 28px rgba(54,226,181,.09),0 0 20px rgba(54,226,181,.08)}.ring strong{font:22px ui-monospace,monospace;color:var(--green)}.ring small{margin-top:7px;color:#7394a6;font-size:9px}.overview-list{flex:1}.overview-list>div{display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid rgba(120,180,200,.1);color:#7795a7;font-size:10px}.overview-list b{color:#d8f6ff;font:13px ui-monospace,monospace}.overview-list b.green{color:var(--green)}.overview-list small{font:9px inherit;color:var(--muted)}.weather-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}.weather-grid>div{display:flex;flex-direction:column;gap:8px;padding:10px;background:rgba(16,67,91,.22)}.weather-grid span{font-size:10px;color:#7e9daf}.weather-grid b{font:17px ui-monospace,monospace;color:#d5f6ff}.weather-grid small{margin-left:3px;color:var(--cyan);font:9px sans-serif}.weather-source{padding:8px 0 0;color:#65a7bc;font-size:9px}.empty-note{padding:14px 0;color:#67879a;font-size:10px;text-align:center}.alarm-empty{display:flex;align-items:center;gap:12px;padding:5px}.alarm-check{display:grid;place-items:center;width:30px;height:30px;border-radius:50%;color:var(--green);border:1px solid rgba(55,231,174,.36);font-size:18px}.alarm-empty strong,.alarm-empty small{display:block}.alarm-empty strong{color:#b5d2df;font-size:11px}.alarm-empty small{margin-top:6px;color:#6c899b;font-size:10px}.chart-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:5px}.chart-head span{display:block;color:#7d9daf;font-size:10px}.chart-head strong{display:block;margin-top:4px;color:var(--green);font:20px ui-monospace,monospace}.chart-switch{display:flex;gap:10px}.chart-switch span{display:inline!important;padding-bottom:5px}.chart-switch .active{color:var(--cyan);border-bottom:1px solid var(--cyan)}.ranking-list{display:flex;flex-direction:column}.rank-row{display:grid;grid-template-columns:25px 1fr auto 28px;gap:7px;align-items:center;padding:8px 0;border-bottom:1px solid rgba(100,180,210,.1);font-size:11px}.rank-row i{font:10px ui-monospace,monospace;color:#5f8396;font-style:normal}.rank-row span{color:#bddde7}.rank-row b{color:#d9f8ff;font:12px ui-monospace,monospace}.rank-row em{color:#68889b;font-style:normal;font-size:9px}.rank-row .pv-rank{color:var(--green)}.bottom-trend{display:grid;grid-template-columns:2fr 1fr;gap:12px;margin-top:12px}.consumption-placeholder{display:grid;grid-template-columns:100px 1fr;gap:10px;align-items:center;min-height:115px}.consumption-number{grid-row:span 2;color:var(--green);font:31px ui-monospace,monospace}.consumption-number small{font:11px sans-serif}.consumption-placeholder strong,.consumption-placeholder span{display:block}.consumption-placeholder strong{color:#c5e9f2;font-size:12px}.consumption-placeholder span{margin-top:6px;color:#6f8d9d;font-size:10px}.consumption-bar{grid-column:2;height:5px;background:rgba(104,172,195,.14);overflow:hidden}.consumption-bar i{display:block;height:100%;background:var(--green);box-shadow:0 0 10px var(--green)}.consumption-live{min-height:150px}.consumption-live-head{display:flex;justify-content:space-between;align-items:center;padding-bottom:9px;border-bottom:1px solid rgba(85,194,230,.18);font-size:10px}.live-title{color:#dffaff;font-weight:700}.live-type{color:#3ee5b1}.consumption-live-main{display:flex;align-items:center;gap:16px;padding:16px 0 10px}.consumption-ring{position:relative;display:grid;place-items:center;width:82px;height:82px;border-radius:50%;background:conic-gradient(#3ee5b1 var(--rate),rgba(82,150,173,.15) 0);box-shadow:0 0 16px rgba(62,229,177,.16)}.consumption-ring:after{content:"";position:absolute;inset:7px;border-radius:50%;background:#071a2b;border:1px solid rgba(62,229,177,.25)}.consumption-ring strong,.consumption-ring small{position:relative;z-index:1}.consumption-ring strong{color:#3ee5b1;font:24px ui-monospace,monospace}.consumption-ring small{color:#3ee5b1;font-size:9px}.consumption-ring.missing{background:conic-gradient(#7896a8 100%,rgba(82,150,173,.15) 0)}.consumption-ring.missing strong,.consumption-ring.missing small{color:#7896a8}.consumption-live-copy{flex:1;min-width:0}.consumption-live-copy strong,.consumption-live-copy span{display:block}.consumption-live-copy strong{color:#dffaff;font-size:12px}.consumption-live-copy span{margin-top:6px;color:#7896a8;font-size:9px}.consumption-live-copy .consumption-bar{margin-top:12px}.consumption-live .consumption-note{display:block;grid-column:auto}.consumption-live .consumption-bar{height:5px}.consumption-note{grid-column:2;color:#7896a8;font-size:9px;line-height:1.4}
.alarm-list{display:flex;flex-direction:column;gap:7px;max-height:330px;overflow:auto;padding-right:3px}.alarm-row{display:flex;align-items:center;gap:9px;padding:9px 8px;background:rgba(16,67,91,.2);border-left:2px solid #3ee5b1}.alarm-row.alarm-warning{border-left-color:#ffb45b}.alarm-row.alarm-danger{border-left-color:#ff6b6b}.alarm-icon{display:grid;place-items:center;width:22px;height:22px;border-radius:50%;border:1px solid currentColor;color:#3ee5b1;font-weight:700}.alarm-warning .alarm-icon{color:#ffb45b}.alarm-danger .alarm-icon{color:#ff6b6b}.alarm-row strong,.alarm-row small{display:block}.alarm-row strong{color:#cfeaf2;font-size:11px}.alarm-row small{margin-top:4px;color:#7896a8;font-size:9px;line-height:1.45}
.liquid-pair{display:flex;justify-content:space-evenly;align-items:center;min-height:170px;padding:8px 0}.liquid-item{display:flex;flex-direction:column;align-items:center;gap:10px}.liquid-item>span{color:#bfe8f2;font-size:11px;font-weight:600}.liquid-ball{--liquid:#20e8aa;--liquid-soft:rgba(32,232,170,.42);position:relative;width:104px;height:104px;border:3px solid var(--liquid);border-radius:50%;overflow:hidden;background:rgba(3,19,32,.95);box-shadow:0 0 16px var(--liquid-soft),inset 0 0 18px rgba(0,0,0,.35)}.liquid-ball.cyan{--liquid:#22d8ff;--liquid-soft:rgba(34,216,255,.42)}.liquid-ball:before{content:"";position:absolute;inset:5px;border:1px solid var(--liquid);border-radius:50%;opacity:.65;z-index:3;pointer-events:none}.liquid-wave{position:absolute;left:-50%;width:200%;height:200%;top:var(--level);background:var(--liquid);border-radius:42%;opacity:.82;animation:liquid-spin 5.5s linear infinite;transition:top .8s cubic-bezier(.2,.8,.2,1)}.liquid-wave.wave-two{background:var(--liquid-soft);border-radius:46%;opacity:.7;animation-duration:8s;animation-direction:reverse;transform:translateY(-5px)}.liquid-ball strong{position:absolute;inset:0;z-index:5;display:grid;place-items:center;color:#efffff;font:700 19px ui-monospace,monospace;text-shadow:0 1px 6px rgba(0,0,0,.7)}.liquid-ball.missing{--liquid:#5c7d8c;--liquid-soft:rgba(92,125,140,.3)}@keyframes liquid-spin{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}@media(prefers-reduced-motion:reduce){.liquid-wave{animation:none}}@media (max-width:1400px){.screen-header{padding:0 18px}.screen-content{padding:14px}.brand-lockup{min-width:330px}.brand-lockup h1{font-size:17px}.main-grid{grid-template-columns:230px minmax(460px,1fr) 230px}.metric-value{font-size:21px}}
.assistant-dock{user-select:none}.assistant-float{cursor:grab;touch-action:none}.assistant-dock.dragging .assistant-float{cursor:grabbing;animation:none;transform:scale(1.03)}
.control-link{border:1px solid rgba(61,224,183,.48);background:rgba(43,184,153,.1);color:#72ebca;padding:8px 12px;font-size:11px;cursor:pointer;text-decoration:none;white-space:nowrap}.control-link:hover{background:rgba(43,184,153,.22);box-shadow:0 0 14px rgba(61,224,183,.12)}
</style>












