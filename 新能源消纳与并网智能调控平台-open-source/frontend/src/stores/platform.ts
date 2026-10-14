import { computed, ref } from 'vue'
import { defineStore } from 'pinia'
import { platformApi } from '../api/platform'
import type { Overview, Station } from '../api/types'

const emptyOverview: Overview = { updated_at: '', stations: [], count: 0 }

export const usePlatformStore = defineStore('platform', () => {
  const stations = ref<Station[]>([])
  const overview = ref<Overview>({ ...emptyOverview })
  const selectedStationId = ref<string | null>(null)
  const connected = ref(false)
  let source: EventSource | null = null
  let reconnectTimer: number | undefined
  let retry = 0

  const isRealtime = computed(() => overview.value.stations.length > 0 && overview.value.stations.every((item) => item.freshness === 'realtime'))

  function applyOverview(payload: Overview) {
    overview.value = payload
  }

  async function loadOverview() {
    // Station metadata is a durable fallback; a realtime/SSE outage must not blank the dashboard.
    try { stations.value = await platformApi.stations() } catch { /* keep bundled station fallback */ }
    try { applyOverview(await platformApi.overview()) } catch { /* keep last good snapshot */ }
  }

  function selectStation(stationId: string | null) {
    selectedStationId.value = stationId || null
  }

  function connectRealtime() {
    if (source || typeof EventSource === 'undefined') return
    source = new EventSource('/api/stream/realtime')
    source.addEventListener('snapshot', (event) => {
      applyOverview(JSON.parse((event as MessageEvent).data) as Overview)
      connected.value = true
      retry = 0
    })
    source.onerror = () => {
      connected.value = false
      source?.close()
      source = null
      const delay = Math.min(30000, 1000 * 2 ** retry++)
      reconnectTimer = window.setTimeout(connectRealtime, delay)
    }
  }

  function disconnectRealtime() {
    if (reconnectTimer) window.clearTimeout(reconnectTimer)
    source?.close()
    source = null
    connected.value = false
  }

  return { stations, overview, selectedStationId, connected, isRealtime, applyOverview, loadOverview, selectStation, connectRealtime, disconnectRealtime }
})
