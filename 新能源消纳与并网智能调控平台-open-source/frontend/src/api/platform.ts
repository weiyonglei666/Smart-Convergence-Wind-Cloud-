import { http } from './http'
import type { AssistantResponse, ConsumptionOverview, ForecastMetrics, ForecastPoint, ForecastWindow, NwpLatest, Overview, PowerPoint, Station, Snapshot, WeatherObservationOverview, WeatherPoint } from './types'

const forecastCache = new Map<string, { expires: number; value: Promise<ForecastPoint[]> }>()
const forecastWindowCache = new Map<string, { expires: number; value: Promise<ForecastWindow> }>()
const forecastMetricsCache = new Map<string, { expires: number; value: Promise<ForecastMetrics> }>()

export const platformApi = {
  stations: () => http.get<Station[]>('/stations').then((response) => response.data),
  overview: () => http.get<Overview>('/overview/realtime').then((response) => response.data),
  snapshot: (stationId: string) => http.get<Snapshot>(`/stations/${stationId}/snapshot`).then((response) => response.data),
  powerSeries: (stationId: string, hours = 24) => http.get<PowerPoint[]>(`/stations/${stationId}/power-series`, { params: { hours } }).then((response) => response.data),
  consumption: () => http.get<ConsumptionOverview>('/overview/consumption').then((response) => response.data),
  weatherObservation: () => http.get<WeatherObservationOverview>('/overview/weather-observation').then((response) => response.data),
  weatherSeries: (stationId: string, hours = 24) => http.get<WeatherPoint[]>(`/stations/${stationId}/weather-series`, { params: { hours } }).then((response) => response.data),
  forecastMetrics: (stationId: string, hours = 24) => {
    const key = `${stationId}:${hours}`; const cached = forecastMetricsCache.get(key); if (cached && cached.expires > Date.now()) return cached.value
    const value = http.get<ForecastMetrics>(`/stations/${stationId}/forecast-metrics`, { params: { hours } }).then((response) => response.data); forecastMetricsCache.set(key,{value,expires:Date.now()+4500}); return value
  },
  forecastWindow: (stationId: string, points = 16) => {
    const key = `${stationId}:${points}`; const cached = forecastWindowCache.get(key); if (cached && cached.expires > Date.now()) return cached.value
    const value = http.get<ForecastWindow>(`/stations/${stationId}/forecast-window`, { params: { points } }).then((response) => response.data); forecastWindowCache.set(key,{value,expires:Date.now()+4500}); return value
  },
  forecastSeries: (stationId: string, hours = 24) => {
    const key = `${stationId}:${hours}`
    const cached = forecastCache.get(key)
    if (cached && cached.expires > Date.now()) return cached.value
    const value = http.get<ForecastPoint[]>(`/stations/${stationId}/forecast-series`, { params: { hours } }).then((response) => response.data)
    forecastCache.set(key, { value, expires: Date.now() + 15000 })
    return value
  },
  nwpLatest: () => http.get<NwpLatest>('/overview/nwp-latest').then((response) => response.data),
  assistantStatus: () => http.get<{ ready: boolean; document_count: number; model_configured: boolean; agents: string[] }>('/assistant/status').then((response) => response.data),
  assistantChat: (question: string, stationId?: string | null, useWebSearch = false, history: Array<{ role: string; content: string }> = []) => http.post<AssistantResponse>('/assistant/chat', { question, station_id: stationId || null, use_web_search: useWebSearch, history }).then((response) => response.data),
  parseAttachment: (filename: string, mimeType: string, contentBase64: string) => http.post<{ filename: string; characters: number; text: string }>('/assistant/attachment', { filename, mime_type: mimeType, content_base64: contentBase64 }).then((response) => response.data),
}
