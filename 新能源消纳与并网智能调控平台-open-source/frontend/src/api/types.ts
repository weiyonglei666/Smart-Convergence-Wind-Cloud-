export type Freshness = 'realtime' | 'near_realtime' | 'delayed' | 'offline' | 'invalid' | 'unavailable'

export interface Station {
  station_id: string
  station_type: 'wind' | 'pv'
  name: string
  longitude: number
  latitude: number
  capacity_mw: number | null
  enabled: boolean
}

export interface Snapshot {
  station_id: string
  observed_at: string | null
  active_power_mw: number | null
  theory_power_mw: number | null
  available_power_mw: number | null
  limit_mark: number | null
  weather: Record<string, number | null>
  freshness: Freshness
  source: string
  quality: 'complete' | 'partial' | 'insufficient'
}

export interface PowerPoint {
  station_id: string
  observed_at: string
  active_power_mw: number | null
  theory_power_mw: number | null
  available_power_mw: number | null
}

export interface Overview {
  updated_at: string
  stations: Snapshot[]
  count: number
}

export interface ConsumptionOverview {
  updated_at: string
  actual_power_mw: number
  theory_power_mw: number
  available_power_mw: number
  consumption_rate: number | null
  curtailment_rate: number | null
  station_count: number
  quality: 'complete' | 'partial' | 'anomaly'
  warning: string | null
  comparable_station_count?: number
  missing_station_ids?: string[]
  anomaly_station_ids?: string[]
  station_breakdown?: Array<{ station_id: string; active_power_mw: number | null; theory_power_mw: number | null; available_power_mw: number | null; consumption_rate: number | null; curtailment_rate: number | null; status: 'complete' | 'missing_fields' | 'anomaly' }>
}

export interface WeatherObservationOverview { updated_at: string; stations: Array<{ station_id: string; station_type: 'wind' | 'pv'; weather: Record<string, number | string | null>; source: string }> }
export interface WeatherPoint { observed_at: string; wind_speed_ms: number | null; wind_direction_deg: number | null; temperature_c: number | null; ghi_wm2: number | null; dni_wm2: number | null; dhi_wm2: number | null }
export interface ForecastMetrics { station_id: string; sample_count: number; mae: number | null; rmse: number | null; acc?: number | null; run_count?: number }
export interface ForecastPoint { station_id: string; observed_at: string; predicted_power_mw: number | null; actual_power_mw: number | null }
export interface ForecastWindow { station_id: string; issue_at: string | null; horizon_points: number; horizon_hours: number; points: Array<ForecastPoint & { phase: 'history' | 'forecast' }> }
export interface NwpLatest { available: boolean; issued_at_utc?: string; issued_at_cn?: string; age_hours?: number; freshness?: 'latest' | 'stale'; refresh_required?: boolean; source_file?: string; stations: Array<{ station_id: string; latitude: number; longitude: number; variables: Record<string, number>; forecast?: Array<{ valid_time_utc: string; valid_time_cn?: string; variables: Record<string, number> }> }> }
export interface AssistantStep { agent: string; status: string; detail: string }
export interface AssistantResponse { answer: string; sources: Array<{ source_id: string; score: number; title?: string; excerpt?: string }>; steps: AssistantStep[]; live_context: Record<string, any>; mode: string; confidence: string }
