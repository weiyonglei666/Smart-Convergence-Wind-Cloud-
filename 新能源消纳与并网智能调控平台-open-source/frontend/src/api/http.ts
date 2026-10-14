import axios from 'axios'

// MySQL aggregation can take a little longer on the first request; keep the UI
// connected long enough to receive the real snapshot instead of showing blanks.
export const http = axios.create({ baseURL: '/api', timeout: 60000 })
