import { createRouter, createWebHistory } from 'vue-router'
import DashboardView from '../views/dashboard/DashboardView.vue'
import StationDetailView from '../views/stations/StationDetailView.vue'
import AnalysisView from '../views/analysis/AnalysisView.vue'
import AdminHomeView from '../views/admin/AdminHomeView.vue'
import AdminLoginView from '../views/admin/AdminLoginView.vue'
import WeatherObservationView from '../views/weather/WeatherObservationView.vue'
import NwpLatestView from '../views/weather/NwpLatestView.vue'
import AssistantView from '../views/assistant/AssistantView.vue'

export const router = createRouter({
  history: createWebHistory(),
  routes: [{ path: '/', redirect: '/dashboard' }, { path: '/dashboard', component: DashboardView }, { path: '/weather', component: WeatherObservationView }, { path: '/stations/:id', component: StationDetailView }, { path: '/analysis', component: AnalysisView }, { path: '/assistant', component: AssistantView }, { path: '/admin/login', component: AdminLoginView }, { path: '/admin', component: AdminHomeView, meta: { requiresAdmin: true } }],
})
router.addRoute({ path: '/weather/nwp', component: NwpLatestView })
router.beforeEach((to) => { if (to.meta.requiresAdmin && !localStorage.getItem('admin_token')) return '/admin/login' })
