<template><div class="admin-shell"><header><div class="admin-brand">⚡ 新能源调控平台 <span>/ 管理中心</span></div><nav><button class="logout" @click="logout">退出登录</button><RouterLink to="/dashboard">返回大屏 ↗</RouterLink></nav></header><main><slot /></main></div></template>
<script setup lang="ts">
import { RouterLink } from 'vue-router'
import { useRouter } from 'vue-router'
const router = useRouter()
function logout(){ localStorage.removeItem('admin_token'); router.push('/admin/login') }
</script>
<style scoped>.admin-shell{min-height:100vh;background:#071522;color:#d9f3fb}.admin-shell header{height:66px;padding:0 28px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid rgba(80,186,223,.25);background:#081d31}.admin-brand{font-size:16px;letter-spacing:.08em}.admin-brand span{color:#6f96aa}.admin-shell nav{display:flex;align-items:center;gap:18px}.admin-shell a,.logout{color:#62d7f5;text-decoration:none;font-size:12px}.logout{border:0;background:transparent;cursor:pointer}.admin-shell main{max-width:1440px;margin:0 auto;padding:26px}</style>
