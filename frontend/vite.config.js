import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const backendTarget = env.VITE_API_URL || 'http://localhost:8001'
  return {
    plugins: [react()],
    server: {
      proxy: {
        '/api': {
          target: backendTarget,
          changeOrigin: true,
        },
        // WebSocket proxy — MUST be here or Vite's own HMR ws server
        // swallows /ws/research connections, causing a permanent hang.
        '/ws': {
          target: backendTarget.replace(/^http/, 'ws'),
          changeOrigin: true,
          ws: true,
        },
      },
    },
  }
})
