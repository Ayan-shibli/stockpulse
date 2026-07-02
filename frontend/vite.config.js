import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // Forward /api/* requests to the FastAPI backend
      '/api': {
        target: 'http://localhost:8001',
        changeOrigin: true,
      },
      // Forward WebSocket connections
      '/ws': {
        target: 'ws://localhost:8001',
        ws: true,
      },
    },
  },
})
