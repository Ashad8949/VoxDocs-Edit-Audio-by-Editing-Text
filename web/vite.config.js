import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// The API server owns /api in every environment, so the dev server proxies it
// rather than the client needing to know a second origin.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: process.env.VOXDOCS_API_URL ?? 'http://localhost:3000',
        changeOrigin: true,
      },
    },
  },
  build: { outDir: 'dist', sourcemap: true },
});
