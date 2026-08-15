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
        // 127.0.0.1, not localhost: some machines resolve "localhost" to the
        // IPv6 loopback (::1) first, and Django's dev server only binds the
        // IPv4 address, so the proxy would connect-fail on every request.
        target: process.env.VOXDOCS_API_URL ?? 'http://127.0.0.1:3000',
        changeOrigin: true,
      },
    },
  },
  build: { outDir: 'dist', sourcemap: true },
});
