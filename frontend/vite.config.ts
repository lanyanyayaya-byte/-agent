import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    host: '127.0.0.1',
    port: 1420,
    strictPort: true,
    fs: { allow: ['.'] },
    headers: {
      'Content-Security-Policy': [
        "default-src 'self'",
        "img-src 'self' blob: data:",
        "connect-src 'self' ws://127.0.0.1:1420 http://127.0.0.1:* ipc: http://ipc.localhost",
        "style-src 'self' 'unsafe-inline'",
        "object-src 'none'",
        "base-uri 'none'",
        "frame-ancestors 'none'",
      ].join('; '),
    },
  },
});
