import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { tanstackStart } from '@tanstack/react-start/plugin/vite'
import tailwindcss from '@tailwindcss/vite'

// Cross-origin isolation enables SharedArrayBuffer, which onnxruntime-web needs
// to run multi-threaded WASM — the proven-fastest backend for the LLVC model
// (≈5.6 ms avg / 8.6 ms max / zero spikes per 13 ms chunk, vs WebGPU's 17.3 ms
// avg / 198 ms cold-start spike). `credentialless` (not `require-corp`) keeps the
// cross-origin Google Fonts @import in styles.css loading.
//
// Two mechanisms are needed: `server.headers` covers static assets (worklet,
// model, wasm), but the SSR'd HTML document is served by the TanStack Start
// server and bypasses `server.headers`. The middleware plugin below sets the
// headers on EVERY response (including the HTML), which is what makes
// `self.crossOriginIsolated` true in the document and the neural worker.
const isolationHeaders = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'credentialless',
}

const crossOriginIsolation = () => ({
  name: 'cross-origin-isolation',
  configureServer(server: { middlewares: { use: (m: unknown) => void } }) {
    server.middlewares.use((_req: unknown, res: { setHeader: (k: string, v: string) => void }, next: () => void) => {
      for (const [k, v] of Object.entries(isolationHeaders)) res.setHeader(k, v)
      next()
    })
  },
  configurePreviewServer(server: { middlewares: { use: (m: unknown) => void } }) {
    server.middlewares.use((_req: unknown, res: { setHeader: (k: string, v: string) => void }, next: () => void) => {
      for (const [k, v] of Object.entries(isolationHeaders)) res.setHeader(k, v)
      next()
    })
  },
})

export default defineConfig({
  plugins: [crossOriginIsolation(), tanstackStart(), tailwindcss(), react()],
  server: { headers: isolationHeaders },
  preview: { headers: isolationHeaders },
})