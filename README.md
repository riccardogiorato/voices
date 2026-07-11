# Voice Lab

A multilingual browser playground for comparing original speech with tonal normalization and neural any-to-one voice conversion, built with Bun, TanStack Start, React, and the Web Audio API.

The app includes five 20–30 second samples generated with Together AI's `cartesia/sonic-3` model: English, Italian, Spanish, French, and Japanese. Runtime playback is static and does not call Together AI.

Voice modes:

- **Common Voice Lite** uses Web Audio EQ, dynamics normalization, saturation, and limiting with near-zero latency.
- **Common Voice Neural** runs a browser-exported LLVC any-to-one model through ONNX Runtime Web. It prefers WebGPU and falls back to WASM.
- **Original** disables processing and plays the source sample unchanged.

The neural model preloads when the page opens and runs entirely inside the browser. Selecting a mode applies it automatically; there is no engage button.

```bash
bun install
bun run dev
```

To regenerate the sample assets, place `TOGETHER_API_KEY` in `.env` and run:

```bash
bun run generate:samples
```

Production checks:

```bash
bun run typecheck
bun run build
bun run test:neural
```

## Neural model provenance

The bundled ONNX graph was exported from KoeAI's MIT-licensed LLVC checkpoint. See [`public/models/README.md`](public/models/README.md) and [`scripts/export-llvc.py`](scripts/export-llvc.py).
