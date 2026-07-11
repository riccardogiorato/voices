# Voice Lab

A realtime browser playground for robotic voice conversion experiments, built with Bun, TanStack Start, React, and the Web Audio API.

Audio stays inside the browser. Use a live microphone or load a local audio file, then adjust distortion, filter cutoff, playback pitch, and intensity.

Voice modes:

- **Common Voice Lite** uses Web Audio EQ, dynamics normalization, saturation, and limiting with near-zero latency.
- **Common Voice Neural** runs a browser-exported LLVC any-to-one model through ONNX Runtime Web. It prefers WebGPU and falls back to WASM.
- **Effects** contains the robotic presets and manual controls.

The neural model is downloaded as a static application asset and runs entirely inside the browser. Microphone audio is never sent to a server.

```bash
bun install
bun run dev
```

Production checks:

```bash
bun run typecheck
bun run build
bun run test:neural
```

## Neural model provenance

The bundled ONNX graph was exported from KoeAI's MIT-licensed LLVC checkpoint. See [`public/models/README.md`](public/models/README.md) and [`scripts/export-llvc.py`](scripts/export-llvc.py).
