# Voice Lab

A multilingual browser playground for comparing original speech with tonal normalization and neural any-to-one voice conversion, built with Bun, TanStack Start, React, and the Web Audio API.

The app includes five 20–30 second samples generated with Together AI's `cartesia/sonic-3` model: English, Italian, Spanish, French, and Japanese. Runtime playback is static and does not call Together AI.

Voice modes:

- **Common Voice Lite** uses Web Audio EQ, dynamics normalization, saturation, and limiting with near-zero latency.
- **Common Voice Neural** runs a compact distilled LLVC any-to-one model through two-threaded ONNX Runtime Web WASM.
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

The full teacher graph was exported from KoeAI's MIT-licensed LLVC checkpoint; the production browser graph is a 128/64 student distilled for the bundled demo corpus. See [`public/models/README.md`](public/models/README.md), [`scripts/export-llvc.py`](scripts/export-llvc.py), and [`scripts/distill-llvc-student.py`](scripts/distill-llvc-student.py).
