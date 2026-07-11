# Neural audio performance notes

## Symptom

Neural mode can sound laggy, glitchy, or clicky even though the source audio plays normally in Original and Lite modes.

The neural pipeline processes audio in 208-sample chunks. At the model's 16 kHz sample rate, each chunk represents about 13 ms of audio, so every inference must finish in less than 13 ms on average and must avoid long timing spikes. If inference falls behind, the output jitter buffer drains and the AudioWorklet has no converted samples to play. That buffer underrun is heard as a gap, click, or stutter.

## Measured baseline

The live Chrome run on WebGPU produced:

- Chunk budget: about 13 ms
- Average inference: 17.3 ms
- Maximum inference: 198 ms
- Worker queue: 30 pending chunks
- Output buffer: 143 ms
- Underruns: 7 in roughly 10 seconds

This proves that the current neural model does not sustain real-time throughput on this browser/device. The 200 ms anti-glitch buffer hides some timing variation, but it cannot compensate when average inference is slower than the incoming audio rate.

## How to reproduce and debug

1. Start the development server with `bun run dev`.
2. Open the app in Chrome.
3. Select **Neural** mode and play the English sample for at least 10 seconds.
4. Read the runtime diagnostics shown under the output status:
   - `avg`: average ONNX inference duration
   - `max`: worst inference duration
   - `queue`: neural chunks waiting to be processed
   - `buffer`: converted audio currently buffered for playback
   - `underruns`: number of times playback requested audio that was unavailable
5. Repeat the same sample after each optimization and record all five values.

The main pass/fail conditions are:

- Average inference should remain comfortably below the 13 ms chunk budget.
- The queue should remain near zero instead of growing over time.
- The output buffer should stabilize instead of trending toward zero.
- Underruns should remain at zero for the entire sample.

Do not judge an optimization only by average latency. A low average with 100–200 ms spikes can still empty the buffer and produce audible glitches. Track p95/p99 or at least the maximum duration as well.

## Useful experiments

Change one variable at a time and replay the same audio sample.

1. Compare WebGPU and WASM using the same model and chunk size.
2. Measure model warm-up separately from steady-state inference.
3. Profile tensor allocation, copying, and worker message transfer around each inference.
4. Test larger inference batches or chunks. This can reduce WebGPU dispatch overhead, but increases latency and may require overlap/crossfade at boundaries.
5. Check that recurrent model state is reused correctly and that no session or tensors are recreated for each chunk.
6. Test the resampling and ring-buffer path independently with a passthrough worker. If underruns remain at zero but clicks remain audible, inspect sample continuity and chunk alignment.
7. Record inference durations for the full 20–30 second sample, not only the first few chunks.

The deterministic performance tests are available through:

```sh
bun run test:neural
bun run test:jitter
```

These tests are useful for regression checks, but the final validation must run in Chrome because browser WebGPU scheduling and GPU stalls are not reproduced accurately by a synthetic timing test.