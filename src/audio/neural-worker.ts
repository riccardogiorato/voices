// Real-time LLVC voice-conversion worker.
//
// The neural pipeline processes 208-sample chunks (13 ms at 16 kHz), so every
// inference must finish in <13 ms or the output jitter buffer drains and the
// AudioWorklet glitches. Backend strategy (see SLOW.md / PROGRESS.md):
//   1. Multi-threaded WASM when the page is cross-origin isolated (needs the
//      COOP/COEP headers set in vite.config.ts). Benchmarked at ≈5.6 ms avg /
//      8.6 ms max with zero spikes — comfortably inside the 13 ms budget.
//   2. Tuned WebGPU fallback: recurrent state stays on the GPU between calls
//      (preferredOutputLocation: 'gpu-buffer') so the ≈1 MB enc_state is not
//      uploaded/downloaded every chunk. For devices where threaded WASM is
//      unavailable.
//   3. Single-threaded WASM last resort.
// Every backend runs warmup inferences before signalling "ready", so the first
// real chunk never stalls on kernel/shader compilation (the 198 ms cold spike).

import * as ort from 'onnxruntime-web/webgpu'

type Runtime = 'webgpu' | 'wasm'
type ReadyMessage = { type: 'ready'; runtime: Runtime; threads: number; isolated: boolean }
type WorkerMessage = { type: 'init' } | { type: 'audio'; samples: Float32Array } | { type: 'reset' }

const STATE_SHAPES: Record<string, number[]> = {
  enc_state: [1, 512, 510],
  dec_state: [1, 2, 13, 256],
  out_state: [1, 512, 4],
  conv_state: [1, 1, 24],
}
const WARMUP_ITERS = 6

let session: ort.InferenceSession | null = null
let runtime: Runtime = 'wasm'
let threads = 1
let states: Record<string, ort.Tensor> = {}
let context = new Float32Array(32)
let audioQueue: Float32Array[] = []
let processing = false
let inferenceCount = 0
let inferenceTotalMs = 0
let inferenceMaxMs = 0
const inferenceTimes: number[] = []

const workerScope = self as unknown as {
  crossOriginIsolated?: boolean
  postMessage: (message: unknown, transfer?: Transferable[]) => void
  onmessage: ((event: MessageEvent<WorkerMessage>) => void) | null
}

function zeros(shape: number[]) {
  return new ort.Tensor('float32', new Float32Array(shape.reduce((total, size) => total * size, 1)), shape)
}

function nextStates(out: Record<string, ort.Tensor>) {
  return {
    enc_state: out.enc_state_next,
    dec_state: out.dec_state_next,
    out_state: out.out_state_next,
    conv_state: out.conv_state_next,
  }
}

// Free the consumed input state tensors, but never the ones aliased into the
// new outputs (ORT may update a state in place). Matters on WebGPU, where
// skipping this leaks a GPU buffer every chunk and eventually OOMs.
function disposeStatesExcept(prev: Record<string, ort.Tensor>, keep: Record<string, ort.Tensor>) {
  const keepSet = new Set(Object.values(keep))
  for (const tensor of Object.values(prev)) if (!keepSet.has(tensor)) tensor.dispose?.()
}

function reset() {
  states = Object.fromEntries(Object.entries(STATE_SHAPES).map(([name, shape]) => [name, zeros(shape)]))
  context = new Float32Array(32)
  audioQueue = []
  inferenceTimes.length = 0
}

async function processQueue() {
  if (processing) return
  processing = true
  while (audioQueue.length) await convert(audioQueue.shift()!)
  processing = false
}

async function warmup() {
  const dummy = new ort.Tensor('float32', new Float32Array(240), [1, 1, 240])
  for (let i = 0; i < WARMUP_ITERS; i++) {
    const prev = states
    const out = (await session!.run({ audio: dummy, ...prev })) as Record<string, ort.Tensor>
    states = nextStates(out)
    disposeStatesExcept(prev, states)
  }
}

async function createSession() {
  const isolated = workerScope.crossOriginIsolated === true
  threads = isolated ? Math.min(8, navigator.hardwareConcurrency || 4) : 1
  ort.env.wasm.numThreads = threads
  ort.env.wasm.simd = true
  ort.env.wasm.proxy = false

  const [model, external] = await Promise.all([
    fetch('/models/common-voice-llvc.onnx').then((response) => response.arrayBuffer()),
    fetch('/models/common-voice-llvc.onnx.data').then((response) => response.arrayBuffer()),
  ])
  const base = {
    externalData: [{ path: 'common-voice-llvc.onnx.data', data: new Uint8Array(external) }],
    graphOptimizationLevel: 'all' as const,
  }

  // 1) Multi-threaded WASM (single-threaded when not cross-origin isolated).
  try {
    session = await ort.InferenceSession.create(model, { ...base, executionProviders: ['wasm'] })
    runtime = 'wasm'
  } catch {
    // 2) Tuned WebGPU — keep recurrent state on the GPU across calls.
    try {
      session = await ort.InferenceSession.create(model, {
        ...base,
        executionProviders: ['webgpu'],
        preferredOutputLocation: {
          enc_state_next: 'gpu-buffer',
          dec_state_next: 'gpu-buffer',
          out_state_next: 'gpu-buffer',
          conv_state_next: 'gpu-buffer',
          converted: 'cpu',
        },
      })
      runtime = 'webgpu'
    } catch {
      // 3) Last-resort single-threaded WASM.
      ort.env.wasm.numThreads = 1
      threads = 1
      session = await ort.InferenceSession.create(model, { ...base, executionProviders: ['wasm'] })
      runtime = 'wasm'
    }
  }

  reset()
  await warmup()
  reset()
  workerScope.postMessage({ type: 'ready', runtime, threads, isolated } satisfies ReadyMessage)
}

async function convert(samples: Float32Array) {
  if (!session) return
  const startedAt = performance.now()

  const audio = new Float32Array(240)
  audio.set(context)
  audio.set(samples, 32)
  context.set(samples.subarray(samples.length - 32))

  const prev = states
  const results = (await session.run({ audio: new ort.Tensor('float32', audio, [1, 1, 240]), ...prev })) as Record<string, ort.Tensor>
  states = nextStates(results)
  disposeStatesExcept(prev, states)

  const output = new Float32Array(results.converted.data as Float32Array)
  workerScope.postMessage({ type: 'output', samples: output }, [output.buffer])

  const elapsedMs = performance.now() - startedAt
  inferenceCount++
  inferenceTotalMs += elapsedMs
  inferenceMaxMs = Math.max(inferenceMaxMs, elapsedMs)
  inferenceTimes.push(elapsedMs)
  if (inferenceTimes.length > 200) inferenceTimes.shift()
  if (inferenceCount % 25 === 0) {
    const sorted = [...inferenceTimes].sort((a, b) => a - b)
    const p99 = sorted[Math.floor(sorted.length * 0.99)]
    workerScope.postMessage({
      type: 'stats',
      averageMs: inferenceTotalMs / inferenceCount,
      maxMs: inferenceMaxMs,
      p99Ms: p99,
      queue: audioQueue.length,
    })
  }
}

workerScope.onmessage = ({ data }: MessageEvent<WorkerMessage>) => {
  if (data.type === 'init') void createSession().catch((error) => workerScope.postMessage({ type: 'error', message: error instanceof Error ? error.message : 'Model initialization failed' }))
  if (data.type === 'reset') reset()
  if (data.type === 'audio') {
    audioQueue.push(data.samples)
    if (audioQueue.length > 24) audioQueue.shift()
    void processQueue().catch((error) => workerScope.postMessage({ type: 'error', message: error instanceof Error ? error.message : 'Inference failed' }))
  }
}