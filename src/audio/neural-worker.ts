// Real-time LLVC voice-conversion worker — fast (non-asyncify) threaded WASM.
//
// The neural pipeline processes 208-sample chunks (13 ms at 16 kHz), so every
// inference must finish in <13 ms or the output jitter buffer drains and the
// AudioWorklet glitches.
//
// The 128/64 distilled student runs at roughly 1.9 ms per chunk in Chrome with
// two WASM threads. The full 512/256 teacher takes roughly 9.3 ms, while WebGPU
// with GPU-resident recurrent state still takes roughly 13.2 ms because this
// small fragmented graph is dominated by dispatch overhead.
//
// 8 warmup runs prime the kernels before 'ready' (no cold-start spike). Falls
// back to single-threaded WASM if cross-origin isolation isn't available.

import * as ort from 'onnxruntime-web/wasm'
import type { InferenceSession, Tensor } from 'onnxruntime-web'

type Runtime = 'wasm'
type ModelVariant = 'current' | 'student128'
type ReadyMessage = { type: 'ready'; runtime: Runtime; threads: number; isolated: boolean }
type WorkerMessage =
  | { type: 'init'; threads?: number; model?: ModelVariant }
  | { type: 'audio'; samples: Float32Array }
  | { type: 'benchmark'; iterations?: number }
  | { type: 'reset' }

let stateShapes: Record<string, number[]> = {
  enc_state: [1, 512, 510],
  dec_state: [1, 2, 13, 256],
  out_state: [1, 512, 4],
  conv_state: [1, 1, 24],
}
const WARMUP_ITERS = 8

let session: InferenceSession | null = null
let runtime: Runtime = 'wasm'
let threads = 1
let states: Record<string, Tensor> = {}
let context = new Float32Array(32)
let audioQueue: Float32Array[] = []
let processing = false
let inferenceCount = 0
let inferenceTotalMs = 0
let inferenceMaxMs = 0
const inferenceTimes: number[] = []
let modelVariant: ModelVariant = 'current'

const workerScope = self as unknown as {
  crossOriginIsolated?: boolean
  postMessage: (message: unknown, transfer?: Transferable[]) => void
  onmessage: ((event: MessageEvent<WorkerMessage>) => void) | null
}

function zeros(shape: number[]) {
  return new ort.Tensor('float32', new Float32Array(shape.reduce((total, size) => total * size, 1)), shape)
}

function nextStates(out: Record<string, Tensor>) {
  return {
    enc_state: out.enc_state_next,
    dec_state: out.dec_state_next,
    out_state: out.out_state_next,
    conv_state: out.conv_state_next,
  }
}

// WASM has no GPU buffers to leak; dispose is a cheap no-op but kept for symmetry.
function disposeStatesExcept(prev: Record<string, Tensor>, keep: Record<string, Tensor>) {
  const keepSet = new Set(Object.values(keep))
  for (const tensor of Object.values(prev)) {
    if (keepSet.has(tensor)) continue
    try {
      tensor.dispose?.()
    } catch {
      // ignore
    }
  }
}

function reset() {
  states = Object.fromEntries(Object.entries(stateShapes).map(([name, shape]) => [name, zeros(shape)]))
  context = new Float32Array(32)
  audioQueue = []
  inferenceCount = 0
  inferenceTotalMs = 0
  inferenceMaxMs = 0
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
    const out = (await session!.run({ audio: dummy, ...prev })) as Record<string, Tensor>
    states = nextStates(out)
    disposeStatesExcept(prev, states)
  }
}

async function createSession(requestedThreads?: number, requestedModel: ModelVariant = 'student128') {
  modelVariant = requestedModel
  const isolated = workerScope.crossOriginIsolated === true
  threads = isolated ? Math.min(requestedThreads ?? (modelVariant === 'student128' ? 2 : 4), navigator.hardwareConcurrency || 4) : 1
  ort.env.wasm.numThreads = threads
  ort.env.wasm.simd = true
  ort.env.wasm.proxy = false
  console.warn('[neural] init: isolated=' + isolated + ' threads=' + threads)

  const configs: Record<ModelVariant, { model: string; external?: string }> = {
    current: { model: '/models/common-voice-llvc.onnx', external: '/models/common-voice-llvc.onnx.data' },
    student128: { model: '/models/common-voice-llvc-student128.onnx', external: '/models/common-voice-llvc-student128.onnx.data' },
  }
  const dimensions = modelVariant === 'student128' ? [128, 64] : [512, 256]
  stateShapes = {
    enc_state: [1, dimensions[0], 510],
    dec_state: [1, 2, 13, dimensions[1]],
    out_state: [1, dimensions[0], 4],
    conv_state: [1, 1, 24],
  }
  const config = configs[modelVariant]
  const [model, external] = await Promise.all([
    fetch(config.model).then((response) => response.arrayBuffer()),
    config.external ? fetch(config.external).then((response) => response.arrayBuffer()) : Promise.resolve(null),
  ])
  const opts: InferenceSession.SessionOptions = {
    graphOptimizationLevel: 'all',
    executionProviders: ['wasm'],
  }
  if (external && config.external) opts.externalData = [{ path: config.external.split('/').pop()!, data: new Uint8Array(external) }]
  try {
    session = await ort.InferenceSession.create(new Uint8Array(model), opts)
  } catch {
    ort.env.wasm.numThreads = 1
    threads = 1
    session = await ort.InferenceSession.create(new Uint8Array(model), opts)
    console.warn('[neural] fell back to single-threaded wasm')
  }
  reset()
  await warmup()
  reset()
  console.warn('[neural] ready: wasm ' + threads + 't')
  workerScope.postMessage({ type: 'ready', runtime, threads, isolated } satisfies ReadyMessage)
}

async function convert(samples: Float32Array, emitOutput = true) {
  if (!session) return
  const startedAt = performance.now()

  const audio = new Float32Array(240)
  audio.set(context)
  audio.set(samples, 32)
  context.set(samples.subarray(samples.length - 32))

  const prev = states
  const results = (await session.run({ audio: new ort.Tensor('float32', audio, [1, 1, 240]), ...prev })) as Record<string, Tensor>
  states = nextStates(results)
  disposeStatesExcept(prev, states)

  const output = new Float32Array(results.converted.data as Float32Array)

  const elapsedMs = performance.now() - startedAt
  if (emitOutput) workerScope.postMessage({ type: 'output', samples: output }, [output.buffer])
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
  return { elapsedMs, output }
}

async function benchmark(iterations = 250) {
  if (!session || processing) return
  processing = true
  reset()
  let finite = true
  let peak = 0
  try {
    for (let i = 0; i < iterations; i++) {
      const samples = new Float32Array(208)
      for (let j = 0; j < samples.length; j++) samples[j] = 0.2 * Math.sin((i * samples.length + j) * 0.057)
      const result = await convert(samples, false)
      if (!result) continue
      for (const value of result.output) {
        finite &&= Number.isFinite(value)
        peak = Math.max(peak, Math.abs(value))
      }
    }
    const sorted = [...inferenceTimes].sort((a, b) => a - b)
    workerScope.postMessage({
      type: 'benchmark-result',
      iterations,
      averageMs: inferenceTotalMs / inferenceCount,
      p50Ms: sorted[Math.floor(sorted.length * 0.5)],
      p95Ms: sorted[Math.floor(sorted.length * 0.95)],
      p99Ms: sorted[Math.floor(sorted.length * 0.99)],
      maxMs: inferenceMaxMs,
      finite,
      peak,
      threads,
      isolated: workerScope.crossOriginIsolated === true,
      model: modelVariant,
      runtime,
    })
  } finally {
    processing = false
    reset()
  }
}

workerScope.onmessage = ({ data }: MessageEvent<WorkerMessage>) => {
  if (data.type === 'init') void createSession(data.threads, data.model).catch((error) => workerScope.postMessage({ type: 'error', message: error instanceof Error ? error.message : 'Model initialization failed' }))
  if (data.type === 'reset') reset()
  if (data.type === 'benchmark') void benchmark(data.iterations).catch((error) => workerScope.postMessage({ type: 'error', message: error instanceof Error ? error.message : 'Benchmark failed' }))
  if (data.type === 'audio') {
    audioQueue.push(data.samples)
    if (audioQueue.length > 24) audioQueue.shift()
    void processQueue().catch((error) => workerScope.postMessage({ type: 'error', message: error instanceof Error ? error.message : 'Inference failed' }))
  }
}
