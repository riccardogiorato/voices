import { readFileSync } from 'node:fs'
import * as ort from 'onnxruntime-web'

const model = readFileSync('public/models/common-voice-llvc.onnx')
const external = readFileSync('public/models/common-voice-llvc.onnx.data')

console.log('=== LLVC ONNX Inference Benchmark (WASM backend) ===\n')

// Create session
const t0 = performance.now()
const session = await ort.InferenceSession.create(model, {
  executionProviders: ['wasm'],
  externalData: [{ path: 'common-voice-llvc.onnx.data', data: external }],
  graphOptimizationLevel: 'all',
})
const sessionCreateMs = performance.now() - t0
console.log(`Session creation: ${sessionCreateMs.toFixed(1)}ms`)

// Helper
const zeros = (shape) => new ort.Tensor('float32', new Float32Array(shape.reduce((t, s) => t * s, 1)), shape)

// Init states
let states = {
  enc_state: zeros([1, 512, 510]),
  dec_state: zeros([1, 2, 13, 256]),
  out_state: zeros([1, 512, 4]),
  conv_state: zeros([1, 1, 24]),
}
let context = new Float32Array(32)

// Warmup (first few inferences include shader/kernel compilation)
console.log('\n--- Warmup (10 iterations) ---')
const warmupTimes = []
for (let i = 0; i < 10; i++) {
  const t = performance.now()
  const audio = new Float32Array(240)
  audio.set(context)
  audio.set(new Float32Array(208).fill(0.3 * Math.sin(i * 0.1)), 32)
  context = audio.slice(208)
  const out = await session.run({ audio: new ort.Tensor('float32', audio, [1, 1, 240]), ...states })
  states = {
    enc_state: out.enc_state_next,
    dec_state: out.dec_state_next,
    out_state: out.out_state_next,
    conv_state: out.conv_state_next,
  }
  warmupTimes.push(performance.now() - t)
}
console.log(`Warmup avg: ${(warmupTimes.reduce((a,b) => a+b, 0) / warmupTimes.length).toFixed(2)}ms`)
console.log(`Warmup range: ${Math.min(...warmupTimes).toFixed(2)}-${Math.max(...warmupTimes).toFixed(2)}ms`)

// Steady-state benchmark
const N = 500
console.log(`\n--- Steady-state (${N} iterations) ---`)
const times = []
const tensorAllocTimes = []
const stateUpdateTimes = []

for (let i = 0; i < N; i++) {
  // Measure tensor allocation
  const ta = performance.now()
  const audio = new Float32Array(240)
  audio.set(context)
  const samples = new Float32Array(208)
  for (let j = 0; j < 208; j++) samples[j] = 0.3 * Math.sin((i * 208 + j) * 0.05)
  audio.set(samples, 32)
  context = audio.slice(208)
  const audioTensor = new ort.Tensor('float32', audio, [1, 1, 240])
  const inputs = { audio: audioTensor, ...states }
  const taEnd = performance.now()
  tensorAllocTimes.push(taEnd - ta)

  // Measure inference
  const t = performance.now()
  const out = await session.run(inputs)
  const inferMs = performance.now() - t
  times.push(inferMs)

  // Measure state update
  const su = performance.now()
  states = {
    enc_state: out.enc_state_next,
    dec_state: out.dec_state_next,
    out_state: out.out_state_next,
    conv_state: out.conv_state_next,
  }
  stateUpdateTimes.push(performance.now() - su)
}

// Sort for percentiles
const sorted = [...times].sort((a, b) => a - b)
const avg = times.reduce((a, b) => a + b, 0) / times.length
const p50 = sorted[Math.floor(sorted.length * 0.5)]
const p95 = sorted[Math.floor(sorted.length * 0.95)]
const p99 = sorted[Math.floor(sorted.length * 0.99)]
const max = sorted[sorted.length - 1]
const min = sorted[0]
const rtf = avg / 13 // 13ms is the real-time budget

const allocAvg = tensorAllocTimes.reduce((a, b) => a + b, 0) / tensorAllocTimes.length
const stateUpdateAvg = stateUpdateTimes.reduce((a, b) => a + b, 0) / stateUpdateTimes.length

console.log(`\nResults:`)
console.log(`  Min:       ${min.toFixed(2)}ms`)
console.log(`  Average:   ${avg.toFixed(2)}ms  (RTF: ${rtf.toFixed(2)}x, budget: 13ms)`)
console.log(`  P50:       ${p50.toFixed(2)}ms`)
console.log(`  P95:       ${p95.toFixed(2)}ms`)
console.log(`  P99:       ${p99.toFixed(2)}ms`)
console.log(`  Max:       ${max.toFixed(2)}ms`)
console.log(`\n  Tensor alloc avg:  ${allocAvg.toFixed(3)}ms`)
console.log(`  State update avg:  ${stateUpdateAvg.toFixed(3)}ms`)
console.log(`  Inference-only avg: ${(avg - allocAvg - stateUpdateAvg).toFixed(2)}ms`)
console.log(`\n  Budget met (<13ms avg): ${avg < 13 ? 'YES ✓' : 'NO ✗'}`)
const spikes50Count = times.filter(t => t >= 50).length
console.log(`  Spikes <50ms:           ${spikes50Count === 0 ? 'YES' : 'NO (' + spikes50Count + ' spikes)'}`)

// Count spikes
const spikes50 = times.filter(t => t >= 50).length
const spikes100 = times.filter(t => t >= 100).length
const spikes200 = times.filter(t => t >= 200).length
console.log(`  Spikes >=50ms:  ${spikes50}`)
console.log(`  Spikes >=100ms: ${spikes100}`)
console.log(`  Spikes >=200ms: ${spikes200}`)

// Output tensor info
const out0 = await session.run({ audio: new ort.Tensor('float32', new Float32Array(240), [1, 1, 240]), ...states })
console.log(`\n--- Model Info ---`)
for (const [name, tensor] of Object.entries(out0)) {
  console.log(`  Output "${name}": shape=[${tensor.dims.join(',')}], type=${tensor.type}, size=${tensor.data.length} floats`)
}
for (const input of session.inputNames) {
  console.log(`  Input "${input}"`)
}
console.log(`  Input names: ${session.inputNames.join(', ')}`)
console.log(`  Output names: ${session.outputNames.join(', ')}`)
