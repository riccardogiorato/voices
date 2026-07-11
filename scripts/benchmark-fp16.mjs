import { readFileSync } from 'node:fs'
import * as ort from 'onnxruntime-web'

const model = readFileSync('public/models/common-voice-llvc-fp16.onnx')

console.log('=== LLVC FP16 Benchmark (WASM) ===\n')

const t0 = performance.now()
const session = await ort.InferenceSession.create(model, {
  executionProviders: ['wasm'],
  graphOptimizationLevel: 'all',
  numThreads: 8,
})
const createMs = performance.now() - t0
console.log(`Session creation: ${createMs.toFixed(0)}ms`)

const zeros = (s) => new ort.Tensor('float32', new Float32Array(s.reduce((t, v) => t * v, 1)), s)
let states = {
  enc_state: zeros([1, 512, 510]),
  dec_state: zeros([1, 2, 13, 256]),
  out_state: zeros([1, 512, 4]),
  conv_state: zeros([1, 1, 24]),
}
let context = new Float32Array(32)

// Warmup
for (let i = 0; i < 10; i++) {
  const audio = new Float32Array(240)
  audio.set(context)
  audio.set(new Float32Array(208).fill(0.3), 32)
  context = audio.slice(208)
  try {
    const out = await session.run({ audio: new ort.Tensor('float32', audio, [1, 1, 240]), ...states })
    states = { enc_state: out.enc_state_next, dec_state: out.dec_state_next, out_state: out.out_state_next, conv_state: out.conv_state_next }
  } catch(e) {
    console.error('Warmup error:', e.message)
    process.exit(1)
  }
}

// Benchmark
const N = 300
const times = []
for (let i = 0; i < N; i++) {
  const audio = new Float32Array(240)
  audio.set(context)
  const s = new Float32Array(208)
  for (let j = 0; j < 208; j++) s[j] = 0.3 * Math.sin((i * 208 + j) * 0.05)
  audio.set(s, 32)
  context = audio.slice(208)
  const t = performance.now()
  const out = await session.run({ audio: new ort.Tensor('float32', audio, [1, 1, 240]), ...states })
  times.push(performance.now() - t)
  states = { enc_state: out.enc_state_next, dec_state: out.dec_state_next, out_state: out.out_state_next, conv_state: out.conv_state_next }
}

const sorted = [...times].sort((a, b) => a - b)
const avg = times.reduce((a, b) => a + b, 0) / times.length
const p50 = sorted[Math.floor(sorted.length * 0.5)]
const p95 = sorted[Math.floor(sorted.length * 0.95)]
const p99 = sorted[Math.floor(sorted.length * 0.99)]
const max = sorted[sorted.length - 1]
const min = sorted[0]
const spikes50 = times.filter(t => t >= 50).length
const spikes100 = times.filter(t => t >= 100).length

console.log(`\nResults (FP16, WASM 8 threads):`)
console.log(`  Min:     ${min.toFixed(2)}ms`)
console.log(`  Avg:     ${avg.toFixed(2)}ms  (RTF: ${(avg/13).toFixed(2)}x)`)
console.log(`  P50:     ${p50.toFixed(2)}ms`)
console.log(`  P95:     ${p95.toFixed(2)}ms`)
console.log(`  P99:     ${p99.toFixed(2)}ms`)
console.log(`  Max:     ${max.toFixed(2)}ms`)
console.log(`  Spikes>=50ms: ${spikes50}, >=100ms: ${spikes100}`)
console.log(`  Budget:  ${avg < 13 ? 'PASS' : 'FAIL'}`)

// Check output is valid
const out0 = await session.run({ audio: new ort.Tensor('float32', new Float32Array(240), [1, 1, 240]), ...states })
const converted = out0.converted.data
console.log(`\n  Output shape: ${out0.converted.dims.join(',')}`)
console.log(`  Output finite: ${converted.every(Number.isFinite) ? 'YES' : 'NO'}`)
console.log(`  Output range: [${Math.min(...converted).toFixed(4)}, ${Math.max(...converted).toFixed(4)}]`)
