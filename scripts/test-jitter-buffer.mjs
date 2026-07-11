import { StreamingAudioBuffer } from '../public/audio-ring-buffer.js'

const buffer = new StreamingAudioBuffer(32768, 3200)
const renderBlock = new Float32Array(128)
const chunk = new Float32Array(208).fill(0.25)
const arrivals = []
let inferenceAvailableAt = 0

for (let i = 0; i < 600; i++) {
  const capturedAt = i * 13
  const processingTime = i > 0 && i % 100 === 0 ? 130 : i > 0 && i % 20 === 0 ? 42 : 8 + (i % 3)
  inferenceAvailableAt = Math.max(capturedAt, inferenceAvailableAt) + processingTime
  arrivals.push(inferenceAvailableAt)
}

let nextArrival = 0
for (let now = 0; now < 7500; now += 128 / 48) {
  while (nextArrival < arrivals.length && arrivals[nextArrival] <= now) {
    buffer.push(chunk)
    nextArrival++
  }
  buffer.render(renderBlock, 16000 / 48000)
}

if (buffer.underruns !== 0) throw new Error(`Jitter buffer underruns: ${buffer.underruns}`)
console.log('Neural jitter-buffer test passed: zero underruns with 130 ms inference spikes')
