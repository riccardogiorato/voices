import { readFileSync } from 'node:fs'
import * as ort from 'onnxruntime-web'

const modelPath = process.argv[2] ?? 'public/models/common-voice-llvc-q8-high.onnx'
const model = readFileSync(modelPath)
const session = await ort.InferenceSession.create(model, {
  executionProviders: ['wasm'],
})
const zeros = (shape) => new ort.Tensor('float32', new Float32Array(shape.reduce((total, size) => total * size, 1)), shape)
const output = await session.run({
  audio: zeros([1, 1, 240]),
  enc_state: zeros([1, 512, 510]),
  dec_state: zeros([1, 2, 13, 256]),
  out_state: zeros([1, 512, 4]),
  conv_state: zeros([1, 1, 24]),
})

if (output.converted.dims.join(',') !== '1,1,208') throw new Error(`Unexpected output shape: ${output.converted.dims}`)
if (![...output.converted.data].every(Number.isFinite)) throw new Error('Neural output contains non-finite samples')
console.log(`${modelPath} WASM smoke test passed: 240 input samples → 208 output samples`)
