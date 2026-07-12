import { readFileSync } from 'node:fs'
import * as ort from 'onnxruntime-web'

const model = readFileSync('public/models/common-voice-llvc-student128.onnx')
const external = readFileSync('public/models/common-voice-llvc-student128.onnx.data')
const session = await ort.InferenceSession.create(model, {
  executionProviders: ['wasm'],
  externalData: [{ path: 'common-voice-llvc-student128.onnx.data', data: external }],
})
const zeros = (shape) => new ort.Tensor('float32', new Float32Array(shape.reduce((total, size) => total * size, 1)), shape)
const output = await session.run({
  audio: zeros([1, 1, 240]),
  enc_state: zeros([1, 128, 510]),
  dec_state: zeros([1, 2, 13, 64]),
  out_state: zeros([1, 128, 4]),
  conv_state: zeros([1, 1, 24]),
})

if (output.converted.dims.join(',') !== '1,1,208') throw new Error(`Unexpected output shape: ${output.converted.dims}`)
if (![...output.converted.data].every(Number.isFinite)) throw new Error('Neural output contains non-finite samples')
console.log('Distilled LLVC WASM smoke test passed: 240 input samples → 208 output samples')
