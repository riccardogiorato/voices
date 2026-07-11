import * as ort from 'onnxruntime-web/webgpu'

type Runtime = 'webgpu' | 'wasm'
type WorkerMessage = { type: 'init' } | { type: 'audio'; samples: Float32Array } | { type: 'reset' }

const STATE_SHAPES: Record<string, number[]> = {
  enc_state: [1, 512, 510],
  dec_state: [1, 2, 13, 256],
  out_state: [1, 512, 4],
  conv_state: [1, 1, 24],
}

let session: ort.InferenceSession | null = null
let runtime: Runtime = 'wasm'
let states: Record<string, ort.Tensor> = {}
let context = new Float32Array(32)
let processing = Promise.resolve()
const workerScope = self as unknown as { postMessage: (message: unknown, transfer?: Transferable[]) => void; onmessage: ((event: MessageEvent<WorkerMessage>) => void) | null }

function zeros(shape: number[]) {
  return new ort.Tensor('float32', new Float32Array(shape.reduce((total, size) => total * size, 1)), shape)
}

function reset() {
  states = Object.fromEntries(Object.entries(STATE_SHAPES).map(([name, shape]) => [name, zeros(shape)]))
  context = new Float32Array(32)
}

async function createSession() {
  const [model, external] = await Promise.all([
    fetch('/models/common-voice-llvc.onnx').then((response) => response.arrayBuffer()),
    fetch('/models/common-voice-llvc.onnx.data').then((response) => response.arrayBuffer()),
  ])
  const options = {
    externalData: [{ path: 'common-voice-llvc.onnx.data', data: new Uint8Array(external) }],
    executionMode: 'sequential' as const,
    graphOptimizationLevel: 'all' as const,
  }

  try {
    session = await ort.InferenceSession.create(model, { ...options, executionProviders: ['webgpu'] })
    runtime = 'webgpu'
  } catch {
    session = await ort.InferenceSession.create(model, { ...options, executionProviders: ['wasm'] })
    runtime = 'wasm'
  }
  reset()
  workerScope.postMessage({ type: 'ready', runtime })
}

async function convert(samples: Float32Array) {
  if (!session) return
  const audio = new Float32Array(240)
  audio.set(context)
  audio.set(samples, 32)
  context = samples.slice(-32)
  const results = await session.run({
    audio: new ort.Tensor('float32', audio, [1, 1, 240]),
    ...states,
  })
  states = {
    enc_state: results.enc_state_next,
    dec_state: results.dec_state_next,
    out_state: results.out_state_next,
    conv_state: results.conv_state_next,
  }
  const output = new Float32Array(results.converted.data as Float32Array)
  workerScope.postMessage({ type: 'output', samples: output }, [output.buffer])
}

workerScope.onmessage = ({ data }: MessageEvent<WorkerMessage>) => {
  if (data.type === 'init') void createSession().catch((error) => workerScope.postMessage({ type: 'error', message: error instanceof Error ? error.message : 'Model initialization failed' }))
  if (data.type === 'reset') reset()
  if (data.type === 'audio') processing = processing.then(() => convert(data.samples)).catch((error) => workerScope.postMessage({ type: 'error', message: error instanceof Error ? error.message : 'Inference failed' }))
}
