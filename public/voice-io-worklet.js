// AudioWorklet for the neural path. Captures input at the device rate, decimates
// to 16 kHz in 208-sample chunks for the worker, and renders the worker's
// converted samples back at the device rate through a streaming ring buffer.
//
// The distilled model runs roughly 5x faster than real time, so 100 ms of
// prebuffer absorbs browser scheduling jitter without imposing the old 400 ms
// startup delay.

import { StreamingAudioBuffer } from './audio-ring-buffer.js'

class NeuralVoiceIO extends AudioWorkletProcessor {
  constructor() {
    super()
    this.inputRatio = sampleRate / 16000
    this.inputPosition = 0
    this.inputPrevious = 0
    this.inputChunk = new Float32Array(208)
    this.inputCount = 0
    this.outputRatio = 16000 / sampleRate
    this.outputBuffer = new StreamingAudioBuffer(65536, 1600)
    this.processCalls = 0
    this.port.onmessage = ({ data }) => {
      if (data?.type === 'output') this.outputBuffer.push(data.samples)
      if (data?.type === 'reset') this.outputBuffer.reset()
    }
  }

  capture(input) {
    while (this.inputPosition < input.length) {
      const left = Math.floor(this.inputPosition)
      const right = Math.min(left + 1, input.length - 1)
      const mix = this.inputPosition - left
      const a = left === 0 ? this.inputPrevious : input[left]
      this.inputChunk[this.inputCount++] = a + (input[right] - a) * mix
      this.inputPosition += this.inputRatio
      if (this.inputCount === this.inputChunk.length) {
        const samples = this.inputChunk
        this.port.postMessage({ type: 'input', samples }, [samples.buffer])
        this.inputChunk = new Float32Array(208)
        this.inputCount = 0
      }
    }
    this.inputPosition -= input.length
    this.inputPrevious = input[input.length - 1] || this.inputPrevious
  }

  process(inputs, outputs) {
    const input = inputs[0]?.[0]
    const output = outputs[0]?.[0]
    if (input?.length) this.capture(input)
    if (output?.length) this.outputBuffer.render(output, this.outputRatio)
    this.processCalls++
    if (this.processCalls % Math.max(1, Math.round(sampleRate / 128)) === 0) {
      this.port.postMessage({ type: 'stats', underruns: this.outputBuffer.underruns, bufferedMs: this.outputBuffer.available / 16 })
    }
    return true
  }
}

registerProcessor('neural-voice-io', NeuralVoiceIO)
