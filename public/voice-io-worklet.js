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
    this.outputBuffer = new StreamingAudioBuffer(32768, 3200)
    this.port.onmessage = ({ data }) => {
      if (data?.type === 'output') this.outputBuffer.push(data.samples)
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
    return true
  }
}

registerProcessor('neural-voice-io', NeuralVoiceIO)
