export class StreamingAudioBuffer {
  constructor(capacity = 32768, prebufferSamples = 3200) {
    this.samples = new Float32Array(capacity)
    this.prebufferSamples = prebufferSamples
    this.readIndex = 0
    this.writeIndex = 0
    this.available = 0
    this.phase = 0
    this.started = false
    this.underruns = 0
    this.fade = 0
    this.lastSample = 0
  }

  reset() {
    this.readIndex = 0
    this.writeIndex = 0
    this.available = 0
    this.phase = 0
    this.started = false
    this.fade = 0
    this.lastSample = 0
  }

  push(chunk) {
    for (let i = 0; i < chunk.length; i++) {
      this.samples[this.writeIndex] = chunk[i]
      this.writeIndex = (this.writeIndex + 1) % this.samples.length
      if (this.available < this.samples.length) {
        this.available++
      } else {
        this.readIndex = (this.readIndex + 1) % this.samples.length
      }
    }
    if (!this.started && this.available >= this.prebufferSamples) this.started = true
  }

  render(output, sourceSamplesPerOutputSample) {
    for (let i = 0; i < output.length; i++) {
      if (this.started && this.available >= 2) {
        const nextIndex = (this.readIndex + 1) % this.samples.length
        const value = this.samples[this.readIndex] + (this.samples[nextIndex] - this.samples[this.readIndex]) * this.phase
        this.fade = Math.min(1, this.fade + 1 / 128)
        output[i] = value * this.fade
        this.lastSample = output[i]
        this.phase += sourceSamplesPerOutputSample
        while (this.phase >= 1) {
          this.phase -= 1
          this.readIndex = (this.readIndex + 1) % this.samples.length
          this.available--
        }
      } else {
        if (this.started) this.underruns++
        this.started = false
        this.fade = Math.max(0, this.fade - 1 / 64)
        this.lastSample *= 0.94
        output[i] = this.lastSample * this.fade
      }
    }
  }
}
