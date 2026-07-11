import { createFileRoute } from '@tanstack/react-router'
import { useEffect, useRef, useState } from 'react'

export const Route = createFileRoute('/')({ component: VoiceLab })

type VoiceMode = 'original' | 'lite' | 'neural'
type NeuralStatus = 'loading' | 'webgpu' | 'wasm' | 'error'
type Diagnostics = { averageMs: number; maxMs: number; p99Ms: number; queue: number; underruns: number; bufferedMs: number; threads: number; isolated: boolean }

const SAMPLES = [
  { id: 'english', language: 'EN', label: 'English', voice: 'Skylar', duration: '0:23', file: '/audio/english.wav' },
  { id: 'italian', language: 'IT', label: 'Italian', voice: 'Giulia', duration: '0:27', file: '/audio/italian.wav' },
  { id: 'spanish', language: 'ES', label: 'Spanish', voice: 'Lucia', duration: '0:28', file: '/audio/spanish.wav' },
  { id: 'french', language: 'FR', label: 'French', voice: 'Amélie', duration: '0:21', file: '/audio/french.wav' },
  { id: 'japanese', language: 'JA', label: 'Japanese', voice: 'Aiko', duration: '0:31', file: '/audio/japanese.wav' },
] as const

type AudioEngine = {
  ctx: AudioContext
  source: MediaElementAudioSourceNode
  analyser: AnalyserNode
  originalGain: GainNode
  liteGain: GainNode
  neuralGain: GainNode
  worklet: AudioWorkletNode
}

function VoiceLab() {
  const [voiceMode, setVoiceMode] = useState<VoiceMode>('original')
  const [selectedSample, setSelectedSample] = useState<string>(SAMPLES[0].id)
  const [neuralStatus, setNeuralStatus] = useState<NeuralStatus>('loading')
  const [level, setLevel] = useState(0)
  const [message, setMessage] = useState('Choose a sample to begin')
  const [diagnostics, setDiagnostics] = useState<Diagnostics>({ averageMs: 0, maxMs: 0, p99Ms: 0, queue: 0, underruns: 0, bufferedMs: 0, threads: 0, isolated: false })
  const audioRef = useRef<HTMLAudioElement>(null)
  const engineRef = useRef<AudioEngine | null>(null)
  const workerRef = useRef<Worker | null>(null)
  const modeRef = useRef<VoiceMode>('original')
  const neuralReadyRef = useRef(false)
  const meterFrame = useRef<number | null>(null)

  useEffect(() => {
    const worker = new Worker(new URL('../audio/neural-worker.ts', import.meta.url), { type: 'module' })
    workerRef.current = worker
    worker.onmessage = ({ data }) => {
      if (data.type === 'ready') {
        neuralReadyRef.current = true
        setNeuralStatus(data.runtime)
        setDiagnostics((current) => ({ ...current, threads: data.threads, isolated: data.isolated }))
        if (modeRef.current === 'neural') setMessage(`Neural ready · ${String(data.runtime).toUpperCase()} · ${data.isolated ? `${data.threads} threads` : 'single-thread'}`)
      } else if (data.type === 'output') {
        engineRef.current?.worklet.port.postMessage({ type: 'output', samples: data.samples }, [data.samples.buffer])
      } else if (data.type === 'error') {
        setNeuralStatus('error')
        setMessage(`Neural error · ${data.message}`)
      } else if (data.type === 'stats') {
        setDiagnostics((current) => ({ ...current, averageMs: data.averageMs, maxMs: data.maxMs, p99Ms: data.p99Ms, queue: data.queue }))
      }
    }
    worker.onerror = () => {
      setNeuralStatus('error')
      setMessage('Neural model could not be loaded')
    }
    worker.postMessage({ type: 'init' })

    return () => {
      worker.terminate()
      if (meterFrame.current) cancelAnimationFrame(meterFrame.current)
      void engineRef.current?.ctx.close()
    }
  }, [])

  function setGain(gain: GainNode, value: number, now: number) {
    gain.gain.cancelScheduledValues(now)
    gain.gain.setTargetAtTime(value, now, 0.015)
  }

  function applyMode(mode: VoiceMode, engine = engineRef.current) {
    modeRef.current = mode
    setVoiceMode(mode)
    if (!engine) return
    const now = engine.ctx.currentTime
    setGain(engine.originalGain, mode === 'original' ? 1 : 0, now)
    setGain(engine.liteGain, mode === 'lite' ? 0.92 : 0, now)
    setGain(engine.neuralGain, mode === 'neural' ? 1 : 0, now)
    if (mode === 'neural') {
      engine.worklet.port.postMessage({ type: 'reset' })
      workerRef.current?.postMessage({ type: 'reset' })
      setMessage(neuralReadyRef.current ? `Neural active · ${neuralStatus.toUpperCase()}` : 'Neural model is still loading…')
    } else {
      setMessage(mode === 'lite' ? 'Common Voice Lite active' : 'Original audio · processing disabled')
    }
  }

  function animateMeter(analyser: AnalyserNode) {
    const values = new Uint8Array(analyser.fftSize)
    const tick = () => {
      analyser.getByteTimeDomainData(values)
      let peak = 0
      for (const value of values) peak = Math.max(peak, Math.abs(value - 128) / 128)
      setLevel(Math.min(1, peak * 1.65))
      meterFrame.current = requestAnimationFrame(tick)
    }
    tick()
  }

  async function ensureAudioGraph() {
    if (engineRef.current) {
      await engineRef.current.ctx.resume()
      return engineRef.current
    }
    const audio = audioRef.current
    if (!audio) throw new Error('Audio player is unavailable')
    const ctx = new AudioContext()
    await ctx.audioWorklet.addModule('/voice-io-worklet.js')
    const source = ctx.createMediaElementSource(audio)
    const analyser = ctx.createAnalyser()
    analyser.fftSize = 256
    const originalGain = ctx.createGain()
    const liteGain = ctx.createGain()
    const neuralGain = ctx.createGain()
    originalGain.gain.value = 0
    liteGain.gain.value = 0
    neuralGain.gain.value = 0

    const highpass = ctx.createBiquadFilter()
    highpass.type = 'highpass'
    highpass.frequency.value = 90
    const voiceShape = ctx.createBiquadFilter()
    voiceShape.type = 'peaking'
    voiceShape.frequency.value = 1450
    voiceShape.Q.value = 0.72
    voiceShape.gain.value = 4.5
    const lowpass = ctx.createBiquadFilter()
    lowpass.type = 'lowpass'
    lowpass.frequency.value = 5200
    const compressor = ctx.createDynamicsCompressor()
    compressor.threshold.value = -28
    compressor.knee.value = 18
    compressor.ratio.value = 7
    compressor.attack.value = 0.006
    compressor.release.value = 0.16
    const saturation = ctx.createWaveShaper()
    saturation.curve = makeSoftCurve(7)
    saturation.oversample = '2x'
    const worklet = new AudioWorkletNode(ctx, 'neural-voice-io', { numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1] })

    source.connect(analyser)
    analyser.connect(originalGain).connect(ctx.destination)
    analyser.connect(highpass).connect(voiceShape).connect(lowpass).connect(compressor).connect(saturation).connect(liteGain).connect(ctx.destination)
    analyser.connect(worklet).connect(neuralGain).connect(ctx.destination)
    worklet.port.onmessage = ({ data }) => {
      if (data.type === 'input' && modeRef.current === 'neural' && neuralReadyRef.current) {
        workerRef.current?.postMessage({ type: 'audio', samples: data.samples }, [data.samples.buffer])
      } else if (data.type === 'stats') {
        setDiagnostics((current) => ({ ...current, underruns: data.underruns, bufferedMs: data.bufferedMs }))
      }
    }

    const engine = { ctx, source, analyser, originalGain, liteGain, neuralGain, worklet }
    engineRef.current = engine
    applyMode(modeRef.current, engine)
    animateMeter(analyser)
    await ctx.resume()
    return engine
  }

  async function playSample(sample: typeof SAMPLES[number]) {
    const audio = audioRef.current
    if (!audio) return
    setSelectedSample(sample.id)
    audio.pause()
    audio.src = sample.file
    audio.load()
    engineRef.current?.worklet.port.postMessage({ type: 'reset' })
    workerRef.current?.postMessage({ type: 'reset' })
    try {
      await ensureAudioGraph()
      await audio.play()
      setMessage(`${sample.label} · ${modeRef.current === 'original' ? 'Original' : modeRef.current === 'lite' ? 'Common Lite' : 'Common Neural'}`)
    } catch {
      setMessage('Press play in the audio player to begin')
    }
  }

  return (
    <main className="page-shell min-h-screen selection:bg-acid selection:text-ink">
      <nav className="topbar supports-[backdrop-filter]:backdrop-blur-sm">
        <div className="wordmark"><span className="mark">◒</span><span>VOICE LAB</span></div>
        <span className="nav-note"><span className="status-dot" /> LOCAL AUDIO EXPERIMENT</span>
      </nav>

      <section className="hero">
        <div className="eyebrow">BROWSER / MULTILINGUAL / 001</div>
        <h1>Many voices.<br /><em>One identity.</em></h1>
        <p className="lede">Compare five languages through an original signal, a shared tonal profile, and one browser-local neural voice.</p>
        <div className="hero-rule" />
      </section>

      <section className="lab-grid">
        <div className="panel input-panel supports-[backdrop-filter]:backdrop-blur-xl">
          <div className="panel-heading"><div><span className="section-index">01</span><h2>Audio samples</h2></div><span className="chip">SONIC 3</span></div>
          <div className="sample-list" role="listbox" aria-label="Audio sample">
            {SAMPLES.map((sample) => <button key={sample.id} className={`sample-button ${selectedSample === sample.id ? 'selected' : ''}`} onClick={() => void playSample(sample)} role="option" aria-selected={selectedSample === sample.id}><span className="language-code">{sample.language}</span><span><strong>{sample.label}</strong><small>{sample.voice}</small></span><time>{sample.duration}</time></button>)}
          </div>
          <audio ref={audioRef} src={SAMPLES[0].file} controls className="audio-player" onPlay={() => void ensureAudioGraph()} />
          <div className="privacy-note"><span>⌁</span> Pre-generated samples · Cartesia Sonic 3</div>
        </div>

        <div className="panel effect-panel supports-[backdrop-filter]:backdrop-blur-xl">
          <div className="panel-heading"><div><span className="section-index">02</span><h2>Voice mode</h2></div><span className="chip chip-live">AUTO</span></div>
          <div className="processor-switch" role="tablist" aria-label="Voice processing mode">
            <button className={voiceMode === 'original' ? 'selected' : ''} onClick={() => applyMode('original')} role="tab" aria-selected={voiceMode === 'original'}><span>01</span>Original</button>
            <button className={voiceMode === 'lite' ? 'selected' : ''} onClick={() => applyMode('lite')} role="tab" aria-selected={voiceMode === 'lite'}><span>02</span>Lite</button>
            <button className={voiceMode === 'neural' ? 'selected' : ''} onClick={() => applyMode('neural')} role="tab" aria-selected={voiceMode === 'neural'}><span>03</span>Neural</button>
          </div>
          <div className="chain"><span>PLAYBACK</span><i>→</i><span className="chain-active">{voiceMode === 'original' ? 'BYPASS' : voiceMode === 'lite' ? 'TONAL PROFILE' : 'LLVC'}</span><i>→</i><span>{voiceMode === 'neural' ? 'ONNX' : 'OUTPUT'}</span></div>
          <ModeCard mode={voiceMode} neuralStatus={neuralStatus} />
        </div>

        <div className="panel output-panel supports-[backdrop-filter]:backdrop-blur-xl">
          <div className="panel-heading"><div><span className="section-index">03</span><h2>Output</h2></div><span className="chip">MONITOR</span></div>
          <div className="meter-label"><span>SIGNAL LEVEL</span><span className="meter-value">{Math.round(level * 100).toString().padStart(3, '0')}%</span></div>
          <div className="meter" aria-label="Signal level"><div className="meter-fill" style={{ width: `${Math.max(2, level * 100)}%` }} /><div className="meter-ticks"><span /><span /><span /><span /><span /></div></div>
          <p className="output-copy">{message}</p>
          <div className={`mode-status ${voiceMode}`}><span>{voiceMode === 'original' ? '○' : '●'}</span><div><strong>{voiceMode === 'original' ? 'Processing disabled' : voiceMode === 'lite' ? 'Tonal profile enabled' : 'Neural conversion enabled'}</strong><small>{voiceMode === 'neural' ? `Runtime: ${neuralStatus.toUpperCase()} · ${diagnostics.isolated ? 'iso' : 'no-iso'} ${diagnostics.threads}t · avg ${diagnostics.averageMs.toFixed(1)}ms · p99 ${diagnostics.p99Ms.toFixed(0)}ms · max ${diagnostics.maxMs.toFixed(0)}ms · queue ${diagnostics.queue} · buffer ${diagnostics.bufferedMs.toFixed(0)}ms · underruns ${diagnostics.underruns}` : 'Mode changes apply automatically'}</small></div></div>
          <div className="tip"><span className="tip-star">✦</span><div><strong>Compare voices</strong><p>Replay the same sample while switching tabs, then compare languages to hear how consistently they converge.</p></div></div>
        </div>
      </section>

      <footer><span>VOICE LAB / SHARED IDENTITY</span><span>TOGETHER SONIC 3 · WEB AUDIO · ONNX</span></footer>
    </main>
  )
}

function makeSoftCurve(amount: number) {
  const curve = new Float32Array(44100)
  const drive = 1 + amount / 4
  for (let index = 0; index < curve.length; index++) {
    const x = (index * 2) / curve.length - 1
    curve[index] = Math.tanh(x * drive) / Math.tanh(drive)
  }
  return curve
}

function ModeCard({ mode, neuralStatus }: { mode: VoiceMode; neuralStatus: NeuralStatus }) {
  if (mode === 'original') return <div className="common-mode-card"><span className="mode-glyph">○</span><div><strong>Original audio</strong><p>The generated voice plays untouched. Use this baseline to hear how much identity changes in the other modes.</p><ul><li>No processing</li><li>Original timing and timbre</li><li>Immediate playback</li></ul></div></div>
  if (mode === 'lite') return <div className="common-mode-card"><span className="mode-glyph">≈</span><div><strong>Common Voice Lite</strong><p>A consistent tonal profile using voice-band EQ, strong dynamics normalization, soft saturation, and limiting.</p><ul><li>Near-zero latency</li><li>No model required</li><li>Subtle identity reduction</li></ul></div></div>
  return <div className="common-mode-card neural-card"><span className="mode-glyph">◇</span><div><strong>Common Voice Neural</strong><p>The LLVC any-to-one model maps every language and speaker toward one learned target identity.</p><ul><li>{neuralStatus === 'loading' ? 'Preloading model…' : neuralStatus === 'error' ? 'Model unavailable' : `${neuralStatus.toUpperCase()} ready`}</li><li>400 ms anti-glitch buffer</li><li>Browser-local inference</li></ul></div></div>
}
