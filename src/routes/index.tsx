import { createFileRoute } from '@tanstack/react-router'
import { useEffect, useRef, useState } from 'react'

export const Route = createFileRoute('/')({ component: VoiceLab })

type Mode = 'mic' | 'file'
type ProcessorMode = 'lite' | 'neural' | 'effects'
type Params = { distortion: number; cutoff: number; pitch: number; intensity: number }
type PresetId = 'daft-punk' | 'cyber-bass' | 'radio-bot' | 'alien-glass' | 'custom'

const DEFAULTS: Params = { distortion: 45, cutoff: 1200, pitch: -4, intensity: 0.7 }
const PRESETS: Array<{ id: Exclude<PresetId, 'custom'>; name: string; description: string; params: Params }> = [
  { id: 'daft-punk', name: 'Daft Punk', description: 'Warm metallic talkbox', params: { distortion: 45, cutoff: 1200, pitch: -4, intensity: 0.7 } },
  { id: 'cyber-bass', name: 'Cyber Bass', description: 'Deep, heavy machine voice', params: { distortion: 72, cutoff: 760, pitch: -9, intensity: 0.9 } },
  { id: 'radio-bot', name: 'Radio Bot', description: 'Tight transmission crunch', params: { distortion: 58, cutoff: 1850, pitch: 0, intensity: 0.62 } },
  { id: 'alien-glass', name: 'Alien Glass', description: 'Bright synthetic character', params: { distortion: 25, cutoff: 3400, pitch: 7, intensity: 0.5 } },
]

function VoiceLab() {
  const [mode, setMode] = useState<Mode>('mic')
  const [processor, setProcessor] = useState<ProcessorMode>('lite')
  const [active, setActive] = useState(false)
  const [params, setParams] = useState(DEFAULTS)
  const [preset, setPreset] = useState<PresetId>('daft-punk')
  const [fileName, setFileName] = useState('No file loaded')
  const [level, setLevel] = useState(0)
  const [message, setMessage] = useState('Ready when you are')
  const audioRef = useRef<HTMLAudioElement>(null)
  const fileUrlRef = useRef<string | null>(null)
  const engine = useRef<{
    ctx: AudioContext
    source: AudioNode
    distortion?: WaveShaperNode
    lowpass?: BiquadFilterNode
    bandpass?: BiquadFilterNode
    gain?: GainNode
    analyser: AnalyserNode
    stream?: MediaStream
    worker?: Worker
    worklet?: AudioWorkletNode
  } | null>(null)
  const meterFrame = useRef<number | null>(null)

  useEffect(() => () => {
    if (meterFrame.current) cancelAnimationFrame(meterFrame.current)
    if (fileUrlRef.current) URL.revokeObjectURL(fileUrlRef.current)
    engine.current?.stream?.getTracks().forEach((track) => track.stop())
    void engine.current?.ctx.close()
  }, [])

  function makeCurve(amount: number) {
    const curve = new Float32Array(44100)
    const deg = Math.PI * amount / 100
    for (let i = 0; i < curve.length; i++) {
      const x = (i * 2) / curve.length - 1
      curve[i] = ((3 + deg) * x * 20 * deg) / (Math.PI + deg * Math.abs(x))
    }
    return curve
  }

  function updateAudio(next: Params) {
    setParams(next)
    const current = engine.current
    if (!current) return
    if (current.distortion) current.distortion.curve = makeCurve(next.distortion)
    if (current.lowpass) current.lowpass.frequency.setTargetAtTime(next.cutoff, current.ctx.currentTime, 0.015)
    if (current.gain) current.gain.gain.setTargetAtTime(0.7 + next.intensity * 0.4, current.ctx.currentTime, 0.015)
    if (audioRef.current) audioRef.current.playbackRate = Math.pow(2, next.pitch / 12)
  }

  function selectPreset(nextPreset: PresetId) {
    setPreset(nextPreset)
    if (nextPreset === 'custom') return
    const selected = PRESETS.find((item) => item.id === nextPreset)
    if (selected) updateAudio(selected.params)
  }

  function selectProcessor(nextProcessor: ProcessorMode) {
    if (active) stop()
    setProcessor(nextProcessor)
    setMessage(nextProcessor === 'neural' ? 'Neural model loads locally when engaged' : nextProcessor === 'lite' ? 'Common Voice Lite is ready' : 'Robot effects are ready')
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

  async function start() {
    try {
      const ctx = new AudioContext()
      await ctx.resume()
      const analyser = ctx.createAnalyser()
      analyser.fftSize = 256
      let source: AudioNode
      let stream: MediaStream | undefined

      if (mode === 'mic') {
        stream = await navigator.mediaDevices.getUserMedia({ audio: true })
        source = ctx.createMediaStreamSource(stream)
        setMessage('Mic live · use headphones to avoid feedback')
      } else if (audioRef.current?.src) {
        source = ctx.createMediaElementSource(audioRef.current)
        setMessage('File live · play the track below')
      } else {
        await ctx.close()
        setMessage('Choose an audio file first')
        return
      }

      if (audioRef.current) audioRef.current.playbackRate = processor === 'effects' ? Math.pow(2, params.pitch / 12) : 1

      if (processor === 'neural') {
        setMessage('Loading Common Voice Neural locally…')
        await ctx.audioWorklet.addModule('/voice-io-worklet.js')
        const worklet = new AudioWorkletNode(ctx, 'neural-voice-io', { numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1] })
        const worker = new Worker(new URL('../audio/neural-worker.ts', import.meta.url), { type: 'module' })
        await new Promise<void>((resolve, reject) => {
          const timeout = window.setTimeout(() => reject(new Error('Model loading timed out')), 45000)
          worker.onmessage = ({ data }) => {
            if (data.type === 'ready') {
              window.clearTimeout(timeout)
              setMessage(`Common Voice Neural live · ${String(data.runtime).toUpperCase()}`)
              resolve()
            } else if (data.type === 'output') {
              worklet.port.postMessage({ type: 'output', samples: data.samples }, [data.samples.buffer])
            } else if (data.type === 'error') {
              setMessage(`Neural error · ${data.message}`)
            }
          }
          worker.onerror = () => reject(new Error('Neural worker failed to load'))
          worker.postMessage({ type: 'init' })
        })
        worklet.port.onmessage = ({ data }) => {
          if (data.type === 'input') worker.postMessage({ type: 'audio', samples: data.samples }, [data.samples.buffer])
        }
        source.connect(analyser).connect(worklet).connect(ctx.destination)
        engine.current = { ctx, source, analyser, stream, worker, worklet }
      } else if (processor === 'lite') {
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
        saturation.curve = makeCurve(7)
        saturation.oversample = '2x'
        const gain = ctx.createGain()
        gain.gain.value = 0.9
        source.connect(highpass).connect(voiceShape).connect(lowpass).connect(compressor).connect(saturation).connect(gain).connect(analyser).connect(ctx.destination)
        engine.current = { ctx, source, lowpass, gain, analyser, stream }
        setMessage('Common Voice Lite live · browser DSP')
      } else {
        const distortion = ctx.createWaveShaper()
        distortion.oversample = '4x'
        distortion.curve = makeCurve(params.distortion)
        const lowpass = ctx.createBiquadFilter()
        lowpass.type = 'lowpass'
        lowpass.frequency.value = params.cutoff
        const bandpass = ctx.createBiquadFilter()
        bandpass.type = 'bandpass'
        bandpass.frequency.value = 800
        bandpass.Q.value = 2
        const gain = ctx.createGain()
        gain.gain.value = 0.7 + params.intensity * 0.4
        source.connect(distortion).connect(lowpass).connect(bandpass).connect(gain).connect(analyser).connect(ctx.destination)
        engine.current = { ctx, source, distortion, lowpass, bandpass, gain, analyser, stream }
        setMessage('Robot effect live')
      }
      animateMeter(analyser)
      setActive(true)
    } catch (error) {
      setMessage(error instanceof DOMException && error.name === 'NotAllowedError' ? 'Microphone permission is needed' : 'Could not start audio')
    }
  }

  function stop() {
    if (meterFrame.current) cancelAnimationFrame(meterFrame.current)
    engine.current?.source.disconnect()
    engine.current?.worklet?.disconnect()
    engine.current?.worker?.terminate()
    engine.current?.stream?.getTracks().forEach((track) => track.stop())
    void engine.current?.ctx.close()
    engine.current = null
    setLevel(0)
    setActive(false)
    setMessage('Paused · audio stays in your browser')
  }

  function chooseFile(file?: File) {
    if (!file) return
    if (fileUrlRef.current) URL.revokeObjectURL(fileUrlRef.current)
    fileUrlRef.current = URL.createObjectURL(file)
    setFileName(file.name)
    setMode('file')
    if (audioRef.current) {
      audioRef.current.src = fileUrlRef.current
      audioRef.current.load()
    }
    setMessage('File ready · press engage, then play')
  }

  function toggle() {
    if (active) stop()
    else void start()
  }

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) return
      if (event.key.toLowerCase() === 'r') toggle()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  })

  return (
    <main className="page-shell min-h-screen selection:bg-acid selection:text-ink">
      <nav className="topbar supports-[backdrop-filter]:backdrop-blur-sm">
        <div className="wordmark"><span className="mark">◒</span><span>VOICE LAB</span></div>
        <span className="nav-note"><span className="status-dot" /> LOCAL AUDIO EXPERIMENT</span>
      </nav>

      <section className="hero">
        <div className="eyebrow">BROWSER / REALTIME / 001</div>
        <h1>Make your voice<br /><em>otherworldly.</em></h1>
        <p className="lede">A tiny Web Audio playground for metallic, robotic voice conversion. Everything runs locally in your browser.</p>
        <div className="hero-rule" />
      </section>

      <section className="lab-grid">
        <div className="panel input-panel supports-[backdrop-filter]:backdrop-blur-xl">
          <div className="panel-heading"><div><span className="section-index">01</span><h2>Input</h2></div><span className="chip">SOURCE</span></div>
          <div className="mode-switch" role="tablist" aria-label="Input mode">
            <button className={mode === 'mic' ? 'selected focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-acid' : 'focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-acid'} onClick={() => setMode('mic')} role="tab">Microphone</button>
            <button className={mode === 'file' ? 'selected focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-acid' : 'focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-acid'} onClick={() => setMode('file')} role="tab">Audio file</button>
          </div>
          {mode === 'mic' ? <div className="mic-card"><div className="mic-icon">♩</div><div><strong>Live microphone</strong><span>Permission requested only when you engage</span></div><span className="live-pill">LIVE INPUT</span></div> : <label className="dropzone"><input type="file" accept="audio/*" onChange={(event) => chooseFile(event.target.files?.[0])} /><span className="upload-icon">↥</span><strong>Drop an audio file here</strong><span>{fileName}</span></label>}
          <audio ref={audioRef} controls className="audio-player" />
          <div className="privacy-note"><span>⌁</span> No upload. No server. Just signal processing.</div>
        </div>

        <div className="panel effect-panel supports-[backdrop-filter]:backdrop-blur-xl">
          <div className="panel-heading"><div><span className="section-index">02</span><h2>Voice mode</h2></div><span className={active ? 'chip chip-live' : 'chip'}>{active ? 'ACTIVE' : 'BYPASS'}</span></div>
          <div className="processor-switch" role="tablist" aria-label="Voice processing mode">
            <button className={processor === 'lite' ? 'selected' : ''} onClick={() => selectProcessor('lite')} role="tab">Common Lite</button>
            <button className={processor === 'neural' ? 'selected' : ''} onClick={() => selectProcessor('neural')} role="tab">Common Neural</button>
            <button className={processor === 'effects' ? 'selected' : ''} onClick={() => selectProcessor('effects')} role="tab">Effects</button>
          </div>
          <div className="chain"><span>INPUT</span><i>→</i><span className="chain-active">{processor === 'neural' ? 'LLVC' : processor === 'lite' ? 'NORMALIZE' : 'WAVESHAPER'}</span><i>→</i><span>{processor === 'neural' ? 'ONNX' : 'FILTER'}</span><i>→</i><span>OUTPUT</span></div>
          {processor === 'lite' && <div className="common-mode-card"><span className="mode-glyph">≈</span><div><strong>Common Voice Lite</strong><p>Instantly narrows every voice toward one shared tonal profile using EQ, dynamics normalization, soft saturation, and limiting.</p><ul><li>Zero download</li><li>Near-zero latency</li><li>Works in every modern browser</li></ul></div></div>}
          {processor === 'neural' && <div className="common-mode-card neural-card"><span className="mode-glyph">◇</span><div><strong>Common Voice Neural</strong><p>Runs the 16 kHz LLVC any-to-one model entirely on your device. WebGPU is preferred; WASM activates automatically as fallback.</p><ul><li>One shared learned voice</li><li>~14 MB model download</li><li>No audio leaves this tab</li></ul></div></div>}
          {processor === 'effects' && <><div className="preset-list" role="radiogroup" aria-label="Voice effect preset">
            {PRESETS.map((item) => <button key={item.id} className={`preset-button ${preset === item.id ? 'selected' : ''}`} role="radio" aria-checked={preset === item.id} onClick={() => selectPreset(item.id)}><span><strong>{item.name}</strong><small>{item.description}</small></span><i>{preset === item.id ? '●' : '○'}</i></button>)}
            <button className={`preset-button custom-preset ${preset === 'custom' ? 'selected' : ''}`} role="radio" aria-checked={preset === 'custom'} onClick={() => selectPreset('custom')}><span><strong>Custom</strong><small>Dial in every parameter</small></span><i>{preset === 'custom' ? '●' : '○'}</i></button>
          </div>
          {preset === 'custom' && <div className="controls custom-controls">
            <Control label="Distortion" hint="metallic crunch" value={params.distortion} min={0} max={100} suffix="%" onChange={(value) => updateAudio({ ...params, distortion: value })} />
            <Control label="Filter cutoff" hint="robotic midrange" value={params.cutoff} min={200} max={4000} suffix=" Hz" onChange={(value) => updateAudio({ ...params, cutoff: value })} />
            <Control label="Pitch shift" hint="playback rate" value={params.pitch} min={-12} max={12} step={0.5} suffix=" st" onChange={(value) => updateAudio({ ...params, pitch: value })} />
            <Control label="Intensity" hint="overall strength" value={params.intensity} min={0} max={1} step={0.05} suffix="" format={(value) => value.toFixed(2)} onChange={(value) => updateAudio({ ...params, intensity: value })} />
          </div>}</>}
        </div>

        <div className="panel output-panel supports-[backdrop-filter]:backdrop-blur-xl">
          <div className="panel-heading"><div><span className="section-index">03</span><h2>Output</h2></div><span className="chip">MONITOR</span></div>
          <div className="meter-label"><span>INPUT LEVEL</span><span className="meter-value">{Math.round(level * 100).toString().padStart(3, '0')}%</span></div>
          <div className="meter" aria-label="Input level"><div className="meter-fill" style={{ width: `${Math.max(2, level * 100)}%` }} /><div className="meter-ticks"><span /><span /><span /><span /><span /></div></div>
          <p className="output-copy">{message}</p>
          <button className={`engage-button focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-acid ${active ? 'engaged' : ''}`} onClick={toggle}><span className="power-icon">{active ? '■' : '↯'}</span>{active ? `DISENGAGE ${processor === 'effects' ? 'EFFECT' : 'VOICE'}` : `ENGAGE ${processor === 'effects' ? 'EFFECT' : 'VOICE'}`}<kbd>R</kbd></button>
          <div className="tip"><span className="tip-star">✦</span><div><strong>Quick tip</strong><p>{processor === 'neural' ? 'The first start downloads and caches the local neural model. Headphones prevent feedback.' : processor === 'lite' ? 'Lite makes voices more consistent, but Neural removes much more speaker identity.' : 'Try low cutoff + high distortion for a classic vocoder texture.'}</p></div></div>
        </div>
      </section>

      <footer><span>VOICE LAB / EXPERIMENTAL AUDIO</span><span>WEB AUDIO API · CLIENT-SIDE ONLY</span></footer>
    </main>
  )
}

function Control({ label, hint, value, min, max, step = 1, suffix, format = (value: number) => String(value), onChange }: { label: string; hint: string; value: number; min: number; max: number; step?: number; suffix: string; format?: (value: number) => string; onChange: (value: number) => void }) {
  return <label className="control"><div className="control-top"><span><strong>{label}</strong><small>{hint}</small></span><output>{format(value)}{suffix}</output></div><input type="range" min={min} max={max} step={step} value={value} onChange={(event) => onChange(Number(event.target.value))} style={{ '--progress': `${((value - min) / (max - min)) * 100}%` } as React.CSSProperties} /></label>
}
