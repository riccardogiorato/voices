import { spawn } from 'node:child_process'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const port = 9334
const samplePath = process.argv[2]
const playbackSeconds = Number(process.argv[3] || 12)
const model = process.argv[4] || 'q8-high'
const threads = Number(process.argv[5] || 4)
const maxAverageMs = Number(process.env.PLAYBACK_THRESHOLD_MS || 13)
const profile = await mkdtemp(join(tmpdir(), 'voices-playback-chrome-'))
const chrome = spawn('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', [
  '--headless=new',
  `--remote-debugging-port=${port}`,
  `--user-data-dir=${profile}`,
  '--no-first-run',
  '--disable-background-networking',
  '--autoplay-policy=no-user-gesture-required',
  `http://localhost:5180/?model=${encodeURIComponent(model)}&threads=${threads}`,
], { stdio: 'ignore' })
const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms))

async function target() {
  for (let i = 0; i < 100; i++) {
    try {
      const values = await fetch(`http://127.0.0.1:${port}/json/list`).then((response) => response.json())
      const page = values.find((value) => value.type === 'page' && value.url.includes('localhost:5180'))
      if (page) return page
    } catch {}
    await delay(100)
  }
  throw new Error('Chrome debugging endpoint did not start')
}

try {
  const page = await target()
  const socket = new WebSocket(page.webSocketDebuggerUrl)
  await new Promise((resolve, reject) => {
    socket.addEventListener('open', resolve, { once: true })
    socket.addEventListener('error', reject, { once: true })
  })
  let id = 0
  const pending = new Map()
  socket.addEventListener('message', ({ data }) => {
    const message = JSON.parse(String(data))
    const callback = pending.get(message.id)
    if (callback) {
      pending.delete(message.id)
      callback(message)
    }
  })
  const send = (method, params = {}) => new Promise((resolve) => {
    const messageId = ++id
    pending.set(messageId, resolve)
    socket.send(JSON.stringify({ id: messageId, method, params }))
  })
  const evaluate = async (expression) => (await send('Runtime.evaluate', { expression, returnByValue: true })).result?.result?.value
  const click = async (expression) => {
    const clicked = await evaluate(`(() => { const e = ${expression}; if (!e) return false; e.click(); return true; })()`)
    if (!clicked) throw new Error(`Playback control not found: ${expression}`)
  }

  await send('Runtime.enable')
  for (let i = 0; i < 100; i++) {
    if (await evaluate(`(() => { const e=[...document.querySelectorAll('[role="tab"]')].find(e => e.textContent.includes('Neural')); return !!e && Object.keys(e).some(k => k.startsWith('__reactProps')); })()`)) break
    if (i === 99) throw new Error('Playback UI did not render')
    await delay(100)
  }
  await click(`[...document.querySelectorAll('[role="tab"]')].find(e => e.textContent.includes('Neural'))`)
  for (let i = 0; i < 200; i++) {
    if (await evaluate(`document.querySelector('.neural-card')?.textContent.includes('WASM ready')`)) break
    if (i === 199) {
      const diagnostic = await evaluate(`JSON.stringify({card:document.querySelector('.neural-card')?.textContent,status:document.querySelector('.mode-status small')?.textContent,message:document.querySelector('.output-copy')?.textContent})`)
      throw new Error(`Neural model did not become ready: ${diagnostic}`)
    }
    await delay(100)
  }
  if (samplePath) {
    const started = await evaluate(`(() => { const audio=document.querySelector('audio'); if (!audio) return false; audio.src=${JSON.stringify(samplePath)}; audio.load(); void audio.play(); return true; })()`)
    if (!started) throw new Error('Audio element was not available')
  } else {
    await click(`[...document.querySelectorAll('[role="option"]')].find(e => e.textContent.includes('English'))`)
  }
  await delay(playbackSeconds * 1000)
  const status = await evaluate(`document.querySelector('.mode-status.neural small')?.textContent || ''`)
  socket.close()
  console.log(status)
  const average = Number(status.match(/avg ([\d.]+)ms/)?.[1])
  const buffer = Number(status.match(/buffer ([\d.]+)ms/)?.[1])
  const underruns = Number(status.match(/underruns (\d+)/)?.[1])
  if (!status.includes(`iso ${threads}t`)) throw new Error(`Expected isolated ${threads}-thread WASM: ${status}`)
  if (!(average <= maxAverageMs)) throw new Error(`Playback average ${average}ms exceeds ${maxAverageMs}ms`)
  if (!(buffer >= 70)) throw new Error(`Playback buffer drained to ${buffer}ms`)
  if (underruns !== 0) throw new Error(`Playback had ${underruns} underruns`)
} finally {
  chrome.kill('SIGTERM')
  await rm(profile, { recursive: true, force: true })
}
