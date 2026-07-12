import { spawn } from 'node:child_process'
import { mkdtemp, readFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const port = 9333
const threads = Number(process.argv[2] || 8)
const model = process.argv[3] || 'q8-high'
const thresholdMs = Number(process.env.BENCHMARK_THRESHOLD_MS || 2.4)
const profile = await mkdtemp(join(tmpdir(), 'voices-chrome-'))
const chrome = spawn('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', [
  '--headless=new',
  `--remote-debugging-port=${port}`,
  `--user-data-dir=${profile}`,
  '--no-first-run',
  '--disable-background-networking',
  `http://localhost:5180/?benchmark=1&threads=${threads}&model=${encodeURIComponent(model)}`,
], { stdio: 'ignore' })

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms))

async function targets() {
  for (let i = 0; i < 100; i++) {
    try {
      const value = await fetch(`http://127.0.0.1:${port}/json/list`).then((response) => response.json())
      const target = value.find((candidate) => candidate.type === 'page' && candidate.url.includes('localhost:5180'))
      if (target) return target
    } catch {}
    await delay(100)
  }
  throw new Error('Chrome debugging endpoint did not start')
}

try {
  const target = await targets()
  const socket = new WebSocket(target.webSocketDebuggerUrl)
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
  await send('Runtime.enable')
  const consoleMessages = []
  socket.addEventListener('message', ({ data }) => {
    const message = JSON.parse(String(data))
    if (message.method === 'Runtime.consoleAPICalled') consoleMessages.push(message.params.args.map((arg) => arg.value ?? arg.description).join(' '))
    if (message.method === 'Runtime.exceptionThrown') consoleMessages.push(message.params.exceptionDetails.text)
  })
  let result
  for (let i = 0; i < 200; i++) {
    const response = await send('Runtime.evaluate', {
      expression: `document.querySelector('[data-testid="benchmark-result"]')?.textContent || ''`,
      returnByValue: true,
    })
    const text = response.result?.result?.value
    if (text) {
      result = JSON.parse(text)
      break
    }
    await delay(100)
  }
  if (!result) {
    const diagnostic = await send('Runtime.evaluate', {
      expression: `JSON.stringify({status: document.querySelector('.mode-status small')?.textContent, message: document.querySelector('.output-copy')?.textContent, result: document.querySelector('[data-testid="benchmark-result"]')?.textContent})`,
      returnByValue: true,
    })
    socket.close()
    throw new Error(`Browser benchmark timed out: ${diagnostic.result?.result?.value}; console=${consoleMessages.join(' | ')}`)
  }
  socket.close()
  console.log(JSON.stringify(result, null, 2))
  if (!result.finite || result.peak < 1e-5) throw new Error('Browser benchmark produced invalid or silent output')
  if (result.averageMs > thresholdMs) throw new Error(`Average ${result.averageMs.toFixed(2)}ms exceeds ${thresholdMs.toFixed(2)}ms target`)
} finally {
  chrome.kill('SIGTERM')
  await rm(profile, { recursive: true, force: true })
}
