import { execFile } from 'node:child_process'
import { mkdir, writeFile } from 'node:fs/promises'
import { promisify } from 'node:util'

const run = promisify(execFile)
const sample = '/audio/long-tests/english.wav'
const seconds = 45
const variants = [
  { model: 'current', threads: 2 },
  { model: 'q8-high', threads: 4 },
  { model: 'q8-max', threads: 4 },
]
const results = []

for (const variant of variants) {
  process.stdout.write(`Playback ${variant.model} ${variant.threads}t for ${seconds}s ... `)
  const { stdout } = await run('bun', [
    'scripts/test-playback-web.mjs', sample, String(seconds), variant.model, String(variant.threads),
  ], {
    env: { ...process.env, PLAYBACK_THRESHOLD_MS: '13' },
    maxBuffer: 1024 * 1024,
  })
  const status = stdout.trim()
  console.log(status)
  results.push({ ...variant, sample, seconds, status })
}

const report = { generatedAt: new Date().toISOString(), results }
await mkdir('reports', { recursive: true })
await writeFile('reports/quantization-playback-test.json', `${JSON.stringify(report, null, 2)}\n`)
console.log('Wrote reports/quantization-playback-test.json')
