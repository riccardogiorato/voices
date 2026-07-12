import { execFile } from 'node:child_process'
import { mkdir, writeFile } from 'node:fs/promises'
import { promisify } from 'node:util'

const run = promisify(execFile)
const samples = ['english-samantha', 'french-jacques', 'italian-flo', 'spanish-grandma']
const variants = [
  { model: 'current', threads: 2 },
  { model: 'q8-high', threads: 4 },
  { model: 'q8-max', threads: 4 },
]
const seconds = 12
const results = []

for (const sample of samples) {
  const path = `/audio/short-unseen-tests/${sample}.wav`
  for (const variant of variants) {
    process.stdout.write(`Chrome ${sample} / ${variant.model} ... `)
    const { stdout } = await run('bun', [
      'scripts/test-playback-web.mjs', path, String(seconds), variant.model, String(variant.threads),
    ], {
      env: { ...process.env, PLAYBACK_THRESHOLD_MS: '13' },
      maxBuffer: 1024 * 1024,
    })
    const status = stdout.trim()
    console.log(status)
    results.push({ sample, ...variant, seconds, status })
  }
}

const report = { generatedAt: new Date().toISOString(), results }
await mkdir('reports', { recursive: true })
await writeFile('reports/quantization-short-unseen-playback.json', `${JSON.stringify(report, null, 2)}\n`)
console.log('Wrote reports/quantization-short-unseen-playback.json')
