import { execFile } from 'node:child_process'
import { mkdir, writeFile } from 'node:fs/promises'
import { promisify } from 'node:util'

const run = promisify(execFile)
const models = ['current', 'fp16', 'q8-high', 'q8-balanced', 'q8-max']
const threadCounts = [1, 2, 4]
const results = []

for (const model of models) {
  for (const threads of threadCounts) {
    process.stdout.write(`Chrome ${model} ${threads}t ... `)
    const { stdout } = await run('bun', ['scripts/benchmark-web.mjs', String(threads), model], {
      env: { ...process.env, BENCHMARK_THRESHOLD_MS: '100' },
      maxBuffer: 1024 * 1024,
    })
    const result = JSON.parse(stdout)
    results.push(result)
    console.log(`${result.averageMs.toFixed(3)}ms (p99 ${result.p99Ms.toFixed(3)}ms)`)
  }
}

const report = {
  generatedAt: new Date().toISOString(),
  browser: 'Google Chrome headless, direct WASM benchmark, 8 warmups + 250 measured stateful chunks',
  results,
}
await mkdir('reports', { recursive: true })
await writeFile('reports/quantization-chrome-benchmark.json', `${JSON.stringify(report, null, 2)}\n`)
console.log('Wrote reports/quantization-chrome-benchmark.json')
