import { mkdir, writeFile } from 'node:fs/promises'

const samples = [
  {
    id: 'english', language: 'en', label: 'English', voiceName: 'Skylar - Friendly Guide', voice: 'db6b0ed5-d5d3-463d-ae85-518a07d3c2b4',
    text: 'Every voice carries a different history: a familiar rhythm, a regional accent, a quiet pause before an important thought. In this experiment, we listen to those differences and then guide them toward one shared voice. The words and timing remain recognizable, but the vocal identity becomes more consistent. This English passage gives the system enough time to settle into a natural speaking pattern and demonstrate the transformation clearly.',
  },
  {
    id: 'italian', language: 'it', label: 'Italian', voiceName: 'Giulia - Teacherly Voice', voice: '36d94908-c5b9-4014-b521-e69aee5bead0',
    text: 'Ogni voce racconta una storia diversa, attraverso il ritmo, l’accento e le piccole pause che rendono unico il modo di parlare. In questo esperimento ascoltiamo queste differenze e poi le accompagniamo verso una voce condivisa. Le parole, l’intonazione e il significato rimangono chiari, mentre l’identità vocale diventa più uniforme. Questo brano italiano permette al sistema di stabilizzarsi e mostrare la trasformazione in modo naturale.',
  },
  {
    id: 'spanish', language: 'es', label: 'Spanish', voiceName: 'Lucia - Radiant Host', voice: 'c0925108-d541-4dc4-bbae-39f4e57ba10c',
    text: 'Cada voz conserva una historia diferente en su ritmo, su acento y las pequeñas pausas que aparecen durante una conversación. En este experimento escuchamos esas diferencias y después las guiamos hacia una sola voz compartida. Las palabras, la intención y el tiempo permanecen claros, mientras la identidad vocal se vuelve más consistente. Este fragmento en español ofrece suficiente duración para que el sistema se estabilice y permita escuchar la transformación con claridad.',
  },
  {
    id: 'french', language: 'fr', label: 'French', voiceName: 'Amélie - Decisive Agent', voice: 'faa75703-00e3-4a57-9955-0703001e3231',
    text: 'Chaque voix porte une histoire différente dans son rythme, son accent et les silences qui donnent du relief à une conversation. Dans cette expérience, nous écoutons ces différences avant de les guider vers une voix commune. Les mots, l’intention et la cadence restent compréhensibles, tandis que l’identité vocale devient plus cohérente. Ce passage en français est assez long pour laisser le système se stabiliser et rendre la transformation facile à entendre.',
  },
  {
    id: 'japanese', language: 'ja', label: 'Japanese', voiceName: 'Aiko - Calming Voice', voice: '498e7f37-7fa3-4e2c-b8e2-8b6e9276f956',
    text: '声にはそれぞれ異なる歴史があります。話す速さ、地域のアクセント、大切なことを伝える前の短い間にも、その人らしさが表れます。この実験では、そうした違いを保ちながら、すべての音声を一つの共通した声へ近づけます。言葉の意味や話すタイミングは分かりやすく残り、声の個性だけが少しずつ統一されます。この日本語の文章は、変換が安定し、その効果を自然に確認できる長さになっています。',
  },
]

if (!process.env.TOGETHER_API_KEY) throw new Error('TOGETHER_API_KEY is missing from .env')
await mkdir('public/audio', { recursive: true })

for (const sample of samples) {
  console.log(`Generating ${sample.label} with ${sample.voiceName}…`)
  const response = await fetch('https://api.together.ai/v1/audio/speech', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${process.env.TOGETHER_API_KEY}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: 'cartesia/sonic-3',
      input: sample.text,
      voice: sample.voice,
      language: sample.language,
      response_format: 'wav',
      sample_rate: 24000,
      stream: false,
    }),
  })
  if (!response.ok) throw new Error(`${sample.label}: ${response.status} ${await response.text()}`)
  const audio = new Uint8Array(await response.arrayBuffer())
  if (String.fromCharCode(...audio.slice(0, 4)) !== 'RIFF') throw new Error(`${sample.label}: API did not return a WAV file`)
  await writeFile(`public/audio/${sample.id}.wav`, audio)
  console.log(`Saved public/audio/${sample.id}.wav (${(audio.byteLength / 1024 / 1024).toFixed(1)} MB)`)
}

await writeFile('public/audio/samples.json', JSON.stringify(samples.map(({ text, voice, ...sample }) => ({ ...sample, file: `/audio/${sample.id}.wav` })), null, 2) + '\n')
