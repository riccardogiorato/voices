import { mkdirSync, unlinkSync } from 'node:fs'
import { spawnSync } from 'node:child_process'
import { join } from 'node:path'

const outputDir = 'public/audio/long-tests'
mkdirSync(outputDir, { recursive: true })

const samples = {
  english: {
    voice: 'Daniel',
    text: `On a clear autumn morning, the city wakes slowly beneath a pale blue sky. Buses begin their routes, cafés lift their shutters, and the first customers pause for warm coffee before work. Near the river, a woman walks her dog while listening to the bells from an old stone church. She has planned a long day: a meeting with colleagues, lunch with a childhood friend, and an evening train to the coast. Although the schedule is busy, she notices small details along the way. Leaves turn in the wind, bicycles cross the bridge, and someone practices a violin behind an open window. These ordinary sounds make the familiar streets feel new again. By noon, clouds gather above the rooftops, but the rain never arrives. Instead, sunlight returns and follows her through the market, where vendors call out prices for bread, flowers, fruit, and bright red peppers.`,
  },
  italian: {
    voice: 'Alice',
    text: `In una mattina limpida d'autunno, la città si sveglia lentamente sotto un cielo azzurro e leggero. Gli autobus iniziano il loro percorso, i bar alzano le serrande e i primi clienti si fermano per un caffè caldo prima del lavoro. Vicino al fiume, una donna porta a passeggio il cane mentre ascolta le campane di una vecchia chiesa di pietra. Ha organizzato una giornata lunga: una riunione con i colleghi, un pranzo con un'amica d'infanzia e un treno serale verso la costa. Anche se il programma è pieno, osserva i piccoli dettagli lungo la strada. Le foglie girano nel vento, le biciclette attraversano il ponte e qualcuno suona il violino dietro una finestra aperta. Questi suoni quotidiani rendono nuove le vie familiari. A mezzogiorno arrivano alcune nuvole, ma la pioggia non cade. Il sole ritorna e la accompagna fino al mercato, tra pane, fiori, frutta e peperoni rossi.`,
  },
  spanish: {
    voice: 'Mónica',
    text: `En una mañana clara de otoño, la ciudad despierta lentamente bajo un cielo azul y tranquilo. Los autobuses comienzan sus recorridos, las cafeterías levantan las persianas y los primeros clientes se detienen a tomar algo caliente antes del trabajo. Cerca del río, una mujer pasea con su perro mientras escucha las campanas de una antigua iglesia de piedra. Ha preparado un día largo: una reunión con sus compañeros, un almuerzo con una amiga de la infancia y un tren nocturno hacia la costa. Aunque la agenda está llena, observa los pequeños detalles del camino. Las hojas giran con el viento, las bicicletas cruzan el puente y alguien practica el violín detrás de una ventana abierta. Estos sonidos cotidianos hacen que las calles conocidas parezcan nuevas. Al mediodía aparecen nubes sobre los tejados, pero la lluvia no llega. El sol regresa y la acompaña hasta el mercado, entre puestos de pan, flores, fruta fresca y pimientos rojos.`,
  },
  french: {
    voice: 'Thomas',
    text: `Par une claire matinée d'automne, la ville se réveille lentement sous un ciel bleu et paisible. Les autobus commencent leur trajet, les cafés ouvrent leurs volets et les premiers clients s'arrêtent pour boire quelque chose de chaud avant le travail. Près de la rivière, une femme promène son chien en écoutant les cloches d'une vieille église de pierre. Elle a prévu une longue journée : une réunion avec ses collègues, un déjeuner avec une amie d'enfance et un train du soir vers la côte. Malgré son emploi du temps chargé, elle remarque les petits détails du chemin. Les feuilles tournent dans le vent, les vélos traversent le pont et quelqu'un joue du violon derrière une fenêtre ouverte. Ces bruits ordinaires donnent un air nouveau aux rues familières. À midi, des nuages arrivent au-dessus des toits, mais la pluie ne tombe pas. Le soleil revient et l'accompagne jusqu'au marché, parmi le pain, les fleurs, les fruits et les poivrons rouges.`,
  },
}

for (const [language, sample] of Object.entries(samples)) {
  const aiff = join(outputDir, `${language}.aiff`)
  const wav = join(outputDir, `${language}.wav`)
  const say = spawnSync('say', ['-v', sample.voice, '-r', '145', '-o', aiff, sample.text], { stdio: 'inherit' })
  if (say.status !== 0) process.exit(say.status ?? 1)
  const ffmpeg = spawnSync('ffmpeg', ['-loglevel', 'error', '-y', '-i', aiff, '-ar', '24000', '-ac', '1', '-c:a', 'pcm_s16le', wav], { stdio: 'inherit' })
  if (ffmpeg.status !== 0) process.exit(ffmpeg.status ?? 1)
  unlinkSync(aiff)
}
