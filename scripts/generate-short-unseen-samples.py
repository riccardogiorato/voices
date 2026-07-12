#!/usr/bin/env python3
"""Generate reproducible, unseen, approximately 15-second multilingual clips."""

import subprocess
import tempfile
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "public/audio/short-unseen-tests"
SAMPLES = {
    "english-samantha": (
        "Samantha", 165,
        "On Thursday morning, the research team carried a small recorder across the quiet harbor. "
        "They compared each signal carefully, noted the changing wind, and repeated the final sentence "
        "twice so the experiment would remain clear and useful.",
    ),
    "italian-flo": (
        "Flo (Italian (Italy))", 175,
        "Giovedì mattina il gruppo di ricerca ha portato un piccolo registratore lungo il porto tranquillo. "
        "Hanno confrontato ogni segnale con attenzione, osservato il vento che cambiava e ripetuto due volte "
        "la frase finale per rendere l'esperimento chiaro e utile.",
    ),
    "spanish-grandma": (
        "Grandma (Spanish (Spain))", 175,
        "El jueves por la mañana, el equipo de investigación llevó una pequeña grabadora por el puerto tranquilo. "
        "Compararon cada señal con cuidado, observaron cómo cambiaba el viento y repitieron dos veces la frase "
        "final para que el experimento fuera claro y útil.",
    ),
    "french-jacques": (
        "Jacques", 165,
        "Jeudi matin, l'équipe de recherche a transporté un petit enregistreur le long du port tranquille. "
        "Ils ont comparé chaque signal avec attention, observé le vent changeant et répété deux fois la dernière "
        "phrase afin que l'expérience reste claire et utile.",
    ),
}


def duration(path):
    with wave.open(str(path), "rb") as audio:
        return audio.getnframes() / audio.getframerate()


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        for name, (voice, rate, text) in SAMPLES.items():
            aiff = temporary / f"{name}.aiff"
            output = OUTPUT / f"{name}.wav"
            subprocess.run(["say", "-v", voice, "-r", str(rate), "-o", str(aiff), text], check=True)
            subprocess.run([
                "ffmpeg", "-loglevel", "error", "-y", "-i", str(aiff),
                "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", str(output),
            ], check=True)
            print(f"{name}: {voice}, {duration(output):.3f}s")


if __name__ == "__main__":
    main()
