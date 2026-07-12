#!/usr/bin/env python3
"""Prepare three reusable 15-second target-voice references for VC evaluation."""

import json
import subprocess
import tempfile
import urllib.parse
import urllib.request
import wave
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "public/audio/model-comparison/references"
LJ_DATASET = "MikhailT/lj-speech"
LJ_API = "https://datasets-server.huggingface.co/rows"
CMU_BASE = "https://www.cs.cmu.edu/~dhuggins/Projects/ASR_Coding/tests_with_mfcc/rms"


def download(url, output):
    with urllib.request.urlopen(url) as response:
        output.write_bytes(response.read())


def decode(path, output):
    subprocess.run([
        "ffmpeg", "-loglevel", "error", "-y", "-i", str(path),
        "-ar", "16000", "-ac", "1", "-f", "f32le", str(output),
    ], check=True)
    return np.fromfile(output, np.float32)


def write_wav(path, audio):
    audio = np.asarray(audio[: 15 * 16000], np.float32)
    if len(audio) < 15 * 16000:
        audio = np.pad(audio, (0, 15 * 16000 - len(audio)))
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(pcm.tobytes())


def lj_speech(temporary):
    params = urllib.parse.urlencode({
        "dataset": LJ_DATASET,
        "config": "default",
        "split": "full",
        "offset": 0,
        "length": 8,
    })
    with urllib.request.urlopen(f"{LJ_API}?{params}") as response:
        rows = json.load(response)["rows"]
    audio = []
    sources = []
    for index, item in enumerate(rows):
        source = item["row"]["audio"][0]["src"]
        wav = temporary / f"lj-{index}.wav"
        download(source, wav)
        audio.append(decode(wav, temporary / f"lj-{index}.f32"))
        sources.append({"file": item["row"]["file"], "text": item["row"]["normalized_text"]})
        if sum(map(len, audio)) >= 15 * 16000:
            break
    return np.concatenate(audio), sources


def cmu_rms(temporary):
    audio = []
    sources = []
    for number in range(1, 13):
        url = f"{CMU_BASE}/{number:03}.wav"
        wav = temporary / f"rms-{number:03}.wav"
        download(url, wav)
        audio.append(decode(wav, temporary / f"rms-{number:03}.f32"))
        sources.append(url)
        if sum(map(len, audio)) >= 15 * 16000:
            break
    return np.concatenate(audio), sources


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    metadata = {
        "generatedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
        "durationSeconds": 15,
        "sampleRate": 16000,
        "targets": {},
    }
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        lj_audio, lj_sources = lj_speech(temporary)
        write_wav(OUTPUT / "lj-speech-linda-johnson.wav", lj_audio)
        metadata["targets"]["lj-speech"] = {
            "label": "LJ Speech / Linda Johnson",
            "license": "Public domain in the US; verify jurisdiction-specific status",
            "provenance": "https://keithito.com/LJ-Speech-Dataset/",
            "sourceSamples": lj_sources,
        }

        rms_audio, rms_sources = cmu_rms(temporary)
        write_wav(OUTPUT / "cmu-arctic-rms.wav", rms_audio)
        metadata["targets"]["cmu-rms"] = {
            "label": "CMU ARCTIC RMS",
            "license": "CMU ARCTIC permissive free-software data license",
            "provenance": "http://festvox.org/cmu_arctic/",
            "sourceSamples": rms_sources,
        }

        llvc_source = ROOT / "public/audio/quantization-comparison/english-fp32.wav"
        llvc_audio = decode(llvc_source, temporary / "llvc.f32")
        write_wav(OUTPUT / "llvc-fixed-target.wav", llvc_audio)
        metadata["targets"]["llvc-target"] = {
            "label": "LLVC fixed target (derived reference)",
            "license": "Derived locally from the MIT-licensed KoeAI LLVC checkpoint",
            "provenance": str(llvc_source.relative_to(ROOT)),
        }

    (OUTPUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
