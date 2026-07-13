#!/usr/bin/env python3
"""Create a calibrated Q8-High LLVC model from an arbitrary compatible ONNX export."""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)


STATE_INPUTS = ("enc_state", "dec_state", "out_state", "conv_state")


def concrete_shape(value):
    if not all(isinstance(item, int) and item > 0 for item in value):
        raise ValueError(f"Expected static model shape, got {value}")
    return tuple(value)


def decode(path, output, sample_rate):
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(path),
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            "-f",
            "f32le",
            str(output),
        ],
        check=True,
    )
    return np.fromfile(output, dtype=np.float32)


def initial_states(session):
    inputs = {item.name: item for item in session.get_inputs()}
    return {
        name: np.zeros(concrete_shape(inputs[name].shape), np.float32)
        for name in STATE_INPUTS
    }


def calibration_rows(source, samples, sample_rate, rows_per_clip):
    teacher = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"])
    input_metadata = {item.name: item for item in teacher.get_inputs()}
    output_metadata = {item.name: item for item in teacher.get_outputs()}
    input_samples = concrete_shape(input_metadata["audio"].shape)[-1]
    output_samples = concrete_shape(output_metadata["converted"].shape)[-1]
    context_samples = input_samples - output_samples
    if context_samples < 0:
        raise ValueError("LLVC output cannot be longer than its streaming input")
    output_names = [item.name for item in teacher.get_outputs()]
    rows = []
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        for sample_index, sample in enumerate(samples):
            audio = decode(sample, temporary / f"{sample_index}.f32", sample_rate)
            chunk_count = max(1, (len(audio) - output_samples) // output_samples)
            selected = set(
                np.linspace(
                    0, chunk_count - 1, min(rows_per_clip, chunk_count), dtype=int
                ).tolist()
            )
            state = initial_states(teacher)
            context = np.zeros(context_samples, np.float32)
            for chunk_index, offset in enumerate(
                range(0, len(audio) - output_samples, output_samples)
            ):
                current = audio[offset : offset + output_samples]
                model_input = (
                    np.concatenate([context, current])
                    .reshape(1, 1, input_samples)
                    .astype(np.float32)
                )
                feeds = {"audio": model_input, **state}
                if chunk_index in selected:
                    rows.append({name: value.copy() for name, value in feeds.items()})
                values = teacher.run(None, feeds)
                result = dict(zip(output_names, values))
                state = {name: result[f"{name}_next"] for name in STATE_INPUTS}
                if context_samples:
                    context = current[-context_samples:]
    return rows


class Reader(CalibrationDataReader):
    def __init__(self, rows):
        self.rows = rows
        self.rewind()

    def get_next(self):
        return next(self.iterator, None)

    def rewind(self):
        self.iterator = iter(self.rows)


def dominant_nodes(source):
    model = onnx.load(str(source), load_external_data=True)
    initializers = {value.name: value for value in model.graph.initializer}
    selected = []
    for node in model.graph.node:
        if node.op_type != "Conv":
            continue
        weights = next(
            (initializers[name] for name in node.input if name in initializers), None
        )
        if weights is not None and list(weights.dims) == [512, 512, 1]:
            selected.append(node.name)
    if not selected:
        raise ValueError("No dominant 512x512 pointwise LLVC convolutions were found")
    return selected


def collect_audio(paths):
    found = []
    for path in paths:
        path = path.resolve()
        if path.is_dir():
            originals = sorted(path.rglob("*_original.wav"))
            found.extend(originals or sorted(path.rglob("*.wav")))
        else:
            found.append(path)
    unique = list(dict.fromkeys(found))
    if not unique:
        raise ValueError("No calibration WAV files found")
    return unique


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, action="append", required=True)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--rows-per-clip", type=int, default=16)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    samples = collect_audio(args.calibration)
    rows = calibration_rows(args.source, samples, args.sample_rate, args.rows_per_clip)
    nodes = dominant_nodes(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        str(args.source),
        str(args.output),
        Reader(rows),
        quant_format=QuantFormat.QOperator,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QUInt8,
        per_channel=True,
        nodes_to_quantize=nodes,
        op_types_to_quantize=["Conv"],
    )
    test_session = ort.InferenceSession(
        str(args.output), providers=["CPUExecutionProvider"]
    )
    reference_session = ort.InferenceSession(
        str(args.source), providers=["CPUExecutionProvider"]
    )
    correlations = []
    maes = []
    finite = True
    for feeds in rows[:16]:
        reference = reference_session.run(None, feeds)[0].reshape(-1)
        candidate_outputs = test_session.run(None, feeds)
        candidate = candidate_outputs[0].reshape(-1)
        finite = finite and all(np.isfinite(value).all() for value in candidate_outputs)
        correlations.append(float(np.corrcoef(reference, candidate)[0, 1]))
        maes.append(float(np.mean(np.abs(reference - candidate))))
    if not finite:
        raise RuntimeError("Quantized model smoke test produced non-finite output")
    source_bytes = args.source.stat().st_size
    external_source = Path(f"{args.source}.data")
    if external_source.exists():
        source_bytes += external_source.stat().st_size
    result = {
        "source": str(args.source),
        "output": str(args.output),
        "sourceBytes": source_bytes,
        "outputBytes": args.output.stat().st_size,
        "calibrationFiles": len(samples),
        "calibrationRows": len(rows),
        "quantizedNodes": len(nodes),
        "finite": finite,
        "fp32Agreement": {
            "rows": min(16, len(rows)),
            "meanCorrelation": float(np.mean(correlations)),
            "meanAbsoluteError": float(np.mean(maes)),
        },
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
