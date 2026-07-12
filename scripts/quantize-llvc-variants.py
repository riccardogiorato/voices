#!/usr/bin/env python3
"""Create no-training precision variants of the full LLVC teacher."""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxconverter_common import float16
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "public/models"
SOURCE = MODEL_DIR / "common-voice-llvc.onnx"
SAMPLES = [
    *sorted((ROOT / "public/audio").glob("*.wav")),
    *sorted((ROOT / "public/audio/long-tests").glob("*.wav")),
]


def initial_states():
    return {
        "enc_state": np.zeros((1, 512, 510), np.float32),
        "dec_state": np.zeros((1, 2, 13, 256), np.float32),
        "out_state": np.zeros((1, 512, 4), np.float32),
        "conv_state": np.zeros((1, 1, 24), np.float32),
    }


def decode(path, output):
    subprocess.run([
        "ffmpeg", "-loglevel", "error", "-y", "-i", str(path),
        "-ar", "16000", "-ac", "1", "-f", "f32le", str(output),
    ], check=True)
    return np.fromfile(output, dtype=np.float32)


def calibration_rows(rows_per_clip=16):
    teacher = ort.InferenceSession(str(SOURCE), providers=["CPUExecutionProvider"])
    output_names = [output.name for output in teacher.get_outputs()]
    rows = []
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        for sample_index, sample in enumerate(SAMPLES):
            audio = decode(sample, temporary / f"{sample_index}.f32")
            chunk_count = max(1, (len(audio) - 208) // 208)
            selected = set(np.linspace(0, chunk_count - 1, rows_per_clip, dtype=int).tolist())
            state = initial_states()
            context = np.zeros(32, np.float32)
            for chunk_index, offset in enumerate(range(0, len(audio) - 208, 208)):
                samples = audio[offset:offset + 208]
                model_input = np.concatenate([context, samples]).reshape(1, 1, 240).astype(np.float32)
                feeds = {"audio": model_input, **state}
                if chunk_index in selected:
                    rows.append({name: value.copy() for name, value in feeds.items()})
                values = teacher.run(None, feeds)
                result = dict(zip(output_names, values))
                state = {
                    "enc_state": result["enc_state_next"],
                    "dec_state": result["dec_state_next"],
                    "out_state": result["out_state_next"],
                    "conv_state": result["conv_state_next"],
                }
                context = samples[-32:]
    return rows


class Reader(CalibrationDataReader):
    def __init__(self, rows):
        self.rows = rows
        self.rewind()

    def get_next(self):
        return next(self.iterator, None)

    def rewind(self):
        self.iterator = iter(self.rows)


def node_sets():
    model = onnx.load(str(SOURCE), load_external_data=True)
    initializers = {value.name: value for value in model.graph.initializer}
    dominant = []
    linear = []
    all_supported = []
    for node in model.graph.node:
        if node.op_type in {"Conv", "MatMul", "Gemm"}:
            all_supported.append(node.name)
        if node.op_type in {"MatMul", "Gemm"}:
            linear.append(node.name)
        if node.op_type == "Conv":
            weights = next((initializers[name] for name in node.input if name in initializers), None)
            if weights is not None and list(weights.dims) == [512, 512, 1]:
                dominant.append(node.name)
    return {
        "q8-high": dominant,
        "q8-balanced": dominant + linear,
        "q8-max": all_supported,
    }


def save_fp16():
    model = onnx.load(str(SOURCE), load_external_data=True)
    converted = float16.convert_float_to_float16(
        model,
        keep_io_types=True,
        op_block_list=["LayerNormalization", "Softmax", "ReduceMean"],
        min_positive_val=1e-7,
        max_finite_val=1e4,
    )
    onnx.save(converted, MODEL_DIR / "common-voice-llvc-fp16.onnx")


def save_q8(name, nodes, rows):
    quantize_static(
        str(SOURCE),
        str(MODEL_DIR / f"common-voice-llvc-{name}.onnx"),
        Reader(rows),
        quant_format=QuantFormat.QOperator,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QUInt8,
        per_channel=True,
        nodes_to_quantize=nodes,
        op_types_to_quantize=["Conv", "MatMul", "Gemm"],
    )


def main():
    save_fp16()
    rows = calibration_rows()
    selections = node_sets()
    for name, nodes in selections.items():
        save_q8(name, nodes, rows)
    result = {}
    for name in ("fp16", "q8-high", "q8-balanced", "q8-max"):
        path = MODEL_DIR / f"common-voice-llvc-{name}.onnx"
        result[name] = {"bytes": path.stat().st_size, "megabytes": path.stat().st_size / 1_000_000}
    print(json.dumps({"calibrationRows": len(rows), "models": result}, indent=2))


if __name__ == "__main__":
    main()
