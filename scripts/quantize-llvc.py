#!/usr/bin/env python3
"""Convert LLVC ONNX model to FP16 and INT8 quantized versions for faster browser inference."""
import argparse
import numpy as np
import onnx
from onnxconverter_common import float16
from pathlib import Path

def to_fp16(input_path, output_path):
    """Convert model weights to FP16. Reduces model size and GPU memory bandwidth by ~50%."""
    model = onnx.load(input_path)
    # Keep certain ops in FP32 for numerical stability
    fp16_model = float16.convert_float_to_float16(
        model,
        keep_io_types=True,  # Keep inputs/outputs as FP32 for compatibility
        op_block_list=['LayerNormalization', 'Softmax', 'ReduceMean'],  # These ops need FP32
        min_positive_val=1e-7,
        max_finite_val=1e4,
    )
    onnx.save(fp16_model, output_path)
    print(f"FP16 model saved to {output_path}")
    print(f"  Original size: {Path(input_path).stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  FP16 size:     {Path(output_path).stat().st_size / 1024 / 1024:.1f} MB")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("public/models/common-voice-llvc.onnx"))
    parser.add_argument("--output", type=Path, default=Path("public/models/common-voice-llvc-fp16.onnx"))
    args = parser.parse_args()
    to_fp16(args.input, args.output)

if __name__ == "__main__":
    main()
