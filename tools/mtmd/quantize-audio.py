#!/usr/bin/env python3
"""Quantize supported Parakeet and Qwen3-TTS mmproj matrices with gguf-py's Q8_0 implementation."""
import argparse
import collections
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))
import gguf


def eligible(tensor, model):
    if tensor.tensor_type not in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16):
        return False
    if not tensor.name.endswith(".weight") or tensor.shape[0] % 32:
        return False
    if model == "parakeet":
        # Depthwise SSM convolution requires F32. Other spatial convolutions retain their original types.
        return len(tensor.shape) == 2 and ".conv_dw." not in tensor.name and "norm" not in tensor.name
    # Keep waveform convolution, speaker convolution, codebooks and normalization in floating point.
    return tensor.name.startswith("a.gen.") and (len(tensor.shape) == 2 or tensor.name in (
        "a.gen.code.head.weight", "a.gen.code.embd.weight"))


def convert(source: Path, output: Path, model: str):
    if output.exists():
        raise ValueError("output already exists")
    reader = gguf.GGUFReader(str(source))
    if reader.get_field("general.architecture").contents() != "clip":
        raise ValueError("use llama-quantize for the talker backbone; this converter accepts mmproj GGUFs")
    profile_key, profile_value = ("clip.projector_type", "parakeet") if model == "parakeet" else ("clip.gen.audio.projector_type", "qwen3tts_gen")
    profile = reader.get_field(profile_key)
    if profile is None or profile.contents() != profile_value:
        raise ValueError(f"GGUF does not match the requested {model} quantization profile")
    writer = gguf.GGUFWriter(str(output), "clip")
    for field in reader.fields.values():
        if field.name.startswith("GGUF.") or field.name in ("general.architecture", "general.file_type", "general.quantization_version"):
            continue
        writer.add_key_value(field.name, field.contents(), field.types[0], field.types[-1] if len(field.types) > 1 else None)
    writer.add_file_type(gguf.LlamaFileType.MOSTLY_Q8_0)
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
    inventory = []
    for tensor in reader.tensors:
        data, dtype = tensor.data, tensor.tensor_type
        if eligible(tensor, model):
            if not np.isfinite(data).all():
                raise ValueError("non-finite tensor: " + tensor.name)
            dtype = gguf.GGMLQuantizationType.Q8_0
            data = gguf.quantize(np.asarray(data, dtype=np.float32), dtype)
        writer.add_tensor(tensor.name, data, raw_dtype=dtype)
        inventory.append(dict(name=tensor.name, before=tensor.tensor_type.name, after=dtype.name, bytes=data.nbytes))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    check = gguf.GGUFReader(str(output))
    if len(check.tensors) != len(reader.tensors):
        raise ValueError("tensor inventory changed")
    for original, actual in zip(reader.tensors, check.tensors):
        if original.name != actual.name or not np.array_equal(original.shape, actual.shape):
            raise ValueError("tensor shape changed")
        if not eligible(original, model) and not np.array_equal(original.data, actual.data):
            raise ValueError("unquantized tensor changed")
    report = dict(source=str(source), output=str(output), source_bytes=source.stat().st_size,
                  output_bytes=output.stat().st_size, types=dict(collections.Counter(t.tensor_type.name for t in check.tensors)),
                  tensors=inventory)
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "tensors"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", required=True, choices=["parakeet", "qwen3tts"])
    args = parser.parse_args()
    convert(args.source, args.output, args.model)
