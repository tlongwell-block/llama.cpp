#!/usr/bin/env python3
"""Development conversion of Frankie's MLX brain to native Qwen3.5 GGUF.

This is a brain component, not a complete Frankie model. Q4_1 is a deliberate
re-quantization; validate output before treating it as reference parity.
"""
import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "gguf-py"))
import gguf
from conversion.qwen import Qwen3_5TextModel


def dequant_affine4(weight, scales, biases):
    if weight.dtype != torch.uint32 or weight.ndim != 2:
        raise ValueError("expected packed two-dimensional uint32 MLX weights")
    if scales.shape != biases.shape or scales.shape != (weight.shape[0], weight.shape[1] // 8):
        raise ValueError("expected MLX affine4 group_size=64")
    packed = weight.numpy()
    output = np.empty((packed.shape[0], packed.shape[1] * 8), dtype=np.float32)
    # Bound temporary expansion even for the vocabulary embedding matrix.
    for start in range(0, packed.shape[0], 128):
        end = min(start + 128, packed.shape[0])
        q = ((packed[start:end, :, None] >> np.arange(0, 32, 4, dtype=np.uint32)) & 15).reshape(end-start, -1, 64)
        output[start:end] = (q.astype(np.float32) * scales[start:end].float().numpy()[..., None]
                            + biases[start:end].float().numpy()[..., None]).reshape(end-start, -1)
    return torch.from_numpy(output)


class FrankieBrain(Qwen3_5TextModel):
    model_arch = gguf.MODEL_ARCH.QWEN35
    no_mtp = True
    mtp_model = None
    mtp_type = gguf.GGMLQuantizationType.Q8_0

    def index_tensors(self, remote_hf_model_id=None):
        index = json.loads((self.dir_model / "model.safetensors.index.json").read_text())["weight_map"]
        self.source_index = index
        tensors = {k.removeprefix("language_model."): None for k in index if k.startswith("language_model.")}
        self.mtp_sources = {}
        if self.mtp_model:
            config = {**self.hparams, **self.hparams.get("text_config", {})}
            head = json.loads((self.mtp_model / "config.json").read_text())
            head = head.get("text_config", head)
            for key in ("hidden_size", "num_hidden_layers", "intermediate_size", "vocab_size",
                        "num_attention_heads", "num_key_value_heads", "head_dim", "rms_norm_eps", "rope_parameters"):
                if config.get(key) != head.get(key):
                    raise ValueError(f"MTP source configuration differs: {key}")
            if head.get("mtp_num_hidden_layers") != 1 or head.get("mtp_use_dedicated_embeddings", False):
                raise ValueError("expected one Qwen MTP layer sharing the brain embedding and output head")
            type(self)._original_block_count = config["num_hidden_layers"]
            for path in sorted(self.mtp_model.glob("*.safetensors")):
                with safe_open(path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        if not key.startswith(("mtp.", "model.mtp.")):
                            continue
                        name, _ = self.filter_tensors((key, None))
                        if name in tensors:
                            raise ValueError(f"duplicate MTP tensor: {name}")
                        self.mtp_sources[name] = (path, key)
                        tensors[name] = None
            if len(self.mtp_sources) != 15:
                raise ValueError("expected all 15 tensors from the matching unquantized Qwen MTP head")
        return tensors

    def tensor(self, name):
        if name in self.mtp_sources:
            path, key = self.mtp_sources[name]
            with safe_open(path, framework="pt", device="cpu") as f:
                return f.get_tensor(key)
        key = "language_model." + name
        with safe_open(self.dir_model / self.source_index[key], framework="pt", device="cpu") as f:
            return f.get_tensor(key)

    def prepare_tensors(self):
        quant = self.hparams.get("quantization", self.hparams.get("quantization_config"))
        if quant != {"group_size": 64, "bits": 4, "mode": "affine"}:
            raise ValueError(f"unsupported source quantization: {quant}")
        for name in self.model_tensors:
            if name.endswith((".scales", ".biases")):
                continue
            data = self.tensor(name)
            quantized = data.dtype == torch.uint32
            if quantized:
                base = name.removesuffix(".weight")
                data = dequant_affine4(data, self.tensor(base + ".scales"), self.tensor(base + ".biases"))
            else:
                data = data.float()
                # This MLX checkpoint is already sanitized to direct RMS weights.
                # The inherited HF converter adds one, so undo that convention first.
                if name not in self.mtp_sources and name.endswith("norm.weight") and not name.endswith("linear_attn.norm.weight"):
                    data = data - 1
            match = re.search(r"\.layers\.(\d+)\.", name)
            bid = int(match.group(1)) if match else None
            for mapped, transformed in self.modify_tensors(data, name, bid):
                array = transformed.contiguous().numpy()
                qtype = gguf.GGMLQuantizationType.Q4_1 if quantized else gguf.GGMLQuantizationType.F32
                if name in self.mtp_sources and array.ndim == 2:
                    qtype = self.mtp_type
                raw = gguf.quants.quantize(array, qtype)
                self.gguf_writer.add_tensor(mapped, raw, raw_dtype=qtype)
                logging.info("%s -> %s %s %s", name, mapped, array.shape, qtype.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mtp-model", type=Path, help="matching HF config and MTP safetensors")
    parser.add_argument("--mtp-type", choices=["q8_0", "q4_1"], default="q8_0", help="draft-head matrix quantization")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    logging.basicConfig(level=logging.INFO)
    torch.set_num_threads(2)
    config = json.loads((args.model / "config.json").read_text())
    FrankieBrain.mtp_model = args.mtp_model
    FrankieBrain.no_mtp = args.mtp_model is None
    FrankieBrain.mtp_type = getattr(gguf.GGMLQuantizationType, args.mtp_type.upper())
    model = FrankieBrain(args.model, gguf.LlamaFileType.MOSTLY_Q4_1, args.output,
                         hparams=config, use_temp_file=True, model_name="Frankie brain component")
    model.write()


if __name__ == "__main__":
    main()
