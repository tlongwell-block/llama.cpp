#!/usr/bin/env python3
"""Convert Frankie's MLX-layout vision component with the upstream Qwen3VL mapper."""
import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "gguf-py"))
import gguf
from conversion.qwen3vl import Qwen3VLVisionModel


class FrankieVision(Qwen3VLVisionModel):
    model_arch = gguf.MODEL_ARCH.MMPROJ

    def index_tensors(self, remote_hf_model_id=None):
        self.source_index = json.loads((self.dir_model / "model.safetensors.index.json").read_text())["weight_map"]
        return {k: None for k in self.source_index if k.startswith("vision_tower.")}

    def get_tensors(self):
        for key in self.model_tensors:
            with safe_open(self.dir_model / self.source_index[key], framework="pt", device="cpu") as f:
                data = f.get_tensor(key)
            if data.dtype not in (torch.bfloat16, torch.float16, torch.float32):
                raise ValueError("unsupported vision tensor type: " + key)
            name = key.replace("vision_tower.", "visual.", 1)
            if name == "visual.patch_embed.proj.weight":
                data = data.permute(0, 4, 1, 2, 3).contiguous()
            yield name, data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    logging.basicConfig(level=logging.INFO)
    torch.set_num_threads(2)
    model = FrankieVision(args.model, gguf.LlamaFileType.MOSTLY_F16, args.output,
                          hparams=json.loads((args.model / "config.json").read_text()),
                          use_temp_file=True, model_name="Frankie vision component")
    model.write()


if __name__ == "__main__":
    main()
