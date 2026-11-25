#!/usr/bin/env python3
"""Convert the MLX build of Qwen3.8-Flash-Next (Frankie v9 Flash brain) to GGUF.

There is no bf16 copy of this brain on disk, and one would not fit beside the
quantized one, so this reads the MLX checkpoint directly:

  * 4-bit affine, group 32 -> Q4_1, bit for bit. Both store w = d*q + m per
    group of 32; only the nibble order and the d/m width (bf16 -> f16) differ.
  * 8-bit affine, group 64 -> dequantized to f32, then the stock converter
    path (renames, V-head reorder, F32 rules) writes them as F16.
  * bf16 tensors -> the stock converter path, written as BF16 (1-D as F32).
  * the n-gram sidecar -> per_layer_token_embd, Q4_1, read in row chunks.

MLX already applied two load-time rewrites, which are undone here so the stock
qwen4exp converter sees the Hugging Face layout: conv kernels are [ch, k, 1]
instead of [ch, 1, k], and the zero-centred norms carry +1.

The vision tower and the MTP head are not written.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from functools import partial
from pathlib import Path
from typing import Callable, cast

import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "gguf-py"))

import gguf  # noqa: E402
from conversion import load_all_models  # noqa: E402
from conversion.qwen4exp import Qwen4ExpTextModel  # noqa: E402

logger = logging.getLogger("convert-flash-mlx")

# MLX sanitize() adds 1 to these when it converts from Hugging Face (mtplx qwen4_exp.py)
MLX_NORM_SHIFT = (".q_norm.weight", ".k_norm.weight", ".q_layernorm.weight", ".k_layernorm.weight",
                  ".hc_norm.weight", ".norm_key.weight", ".norm_query.weight", ".norm_conv.weight")

Q4_1 = gguf.GGMLQuantizationType.Q4_1


def to_numpy(t: torch.Tensor) -> np.ndarray:
    if t.dtype == torch.uint32:
        return t.view(torch.int32).numpy().view(np.uint32)
    if t.dtype == torch.bfloat16:
        return t.float().numpy()
    return t.numpy()


def q4_1_blocks(w: np.ndarray, scales: np.ndarray, biases: np.ndarray) -> np.ndarray:
    """MLX 4-bit gs32 rows -> ggml block_q4_1 rows: uint8, 20 bytes (f16 d, f16 m, 16 code bytes) per 32."""
    packed = np.ascontiguousarray(w).view(np.uint8)          # element 2i low nibble, 2i+1 high
    codes = np.stack((packed & 0x0F, packed >> 4), axis=-1).reshape(*packed.shape[:-1], -1)
    n_groups = scales.shape[-1]
    codes = codes.reshape(*codes.shape[:-1], n_groups, 32)
    qs = codes[..., :16] | (codes[..., 16:] << 4)               # ggml: j low, j+16 high
    d = scales.astype(np.float16)[..., None].view(np.uint8)
    m = biases.astype(np.float16)[..., None].view(np.uint8)
    blocks = np.concatenate((d, m, qs), axis=-1)                # block_q4_1 {d, m, qs[16]}
    return np.ascontiguousarray(blocks.reshape(*blocks.shape[:-2], n_groups * 20))


def mlx_dequant(w: np.ndarray, scales: np.ndarray, biases: np.ndarray, bits: int, group: int) -> np.ndarray:
    packed = np.ascontiguousarray(w).view(np.uint8)
    if bits == 8:
        codes = packed
    elif bits == 4:
        codes = np.stack((packed & 0x0F, packed >> 4), axis=-1).reshape(*packed.shape[:-1], -1)
    else:
        raise ValueError(f"{bits}-bit MLX weights are not handled")
    codes = codes.reshape(*codes.shape[:-1], scales.shape[-1], group).astype(np.float32)
    out = codes * scales[..., None] + biases[..., None]
    return out.reshape(*out.shape[:-2], -1)


class Checkpoint:
    """Name -> shard index over the MLX safetensors, opened on demand."""

    def __init__(self, files: dict[str, Path]):
        self.files = files
        self._open: dict[Path, object] = {}

    def handle(self, name: str):
        path = self.files[name]
        if path not in self._open:
            self._open[path] = safe_open(str(path), framework="pt")
        return self._open[path]

    def get(self, name: str) -> torch.Tensor:
        return self.handle(name).get_tensor(name)

    def rows(self, name: str, a: int, b: int) -> np.ndarray:
        return to_numpy(self.handle(name).get_slice(name)[a:b])

    def shape(self, name: str) -> list[int]:
        return list(self.handle(name).get_slice(name).get_shape())


class FlashMLXModel(Qwen4ExpTextModel):
    model_arch = gguf.MODEL_ARCH.QWEN4EXP

    def __init__(self, *args, mlx_quant: dict, **kwargs):
        self.mlx_quant = mlx_quant
        self.bf16_sources: set[str] = set()
        super().__init__(*args, **kwargs)

    # -- tensor source ------------------------------------------------------

    def index_tensors(self, remote_hf_model_id=None):
        weight_map = json.loads((self.dir_model / "model.safetensors.index.json").read_text())["weight_map"]
        files = {k: self.dir_model / f for k, f in weight_map.items()}
        sidecar = self.dir_model / "ngram-table.safetensors"
        if sidecar.is_file():
            with safe_open(str(sidecar), framework="pt") as f:
                for k in f.keys():
                    files[k] = sidecar
        self.ckpt = Checkpoint(files)

        self.q4: dict[str, str] = {}          # hf name -> mlx base name
        tensors = {}
        for key in files:
            if key.startswith(("vision_tower.", "mtp.")) or key.startswith("ngram."):
                continue
            if key.endswith((".scales", ".biases")):
                continue
            base = key[:-len(".weight")] if key.endswith(".weight") else None
            name = key.replace("language_model.", "", 1)
            if base is not None and f"{base}.scales" in files:
                bits, group = self.quant_of(base)
                if (bits, group) == (4, 32):
                    self.q4[name] = base
                    continue
                if bits != 8:
                    raise ValueError(f"{key}: {bits}-bit/{group} is not handled")
                tensors[name] = lambda base=base, g=group: torch.from_numpy(mlx_dequant(
                    to_numpy(self.ckpt.get(f"{base}.weight")), to_numpy(self.ckpt.get(f"{base}.scales")),
                    to_numpy(self.ckpt.get(f"{base}.biases")), 8, g))
                continue
            name, load = self.hf_float(key, name)
            tensors[name] = load
        return tensors

    def quant_of(self, base: str) -> tuple[int, int]:
        q = self.mlx_quant.get(base)
        if isinstance(q, dict):
            return int(q["bits"]), int(q["group_size"])
        return int(self.mlx_quant["bits"]), int(self.mlx_quant["group_size"])

    def hf_float(self, key: str, name: str):
        """Undo the MLX load-time rewrites so the stock converter sees the HF tensor."""
        get = lambda: self.ckpt.get(key)  # noqa: E731
        if name.endswith("ple.conv_weight"):
            name = name.replace("ple.conv_weight", "ple.conv1d.weight")
            load = lambda: get().movedim(1, 2)  # noqa: E731  [ch, k, 1] -> [ch, 1, k]
        elif name.endswith("linear_attn.conv1d.weight"):
            load = lambda: get().movedim(1, 2)  # noqa: E731
        elif name.endswith(MLX_NORM_SHIFT):
            load = lambda: get().float() - 1.0  # noqa: E731  exact: bf16 near 1 is f32-exact minus 1
        else:
            load = get
        if self.ckpt.handle(key).get_slice(key).get_dtype() == "BF16":
            self.bf16_sources.add(name)
        return name, load

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        # the conv kernels feed f32-only ops (ssm_conv on Metal)
        if any(self.match_model_tensor_name(new_name, key, bid)
               for key in (gguf.MODEL_TENSOR.SSM_CONV1D, gguf.MODEL_TENSOR.PLE_CONV1D)):
            return gguf.GGMLQuantizationType.F32
        if name in self.bf16_sources and n_dims >= 2:
            return gguf.GGMLQuantizationType.BF16
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    # -- 4-bit tensors, repacked rather than requantized ---------------------

    def prepare_tensors(self):
        super().prepare_tensors()
        for name, base in self.q4.items():
            self.add_q4(name, base)
        if "ngram.weight" in self.ckpt.files:
            self.add_ple_table()

    def row_map(self, name: str, shape: list[int]) -> list[tuple[str, np.ndarray | None]]:
        """Run the stock rename/reorder on a row-index carrier; None means rows unchanged."""
        bid = next((int(p) for p in name.split(".") if p.isdecimal()), None)
        if ".switch_mlp." in name:
            proj = name.rsplit(".switch_mlp.", 1)[1].split(".")[0]
            key = {"gate_proj": gguf.MODEL_TENSOR.FFN_GATE_EXP, "up_proj": gguf.MODEL_TENSOR.FFN_UP_EXP,
                   "down_proj": gguf.MODEL_TENSOR.FFN_DOWN_EXP}[proj]
            return [(self.format_tensor_name(key, bid), None)]
        rows = shape[0]
        carrier = torch.arange(rows, dtype=torch.float64).reshape(rows, 1)
        out = []
        for new_name, t in self.modify_tensors(carrier, name, bid):
            if t.ndim != 2 or t.shape[1] != 1:
                raise ValueError(f"{name}: the stock transform touched columns, which 4-bit repacking cannot follow")
            idx = t[:, 0].numpy().astype(np.int64)
            out.append((new_name, None if np.array_equal(idx, np.arange(rows)) else idx))
        return out

    def add_q4(self, name: str, base: str):
        w_shape = self.ckpt.shape(f"{base}.weight")
        shape = w_shape[:-1] + [w_shape[-1] * 8]
        for new_name, idx in self.row_map(name, shape):
            self.verify_q4(base)
            chunks: list[Callable[[], np.ndarray]]
            if idx is None:
                n = shape[0]
                step = max(1, (1 << 28) // int(np.prod(shape[1:])))   # ~256M weights per chunk
                chunks = [partial(self.q4_rows, base, a, min(n, a + step)) for a in range(0, n, step)]
                out_rows = n
            else:
                chunks = [partial(self.q4_picked, base, idx)]
                out_rows = len(idx)
            out_shape = [out_rows] + shape[1:]
            byte_shape = tuple(gguf.quant_shape_to_byte_shape(tuple(out_shape), Q4_1))
            self.gguf_writer.add_tensor(new_name, cast(np.ndarray, gguf.LazyChunkedTensor(chunks, byte_shape, np.uint8)),
                                        raw_shape=byte_shape, raw_dtype=Q4_1)
            logger.info(f"{new_name:<40} MLX 4-bit --> Q4_1, shape = {{{', '.join(map(str, reversed(out_shape)))}}}"
                        + ("" if idx is None else " (rows reordered)"))

    def q4_rows(self, base: str, a: int, b: int) -> np.ndarray:
        return q4_1_blocks(self.ckpt.rows(f"{base}.weight", a, b), self.ckpt.rows(f"{base}.scales", a, b),
                           self.ckpt.rows(f"{base}.biases", a, b))

    def q4_picked(self, base: str, idx: np.ndarray) -> np.ndarray:
        return q4_1_blocks(to_numpy(self.ckpt.get(f"{base}.weight"))[idx], to_numpy(self.ckpt.get(f"{base}.scales"))[idx],
                           to_numpy(self.ckpt.get(f"{base}.biases"))[idx])

    def verify_q4(self, base: str):
        """The repack is exact up to bf16->f16 on d and m; prove it on the first rows."""
        n = min(64, self.ckpt.shape(f"{base}.weight")[0])
        w, s, b = (self.ckpt.rows(f"{base}.{p}", 0, n) for p in ("weight", "scales", "biases"))
        ref = mlx_dequant(w, s.astype(np.float16).astype(np.float32), b.astype(np.float16).astype(np.float32), 4, 32)
        got = gguf.quants.dequantize(q4_1_blocks(w, s, b), Q4_1).reshape(ref.shape)
        if not np.array_equal(got.astype(np.float32), ref):
            err = float(np.max(np.abs(got - ref)))
            if err > 1e-6:
                raise ValueError(f"{base}: Q4_1 repack differs from MLX dequant by {err}")
        lossy = int(np.sum(s.astype(np.float16).astype(np.float32) != s) + np.sum(b.astype(np.float16).astype(np.float32) != b))
        if lossy:
            logger.warning(f"{base}: {lossy} d/m values not exact in f16 (first {n} rows)")

    def add_ple_table(self):
        rows, words = self.ckpt.shape("ngram.weight")
        dim = words * 8
        meta = safe_open(str(self.ckpt.files["ngram.weight"]), framework="pt").metadata() or {}
        if (int(meta.get("ngram_bits", 4)), int(meta.get("ngram_group_size", 32))) != (4, 32):
            raise ValueError(f"n-gram table is {meta}, only 4-bit group 32 is handled")
        self.verify_q4("ngram")
        step = 1 << 21
        chunks: list[Callable[[], np.ndarray]] = [partial(self.q4_rows, "ngram", a, min(rows, a + step)) for a in range(0, rows, step)]
        byte_shape = tuple(gguf.quant_shape_to_byte_shape((rows, dim), Q4_1))
        name = gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.PER_LAYER_TOKEN_EMBD] + ".weight"
        self.gguf_writer.add_tensor(name, cast(np.ndarray, gguf.LazyChunkedTensor(chunks, byte_shape, np.uint8)),
                                    raw_shape=byte_shape, raw_dtype=Q4_1)
        self.gguf_writer.add_embedding_length_per_layer_input(dim)
        logger.info(f"{name:<40} MLX 4-bit --> Q4_1, shape = {{{dim}, {rows}}}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", type=Path, help="MLX checkpoint directory")
    ap.add_argument("--outfile", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true", help="plan every tensor, write nothing")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)

    load_all_models()
    config = json.loads((args.model / "config.json").read_text())
    model = FlashMLXModel(args.model, gguf.LlamaFileType.MOSTLY_F16, args.outfile,
                          eager=True, dry_run=args.dry_run, mlx_quant=config["quantization"])
    model.write()


if __name__ == "__main__":
    main()
