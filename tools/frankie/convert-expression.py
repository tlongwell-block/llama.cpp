#!/usr/bin/env python3
"""Convert a Frankie linear expression head and Qwen speaker task vectors to GGUF."""
import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))
import gguf


def convert(head_path: Path, vectors_path: Path, output: Path):
    with np.load(head_path, allow_pickle=False) as head, np.load(vectors_path, allow_pickle=False) as vectors:
        classes = ["angry", "happy", "sad", "neutral"]
        if head["classes"].tolist() != classes:
            raise ValueError("expression class order must be angry, happy, sad, neutral")
        weights = {key: np.asarray(head[key], dtype=np.float32) for key in ("weight", "bias", "mu", "sd")}
        shapes = {"weight": (4, 5120), "bias": (4,), "mu": (5120,), "sd": (5120,)}
        for key, shape in shapes.items():
            if weights[key].shape != shape or not np.isfinite(weights[key]).all():
                raise ValueError(f"invalid expression {key}")
        if not (weights["sd"] > 0).all():
            raise ValueError("expression sd must be positive")
        directions = [np.asarray(vectors["tau__" + key], dtype=np.float32) * cap
                      for key, cap in zip(classes[:3], [1.0, 3.0, 3.0])]
        if any(row.shape != (2048,) or not np.isfinite(row).all() for row in directions):
            raise ValueError("invalid speaker task vector")
        weights["directions"] = np.stack(directions + [np.zeros(2048, dtype=np.float32)], axis=1)
        writer = gguf.GGUFWriter(str(output), "frankie-expression")
        writer.add_string("expression.classes", ",".join(classes))
        for key, value in weights.items():
            writer.add_tensor("expression." + key, np.ascontiguousarray(value))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("head", type=Path)
    parser.add_argument("vectors", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    convert(args.head, args.vectors, args.output)
