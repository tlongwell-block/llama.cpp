#!/usr/bin/env python3
"""Convert Frankie brain-led delivery assets to GGUF.

The folder holds directions.npz (16 feelings) and words.npz (24 words) read
from the brain's final state, calib-feelings.npz and calib-words.npz (length
calibration), adapter.npz (readings to row weights and guidance strength),
optionally flash-scale.npz, and rows/NAME.slot.safetensors for each bank name.
"""
import argparse
import json
from pathlib import Path
import struct
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))
import gguf


def slot_rows(path: Path):
    data = path.read_bytes()
    size = struct.unpack("<Q", data[:8])[0]
    header = json.loads(data[8:8 + size])
    theta = header["theta"]
    if theta["dtype"] != "F32" or theta["shape"] != [1, 16, 2048]:
        raise ValueError(f"invalid row set: {path.name}")
    begin, end = theta["data_offsets"]
    return np.frombuffer(data[8 + size + begin:8 + size + end], dtype="<f4").reshape(16, 2048)


def convert(folder: Path, output: Path):
    load = lambda name: dict(np.load(folder / name, allow_pickle=False))
    feelings, words = load("directions.npz"), load("words.npz")
    adapter = load("adapter.npz")
    width = feelings["mu"].shape[-1]
    if words["mu"].shape[-1] != width or width not in (5120, 10240):
        raise ValueError("feeling and word directions must read one brain width")
    bank = [str(x) for x in adapter["bank"]]
    tensors = {}
    for prefix, d, cal, n in (("feelings", feelings, load("calib-feelings.npz"), 16),
                              ("words", words, load("calib-words.npz"), 24)):
        if d["V"].shape != (n, width) or d["center"].shape != (n,) or d["spread"].shape != (n,):
            raise ValueError(prefix + " directions shape")
        tensors[prefix + ".V"] = np.asarray(d["V"], np.float32)
        tensors[prefix + ".mu"] = np.asarray(d["mu"], np.float32)
        tensors[prefix + ".center"] = np.asarray(d["center"], np.float32)
        tensors[prefix + ".spread"] = np.asarray(d["spread"], np.float32)
        for key in "abce":
            if cal[key].shape != (n,):
                raise ValueError(prefix + " calibration shape")
            tensors[prefix + ".cal_" + key] = np.asarray(cal[key], np.float64)
        tensors[prefix + ".cal_ref"] = np.asarray([cal["ref"]], np.float64)
    if (folder / "flash-scale.npz").exists():
        scale = load("flash-scale.npz")
        for key, n in (("fa", 16), ("fb", 16), ("fc", 16), ("fe", 16), ("wa", 24), ("wb", 24), ("wc", 24), ("we", 24)):
            if scale[key].shape != (n,):
                raise ValueError("scale shape")
            tensors["scale." + key] = np.asarray(scale[key], np.float64)
    if adapter["W"].shape != (41, len(bank)) or adapter["A"].shape != (41,) or adapter["mu"].shape != (40,) or adapter["sd"].shape != (40,):
        raise ValueError("adapter shape")
    for key in ("W", "A", "mu", "sd"):
        tensors["adapter." + key] = np.asarray(adapter[key], np.float64)
    tensors["rows"] = np.stack([slot_rows(folder / "rows" / f"{name}.slot.safetensors") for name in bank])
    for name, value in tensors.items():
        if not np.isfinite(value).all():
            raise ValueError("non-finite " + name)
    if not (tensors["adapter.sd"] > 0).all():
        raise ValueError("adapter sd must be positive")
    writer = gguf.GGUFWriter(str(output), "frankie-delivery")
    writer.add_uint32("delivery.width", width)
    writer.add_array("delivery.bank", bank)
    writer.add_array("delivery.axes", [str(x) for x in feelings["axes"]])
    for name, value in tensors.items():
        writer.add_tensor("delivery." + name, np.ascontiguousarray(value))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    convert(args.folder, args.output)
