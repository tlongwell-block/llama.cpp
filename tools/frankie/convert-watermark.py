#!/usr/bin/env python3
"""Convert AudioSeal's streaming 16-bit generator to a native F32 GGUF.

Input is the upstream ``generator_streaming.pth`` (facebookresearch/audioseal,
card ``audioseal_wm_streaming``). Loading it needs ``torch``, ``omegaconf`` and
the ``audioseal`` package (0.2+, streaming support); weight norm is folded into
plain convolution weights. No weights are downloaded.
"""
import argparse
import hashlib
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'gguf-py'))
import gguf


def tensors(model):
    import torch
    for module in model.modules():
        try:
            torch.nn.utils.remove_weight_norm(module)
        except ValueError:
            pass
    out = {}
    for name, value in model.state_dict().items():
        key = (name.replace('.conv.conv.inner_conv', '').replace('.convtr.convtr.inner_conv', '')
                   .replace('.lstm.', '.').replace('msg_processor.msg_processor', 'message'))
        data = value.detach().cpu().numpy()
        if '.convtr.' in name and name.endswith('.weight'):
            # [IC, OC, K] -> [OC, K, IC], so the native graph multiplies it without a transpose.
            data = data.transpose(1, 2, 0)
        if data.dtype != np.float32 or not np.isfinite(data).all():
            raise ValueError('Expected finite F32 tensor: ' + name)
        out['wm.' + key] = np.ascontiguousarray(data)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input', type=Path, help='generator_streaming.pth')
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    from audioseal import AudioSeal  # type: ignore[import-not-found, ty:unresolved-import]
    model = AudioSeal.load_generator(str(args.input), nbits=16).eval().cpu()
    if model.frame_size != 320 or model.msg_processor is None or model.normalizer is not None:
        raise ValueError('Expected the 16-bit streaming generator with 320-sample frames and no normalizer')
    values = tensors(model)
    if len(values) != 73 or values['wm.message.weight'].shape != (32, 128):
        raise ValueError(f'Unexpected AudioSeal generator inventory ({len(values)} tensors)')
    writer = gguf.GGUFWriter(args.output, 'clip')
    writer.add_name('AudioSeal streaming watermark generator, 16 bits')
    writer.add_string('wm.source.sha256', hashlib.sha256(args.input.read_bytes()).hexdigest())
    writer.add_uint32('wm.sample_rate', 16000)
    writer.add_uint32('wm.frame', 320)
    writer.add_uint32('wm.bits', 16)
    for name, value in values.items():
        writer.add_tensor(name, value)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f'wrote {args.output} ({len(values)} tensors)')


if __name__ == '__main__':
    main()
