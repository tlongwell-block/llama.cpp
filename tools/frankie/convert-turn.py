#!/usr/bin/env python3
"""Convert a frozen MaAI CPC VAP/backchannel checkpoint to shared mtmd GGUF.

Input is an NPZ of the installed reference's complete state_dict (including its
ALiBi buffers and objective codebook). This converter does not download weights.
"""
import argparse
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'gguf-py'))
import gguf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--mode', required=True, choices=['vap', 'bc'])
    args = parser.parse_args()
    state = np.load(args.input, allow_pickle=False)
    tensors = {}
    for name in state.files:
        value = state[name]
        if value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(f'Expected finite F32 tensor: {name}')
        if name.startswith('encoder1.'):
            other = name.replace('encoder1.', 'encoder2.', 1)
            if other not in state or not np.array_equal(value, state[other]):
                raise ValueError('This architecture requires identical channel encoder weights')
            name = name.replace('encoder1.', 'encoder.', 1)
        elif name.startswith('encoder2.'):
            continue
        elif not name.startswith(('ar.', 'ar_channel.', 'vap_head.' if args.mode == 'vap' else 'bc_head.')):
            continue
        if 'batchNorm' in name:
            value = value.reshape(256)
        tensors['turn.' + name] = np.ascontiguousarray(value)
    kernel = tensors['turn.encoder.downsample.1.weight'].shape[-1]
    if kernel != (10 if args.mode == 'vap' else 5):
        raise ValueError('Checkpoint downsampling cadence differs from the supported reference')
    if args.mode == 'vap':
        states = state['objective.codebook.emb.weight'].reshape(256, 2, 4)
        if not np.isin(states, [0, 1]).all():
            raise ValueError('Invalid VAP objective codebook')
        tensors['turn.now.weight'] = np.ascontiguousarray(states[:, :, :2].sum(-1).T)
    writer = gguf.GGUFWriter(args.output, 'maai-turn')
    writer.add_name('MaAI ' + args.mode + ' streaming CPC/ALiBi')
    writer.add_string('turn.mode', args.mode)
    writer.add_uint32('turn.sample_rate', 16000)
    writer.add_uint32('turn.step_samples', 1600)
    writer.add_uint32('turn.context_frames', 200)
    for name, value in tensors.items():
        writer.add_tensor(name, value)
    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
    print(f'{args.output}: {len(tensors)} tensors, {args.output.stat().st_size} bytes')


if __name__ == '__main__':
    main()
