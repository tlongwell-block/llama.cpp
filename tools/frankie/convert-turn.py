#!/usr/bin/env python3
"""Convert a MaAI VAP-BC checkpoint to a shared, dual-head mtmd GGUF.

Input is the checkpoint state_dict (.pt), or its complete F32 NPZ export,
including ALiBi buffers and objective codebook. No weights are downloaded.
"""
import argparse
import hashlib
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'gguf-py'))
import gguf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.input.suffix == '.npz':
        state = dict(np.load(args.input, allow_pickle=False))
    else:
        import torch
        state = {name: tensor.detach().cpu().float().numpy() for name, tensor in
                 torch.load(args.input, map_location='cpu', weights_only=True).items()}
    tensors = {}
    for name, value in state.items():
        if value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(f'Expected finite F32 tensor: {name}')
        if name.startswith('encoder1.'):
            other = name.replace('encoder1.', 'encoder2.', 1)
            if other not in state or not np.array_equal(value, state[other]):
                raise ValueError('This architecture requires identical channel encoder weights')
            name = name.replace('encoder1.', 'encoder.', 1)
        elif name.startswith('encoder2.'):
            continue
        elif not name.startswith(('encoder.', 'ar.', 'ar_channel.', 'vap_head.', 'bc_head.')):
            continue
        if 'batchNorm' in name:
            value = value.reshape(256)
        tensors['turn.' + name] = np.ascontiguousarray(value)
    kernel = tensors['turn.encoder.downsample.1.weight'].shape[-1]
    if kernel != 5:
        raise ValueError('Expected the VAP-BC checkpoint with a five-sample downsampling kernel')
    for name, shape in [('vap_head.weight', (256, 256)), ('vap_head.bias', (256,)),
                        ('bc_head.weight', (1, 256)), ('bc_head.bias', (1,))]:
        if tensors.get('turn.' + name, np.empty(0)).shape != shape:
            raise ValueError('Missing or invalid VAP-BC head: ' + name)
    states = state['objective.codebook.emb.weight'].reshape(256, 2, 4)
    if not np.isin(states, [0, 1]).all():
        raise ValueError('Invalid VAP objective codebook')
    tensors['turn.now.weight'] = np.ascontiguousarray(states[:, :, :2].sum(-1).T)
    writer = gguf.GGUFWriter(args.output, 'maai-turn')
    writer.add_name('MaAI VAP-BC streaming CPC/ALiBi')
    writer.add_string('turn.mode', 'duplex')
    writer.add_string('turn.source.sha256', hashlib.sha256(args.input.read_bytes()).hexdigest())
    writer.add_uint32('turn.sample_rate', 16000)
    writer.add_uint32('turn.step_samples', 1600)
    writer.add_uint32('turn.context_frames', 200)
    for name, value in tensors.items():
        writer.add_tensor(name, value)
    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
    print(f'{args.output}: {len(tensors)} tensors, {args.output.stat().st_size} bytes')


if __name__ == '__main__':
    main()
