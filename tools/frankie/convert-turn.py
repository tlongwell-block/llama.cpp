#!/usr/bin/env python3
"""Convert MaAI VAP-BC or human BC-Det checkpoints to native mtmd GGUFs.

Input is the checkpoint state_dict (.pt), or its complete F32 NPZ export,
including ALiBi buffers and objective codebook. BC-Det also needs --mimi
with the matching F32 continuous-encoder ONNX export. No weights are downloaded.
"""
import argparse
import hashlib
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'gguf-py'))
import gguf


def detector_tensors(state, mimi):
    """The upstream continuous-Mimi export contains the original F32 weights."""
    import onnx
    model = onnx.load(mimi)
    tensors = {}
    mapping = {
        'input_layernorm': 'ln1', 'post_attention_layernorm': 'ln2',
        'self_attn.q_proj': 'attn_q', 'self_attn.k_proj': 'attn_k',
        'self_attn.v_proj': 'attn_v', 'self_attn.o_proj': 'attn_out',
        'mlp.fc1': 'ffn_up', 'mlp.fc2': 'ffn_down',
        'self_attn_layer_scale': 'ls1', 'mlp_layer_scale': 'ls2',
    }
    for value in model.graph.initializer:
        name = value.name.removeprefix('encoder.model.')
        data = onnx.numpy_helper.to_array(value)
        if name.startswith('encoder.layers.'):
            parts = name.split('.')
            index, suffix = int(parts[2]), parts[-1]
            if index == 0:
                key = 'a.seanet.conv_in'
            elif index == 14:
                key = 'a.seanet.conv_out'
            elif index % 3 == 0:
                key = f'a.seanet.blk.{index // 3 - 1}.scale_conv'
            elif index % 3 == 1:
                key = f'a.seanet.blk.{index // 3}.res_conv{1 if parts[4] == "1" else 2}'
            else:
                raise ValueError('Unsupported Mimi convolution: ' + name)
            key += '.' + suffix
        elif name.startswith('encoder_transformer.layers.'):
            _, _, layer, rest = name.split('.', 3)
            module, suffix = rest.rsplit('.', 1)
            key = f'a.blk.{layer}.{mapping[module]}.{ "weight" if suffix == "scale" else suffix}'
        elif name == 'downsample.conv.weight':
            key = 'a.downsample.conv.weight'
        else:
            raise ValueError('Unsupported Mimi initializer: ' + name)
        if data.dtype != np.float32 or not np.isfinite(data).all():
            raise ValueError('Expected finite F32 Mimi weights: ' + name)
        if key in tensors:
            raise ValueError('Duplicate Mimi tensor: ' + key)
        tensors[key] = np.ascontiguousarray(data)
    if len(tensors) != 125:
        raise ValueError('Expected the eight-layer continuous Mimi encoder')
    for name, value in state.items():
        if name.startswith(('ar.', 'ar_channel.', 'decrease_dimension.', 'bc_classifier.')):
            if value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError('Expected finite F32 detector tensor: ' + name)
            tensors['turn.' + name] = np.ascontiguousarray(value)
    for name, shape in [('decrease_dimension.weight', (256, 512)), ('decrease_dimension.bias', (256,)),
                        ('bc_classifier.weight', (1, 256)), ('bc_classifier.bias', (1,))]:
        if tensors.get('turn.' + name, np.empty(0)).shape != shape:
            raise ValueError('Missing or invalid BC-Det head: ' + name)
    return tensors


def write_detector(args, state):
    tensors = detector_tensors(state, args.mimi)
    writer = gguf.GGUFWriter(args.output, 'clip')
    writer.add_name('MaAI backchannel detection with continuous Mimi')
    writer.add_string('turn.mode', 'detection')
    writer.add_string('turn.source.sha256', hashlib.sha256(args.input.read_bytes()).hexdigest())
    writer.add_string('turn.encoder.sha256', hashlib.sha256(args.mimi.read_bytes()).hexdigest())
    writer.add_uint32('turn.sample_rate', 16000)
    writer.add_uint32('turn.step_samples', 1280)
    writer.add_uint32('turn.context_frames', 250)
    writer.add_clip_has_audio_encoder(True)
    writer.add_clip_audio_projector_type(gguf.VisionProjectorType.POCKETTTS_SPKENC)
    writer.add_gen_audio_model_variant('mimi')
    writer.add_audio_projection_dim(512)
    writer.add_audio_embedding_length(512)
    writer.add_audio_block_count(8)
    writer.add_audio_feed_forward_length(2048)
    writer.add_audio_head_count(8)
    writer.add_audio_attention_layernorm_eps(1e-5)
    writer.add_audio_num_mel_bins(1)
    for name, value in tensors.items():
        writer.add_tensor(name, value)
    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
    print(f'{args.output}: {len(tensors)} tensors, {args.output.stat().st_size} bytes')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--mimi', type=Path, help='convert BC-Det with the upstream F32 continuous-Mimi ONNX export')
    args = parser.parse_args()
    if args.input.suffix == '.npz':
        state = dict(np.load(args.input, allow_pickle=False))
    else:
        import torch
        checkpoint = torch.load(args.input, map_location='cpu', weights_only=True)
        state = {name: tensor.detach().cpu().float().numpy() for name, tensor in
                 checkpoint.get('state_dict', checkpoint).items()}
    if args.mimi:
        write_detector(args, state)
        return
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
