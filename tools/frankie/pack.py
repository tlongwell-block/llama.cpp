#!/usr/bin/env python3
"""Pack Frankie components and voice assets without re-quantizing any tensors.

An optional base package supplies unchanged components. Each --component NAME=GGUF
supplies one component; --voice plus matching transcript/codes sets its default voice.
The completed package is published only after every tensor is checked byte for byte.
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'gguf-py'))
import gguf

COMPONENTS = ('ear', 'bridge', 'side', 'vad', 'brain', 'talker', 'mouth', 'vision')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, help='reuse components and assets from an existing package')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--component', action='append', default=[], metavar='NAME=GGUF')
    parser.add_argument('--voice', type=Path)
    parser.add_argument('--voice-text-file', type=Path)
    parser.add_argument('--voice-codes', type=Path, help='frame-major little-endian I32, 16 codebooks')
    for name in ('expression', 'turn', 'breeze'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--bc-det', type=Path, help='native continuous Mimi plus MaAI human backchannel detector')
    args = parser.parse_args()
    temporary = args.output.with_name(args.output.name + '.incomplete')
    if args.output.exists() or temporary.exists():
        raise ValueError('output or incomplete output already exists')
    if bool(args.voice_text_file) != bool(args.voice_codes) or ((args.voice_text_file or args.voice_codes) and not args.voice):
        raise ValueError('matching ICL text/codes require --voice and each other')
    base = gguf.GGUFReader(args.base) if args.base else None
    if base is not None and (base.get_field('general.architecture').contents() != 'frankie' or base.get_field('frankie.version').contents() != 1):
        raise ValueError('unsupported base package')
    if not args.turn and (base is None or not any(t.name == 'assets.turn.gguf' for t in base.tensors)):
        raise ValueError('supply --turn with a combined VAP-BC asset; split legacy turn assets are not repackaged')
    if args.turn:
        turn = gguf.GGUFReader(args.turn)
        field = turn.get_field('turn.mode')
        tensors = {t.name: t for t in turn.tensors}
        if (field is None or field.contents() != 'duplex' or
                not {'turn.vap_head.weight', 'turn.bc_head.weight', 'turn.now.weight'} <= tensors.keys() or
                tensors['turn.encoder.downsample.1.weight'].shape.tolist() != [5, 256, 256]):
            raise ValueError('--turn must be a combined VAP-BC checkpoint from convert-turn.py')
    if args.bc_det:
        detector = gguf.GGUFReader(args.bc_det)
        fields = {name: detector.get_field(name) for name in ('turn.mode', 'clip.gen.audio.model_variant')}
        if any(field is None for field in fields.values()) or fields['turn.mode'].contents() != 'detection' or fields['clip.gen.audio.model_variant'].contents() != 'mimi':
            raise ValueError('--bc-det must contain BC-Det and continuous Mimi from convert-turn.py --mimi')
    replacements = {}
    for item in args.component:
        name, path = item.split('=', 1)
        if name not in COMPONENTS or name in replacements:
            raise ValueError('unknown or duplicate component: ' + name)
        replacements[name] = Path(path)
    if base is None:
        missing = [name for name in COMPONENTS if name not in replacements]
        if missing: raise ValueError('missing components: ' + ', '.join(missing))
        if not all((args.voice, args.voice_text_file, args.voice_codes, args.expression)):
            raise ValueError('a new package requires voice WAV/text/codes, expression and turn assets')
    source = {name: gguf.GGUFReader(path) for name, path in replacements.items()}
    tensor_map = {}
    source_shapes = {}
    sizes = {}
    for name in COMPONENTS:
        if name in source:
            reader = source[name]
            sizes[name] = replacements[name].stat().st_size
            tensor_map[f'assets.{name}.header'] = (np.memmap(replacements[name], mode='r', dtype=np.int8, shape=(reader.data_offset,)), gguf.GGMLQuantizationType.I8)
            source_shapes[f'assets.{name}.header'] = [reader.data_offset]
            for tensor in reader.tensors:
                tensor_map[f'frankie.{name}.{tensor.name}'] = (tensor.data, tensor.tensor_type)
                source_shapes[f'frankie.{name}.{tensor.name}'] = tensor.shape.tolist()
        else:
            field = base.get_field(f'frankie.{name}.size')
            if field is None: raise ValueError('base package is missing ' + name)
            sizes[name] = field.contents()
    for tensor in base.tensors if base is not None else ():
        if tensor.name in ('assets.vap.gguf', 'assets.bc.gguf'):
            continue
        if any(tensor.name.startswith(f'frankie.{name}.') or tensor.name == f'assets.{name}.header' for name in replacements):
            continue
        tensor_map[tensor.name] = (tensor.data, tensor.tensor_type)
        source_shapes[tensor.name] = tensor.shape.tolist()
    for name in ('expression', 'turn', 'breeze', 'bc_det'):
        path = getattr(args, name)
        if path:
            if path.stat().st_size > (256 if name == 'bc_det' else 64) * 1024 * 1024: raise ValueError(name + ' asset too large')
            tensor_map[f'assets.{name}.gguf'] = (np.fromfile(path, np.int8), gguf.GGMLQuantizationType.I8)
            source_shapes[f'assets.{name}.gguf'] = [path.stat().st_size]
    if args.voice:
        if args.voice.stat().st_size > 16 * 1024 * 1024: raise ValueError('voice asset too large')
        tensor_map['assets.voice.wav'] = (np.fromfile(args.voice, np.int8), gguf.GGMLQuantizationType.I8)
        source_shapes['assets.voice.wav'] = [args.voice.stat().st_size]
        if not args.voice_codes: raise ValueError('a package default voice requires matching ICL text/codes; runtime --voice also supports WAV alone')
        codes = np.fromfile(args.voice_codes, dtype='<i4')
        if not codes.size or codes.size % 16 or codes.size > 250 * 16 or np.any(codes < 0) or np.any(codes >= 2048):
            raise ValueError('invalid reference codec rows')
        tensor_map['assets.voice.codes'] = (codes.reshape(-1, 16), gguf.GGMLQuantizationType.I32)
        source_shapes['assets.voice.codes'] = [16, codes.size // 16]
    required = sum(data.nbytes for data, _ in tensor_map.values()) + 64 * 1024 * 1024
    if shutil.disk_usage(args.output.parent).free < required + 2 * 1024**3:
        raise ValueError('need output size plus 2 GiB free reserve')
    writer = gguf.GGUFWriter(temporary, 'frankie')
    changed = {f'frankie.{name}.size' for name in COMPONENTS}
    changed.update(('general.architecture', 'frankie.side_scale', 'frankie.components'))
    if args.voice: changed.update(('frankie.voice.ref_text', 'frankie.voice.sample_rate', 'frankie.voice.prime_frames'))
    for field in base.fields.values() if base is not None else ():
        if field.name.startswith('GGUF.') or field.name in changed: continue
        writer.add_key_value(field.name, field.contents(), field.types[0], field.types[-1] if len(field.types) > 1 else None)
    if base is None:
        writer.add_uint32('frankie.version', 1)
        writer.add_uint32('frankie.bridge_layer', 16)
    writer.add_array('frankie.components', list(COMPONENTS))
    writer.add_float32('frankie.side_scale', 0)
    for name, size in sizes.items(): writer.add_uint64(f'frankie.{name}.size', size)
    if args.voice:
        text = args.voice_text_file.read_text().strip()
        if not text or len(text.encode()) > 8192: raise ValueError('invalid reference transcript')
        writer.add_string('frankie.voice.ref_text', text)
        writer.add_uint32('frankie.voice.sample_rate', 24000)
        writer.add_uint32('frankie.voice.prime_frames', min(200, codes.size // 16))
    for name, (data, dtype) in tensor_map.items(): writer.add_tensor(name, data, raw_dtype=dtype)
    print(f"Writing {len(tensor_map)} tensors ({required / 1024**3:.2f} GiB)", flush=True)
    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
    check = gguf.GGUFReader(temporary)
    if len(check.tensors) != len(tensor_map): raise ValueError('packaged tensor inventory differs')
    manifest = {'base': args.base.name if args.base else None, 'replacements': {k:v.name for k,v in replacements.items()},
                'bytes': temporary.stat().st_size, 'tensors': {}}
    for tensor in check.tensors:
        expected, dtype = tensor_map[tensor.name]
        sha = hashlib.sha256(tensor.data).hexdigest()
        if tensor.tensor_type != dtype or tensor.shape.tolist() != source_shapes[tensor.name] or sha != hashlib.sha256(expected).hexdigest():
            raise ValueError('packaged tensor changed: ' + tensor.name)
        manifest['tensors'][tensor.name] = {'type':dtype.name, 'shape':tensor.shape.tolist(), 'bytes':tensor.n_bytes, 'sha256':sha}
    temporary.rename(args.output)
    args.output.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'path':str(args.output), 'bytes':manifest['bytes'], 'verified_tensors':len(check.tensors)}))


if __name__ == '__main__':
    main()
