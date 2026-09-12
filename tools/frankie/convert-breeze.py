#!/usr/bin/env python3
"""Convert Breeze speech components using the existing Qwen3, Gemma and codec paths."""

import argparse
import json
import logging
from pathlib import Path
import sys

import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "gguf-py")]
import gguf
from conversion.gemma import Gemma3Model
from conversion.qwen import Qwen3Model
from conversion.qwen3tts import Qwen3TTSSpeakerEncoderModel
from conversion.pockettts import PocketTTSMmprojModel


def tensor(directory, name):
    index = directory / "model.safetensors.index.json"
    filename = json.loads(index.read_text())["weight_map"][name] if index.exists() else "model.safetensors"
    with safe_open(directory / filename, framework="pt", device="cpu") as source:
        return source.get_tensor(name)


class BreezeBackbone(Qwen3Model):
    model_arch = gguf.MODEL_ARCH.QWEN3

    @classmethod
    def filter_tensors(cls, item):
        name, load = item
        if name.startswith("backbone_model.layers.") or name == "backbone_model.norm.weight":
            return name.replace("backbone_model.", "model.", 1), load
        if name == "lm_head.weight":
            return item
        return None

    def set_vocab(self):
        w = self.gguf_writer
        w.add_tokenizer_model("llama")
        w.add_token_list([f"<|codec_{i}|>" for i in range(2052)])
        w.add_token_types([gguf.TokenType.NORMAL] * 2051 + [gguf.TokenType.CONTROL])
        w.add_token_scores([0.0] * 2052)
        w.add_bos_token_id(2051)
        w.add_eos_token_id(2051)
        w.add_add_bos_token(False)
        w.add_add_eos_token(False)
        w.add_suppress_tokens([0, 2048, 2049, 2050])

    def generate_extra_tensors(self):
        embedding = tensor(self.dir_model, "depth_decoder.model.embed_tokens.weight")[:2051]
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:1])])
        yield "model.embed_tokens.weight", embedding


class BreezeEncoder(Gemma3Model):
    model_arch = gguf.MODEL_ARCH.GEMMA_EMBEDDING

    @classmethod
    def filter_tensors(cls, item):
        name, load = item
        if not name.startswith("text_encoder.") or name.endswith("eoi_embedding"):
            return None
        name = name.replace("text_encoder.", "model.", 1)
        for original, canonical in (
            ("pre_self_attn_layernorm", "input_layernorm"),
            ("post_self_attn_layernorm", "post_attention_layernorm"),
            ("pre_feedforward_layernorm", "pre_feedforward_layernorm"),
            ("post_feedforward_layernorm", "post_feedforward_layernorm"),
        ):
            name = name.replace(original, canonical)
        return name, load

    def get_vocab_base(self):
        from transformers import PreTrainedTokenizerFast

        tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(self.dir_model / "tokenizer.json"))
        return super().get_vocab_base(tokenizer)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_causal_attention(False)
        # All rows fit within both halves of the native encoder's local window.
        self.gguf_writer.add_context_length(256)
        self.gguf_writer.add_pooling_type(gguf.PoolingType.NONE)

    def modify_tensors(self, data, name, bid):
        if name == "model.embed_tokens.weight":
            data = data.float().clone()
            eoi = tensor(self.dir_model, "text_encoder.embed_tokens.eoi_embedding")
            data[self.hparams["eoi_token_index"]] = eoi / data.shape[1] ** .5
        yield from super().modify_tensors(data, name, bid)


class BreezeAcoustic(Qwen3TTSSpeakerEncoderModel):
    model_arch = gguf.MODEL_ARCH.MMPROJ

    def index_tensors(self, remote_hf_model_id=None):
        return {}

    def _wav_decoder_dir(self):
        return self.dir_model / "audio_tokenizer"

    def set_gguf_parameters(self):
        w = self.gguf_writer
        w.add_file_type(self.ftype)
        w.add_clip_has_gen_audio_encoder(True)
        w.add_clip_gen_audio_projector_type(gguf.VisionProjectorType.QWEN3TTS_GEN)
        w.add_gen_audio_model_variant("breeze")
        w.add_clip_has_audio_encoder(True)
        w.add_clip_audio_projector_type(gguf.VisionProjectorType.POCKETTTS_SPKENC)
        w.add_audio_projection_dim(2048)
        w.add_audio_block_count(8)
        w.add_audio_embedding_length(512)
        w.add_audio_feed_forward_length(2048)
        w.add_audio_head_count(8)
        w.add_audio_attention_layernorm_eps(1e-5)
        w.add_audio_num_mel_bins(1)
        config = self.global_config["talker_config"]["code_predictor_config"]
        for key, value in {"projection_dim": 2048, "embedding_length": config["hidden_size"],
                           "feed_forward_length": config["intermediate_size"],
                           "block_count": config["num_hidden_layers"], "head_count": config["num_attention_heads"],
                           "head_count_kv": config["num_key_value_heads"],
                           "attention_layernorm_eps": config["rms_norm_eps"]}.items():
            getattr(w, "add_gen_audio_" + key)(value)

    def generate_extra_tensors(self):
        yield from self._generate_code2wav_tensors()
        yield from self._generate_encoder_tensors()
        source = self.dir_model
        embedding = tensor(source, "depth_decoder.model.embed_tokens.weight").reshape(16, 2051, 2048)[:, :2048]
        yield "a.gen.code.out_embd.weight", embedding[0]
        yield "a.gen.code.embd.weight", embedding[1:]
        yield "a.gen.code.head.weight", tensor(source, "depth_decoder.codebooks_head.weight").transpose(1, 2)[:, :2048]
        yield "a.gen.code.output_norm.weight", tensor(source, "depth_decoder.model.norm.weight")
        yield "a.gen.code.proj_in.weight", tensor(source, "depth_decoder.model.inputs_embeds_projector.weight")
        for layer in range(12):
            for name, mapped in self._CODE_LAYER_TENSOR_MAP.items():
                if name in ("self_attn.q_norm", "self_attn.k_norm"):
                    continue
                yield self.format_tensor_name(mapped, layer), tensor(source, f"depth_decoder.model.layers.{layer}.{name}.weight")
        # Llama 3 frequency scaling; GGML RoPE divides frequencies by this table.
        config = self.global_config["talker_config"]["code_predictor_config"]
        rope = config["rope_scaling"]
        frequency = config["rope_theta"] ** (-np.arange(0, 128, 2, dtype=np.float64) / 128)
        wavelength = 2 * np.pi / frequency
        low = rope["original_max_position_embeddings"] / rope["low_freq_factor"]
        high = rope["original_max_position_embeddings"] / rope["high_freq_factor"]
        smooth = (rope["original_max_position_embeddings"] / wavelength - rope["low_freq_factor"]) / (rope["high_freq_factor"] - rope["low_freq_factor"])
        multiplier = np.where(wavelength < high, 1, np.where(wavelength > low, 1 / rope["factor"],
                              (1 - smooth) / rope["factor"] + smooth))
        yield "a.gen.code.rope_freqs.weight", torch.from_numpy((1 / multiplier).astype(np.float32))

    def _generate_encoder_tensors(self):
        root = self._wav_decoder_dir()
        with safe_open(root / "model.safetensors", framework="pt", device="cpu") as source:
            for name in source.keys():
                if name.startswith("encoder.encoder.layers."):
                    canonical = name.replace("encoder.encoder.layers.", "mimi.encoder.model.", 1)
                    yield from PocketTTSMmprojModel._seanet_tensor(self, canonical, source.get_tensor(name), 4)
        yield "a.downsample.conv.weight", tensor(root, "encoder.downsample.conv.weight")
        mappings = {
            "input_layernorm": "ln1", "post_attention_layernorm": "ln2",
            "self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k",
            "self_attn.v_proj": "attn_v", "self_attn.o_proj": "attn_out",
            "mlp.fc1": "ffn_up", "mlp.fc2": "ffn_down",
            "self_attn_layer_scale": "ls1", "mlp_layer_scale": "ls2",
        }
        for layer in range(8):
            for name, mapped in mappings.items():
                suffix = "scale" if name.endswith("layer_scale") else "weight"
                yield f"a.blk.{layer}.{mapped}.weight", tensor(root, f"encoder.encoder_transformer.layers.{layer}.{name}.{suffix}")
                if name.endswith("layernorm"):
                    yield f"a.blk.{layer}.{mapped}.bias", tensor(root, f"encoder.encoder_transformer.layers.{layer}.{name}.bias")
        for i, (group, count) in enumerate((("semantic", 1), ("acoustic", 15))):
            prefix = f"encoder.quantizer.{group}_residual_vector_quantizer"
            yield f"a.rvq.{i}.in.weight", tensor(root, prefix + ".input_proj.weight").squeeze(-1)
            tables = []
            for layer in range(count):
                name = prefix + f".layers.{layer}.codebook."
                table = tensor(root, name + "embed_sum").float() / tensor(root, name + "cluster_usage").float().clamp_min(1e-5)[:, None]
                tables.append(table)
            codebooks = torch.stack(tables)
            yield f"a.rvq.{i}.codebook.weight", codebooks
            yield f"a.rvq.{i}.norm", codebooks.square().sum(-1) / 2

    def modify_tensors(self, data, name, bid):
        yield name, data

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        if not name.startswith("a.gen.code."):
            return gguf.GGMLQuantizationType.F32
        if name.endswith("rope_freqs.weight"):
            return gguf.GGMLQuantizationType.F32
        return gguf.GGMLQuantizationType.Q8_0 if n_dims >= 2 else gguf.GGMLQuantizationType.F32


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--voice-conditioning", type=Path, help="NPZ containing a cached Breeze voice_prefix")
    ap.add_argument("--reference-rms", type=float, help="RMS of the conditioning voice, in (0, 1]")
    args = ap.parse_args()
    if bool(args.voice_conditioning) != (args.reference_rms is not None):
        ap.error("--voice-conditioning and --reference-rms must be supplied together")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    logging.basicConfig(level=logging.INFO)
    cfg = json.loads((args.model / "config.json").read_text())
    if (cfg["backbone_config"]["hidden_size"], cfg["depth_decoder_config"]["num_hidden_layers"],
        cfg["text_encoder_config"]["hidden_size"]) != (2048, 12, 1152):
        raise ValueError("Unsupported Breeze component geometry")
    depth = cfg["depth_decoder_config"]
    acoustic = {"talker_config": {"hidden_size": 2048, "code_predictor_config": depth},
                "speaker_encoder_config": {"hidden_size": 2048}}
    for name, cls, config in (("talker", BreezeBackbone, {**cfg["backbone_config"], "vocab_size": 2052}),
                              ("encoder", BreezeEncoder, cfg["text_encoder_config"]),
                              ("acoustic", BreezeAcoustic, acoustic)):
        cls(args.model, gguf.LlamaFileType.MOSTLY_Q8_0, args.output / (name + ".gguf"),
            hparams=config, use_temp_file=True, model_name="Breeze " + name).write()
    if args.voice_conditioning:
        with np.load(args.voice_conditioning, allow_pickle=False) as source:
            voice = source["voice_prefix"].astype(np.float32)
        if voice.ndim != 2 or voice.shape[1] != 2048 or not 1 <= len(voice) <= 1024 or not np.isfinite(voice).all():
            raise ValueError("Invalid Breeze voice prefix")
        if not 0 < args.reference_rms <= 1:
            raise ValueError("Invalid reference RMS")
        w = gguf.GGUFWriter(args.output / "conditioning.gguf", "breeze")
        w.add_tensor("breeze.projection", tensor(args.model, "text_encoder_proj.weight").float().numpy())
        w.add_tensor("voice.prefix", voice)
        embedding = tensor(args.model, "depth_decoder.model.embed_tokens.weight").reshape(16, 2051, 2048)
        w.add_tensor("voice.eos", embedding[:, 0].float().sum(0).numpy())
        w.add_tensor("voice.rms", np.array([args.reference_rms], dtype=np.float32))
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()


if __name__ == "__main__":
    main()
