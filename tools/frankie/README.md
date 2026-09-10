# Frankie Realtime

This opt-in server runs Frankie's Qwen brain, Parakeet ear, bridges, Qwen3-TTS voice, vision and turn-taking models through llama.cpp, mtmd and ggml. It accepts a single Frankie GGUF and binds to loopback. Buzz owns tools, permissions and conversation policy; the native server performs inference.

## Build

ICU development headers are required for NFC speech alignment. No model assets are downloaded by the build.

```sh
# Metal on macOS; use -DICU_ROOT=<prefix> if ICU is outside CMake's search paths.
cmake -S . -B build-frankie -DLLAMA_BUILD_FRANKIE=ON -DGGML_METAL=ON
cmake --build build-frankie --target llama-frankie-realtime -j4

# CPU-only: add -DGGML_METAL=OFF -DGGML_CUDA=OFF, then run --device cpu.
# CUDA: add -DGGML_CUDA=ON -DGGML_METAL=OFF, then run --device gpu.
# A portable macOS build also needs -DGGML_NATIVE=OFF.
```

CPU and Metal have integration coverage. CUDA uses the same model graphs and runtime options; an actual CUDA run and 24 GiB full-context memory measurement are still required before that hardware target is accepted.

## Start and choose thinking

```sh
export FRANKIE_REALTIME_TOKEN="$(openssl rand -hex 32)"
build-frankie/bin/llama-frankie-realtime /path/to/frankie.gguf 18793 \
  --device gpu --ctx-size 131072 --cache-type q4_0 --thinking none
```

The authenticated WebSocket endpoint is `/v1/realtime`. `/health` becomes ready after the mouth warm-up. Only one connection owns the model session at a time. There is no automatic reconnect or replay of tool effects.

Thinking defaults to off. `--thinking` accepts `none` (or `off`), `minimal`, `low`, `medium`, `high`, `xhigh`, and `max`. Their reasoning budgets are 0, 128, 512, 2048, 8192, 16384, and 32768 tokens. The existing llama.cpp chat template and common sampler enforce the budget; reasoning does not feed the speech bridge or appear in spoken transcripts. More thinking can delay the first spoken answer substantially.

A client may change the level with an ordinary session update while the server is idle:

```json
{"type":"session.update","session":{"type":"realtime","reasoning":{"effort":"high"}}}
```

An optional `budget_tokens` supplies an explicit bounded budget; `none` requires zero. The server acknowledges the effective setting in `session.updated`. Buzz forwards its existing thinking-effort configuration and verifies the acknowledgment. The companion browser presents the level before connecting.

## Supply a voice reference

```sh
build-frankie/bin/llama-frankie-realtime /path/to/frankie.gguf 18793 \
  --device gpu --voice /path/to/my-voice.wav
```

A non-silent WAV up to 20 seconds uses Qwen3-TTS's existing native speaker encoder. This path replaces the packaged voice and clears its ICL text and codes. It does not require rebuilding the GGUF or installing Python inference services.

For reference-text/codec conditioning as well as the speaker embedding, supply all three matching inputs:

```sh
# I32 file: little-endian, frame-major [frames,16], codec IDs 0..2047.
# These codes must come from this WAV, and the UTF-8 text must transcribe it.
... --voice my-voice.wav --voice-text-file my-voice.txt --voice-codes my-voice.i32
```

The native runtime does not yet encode arbitrary WAVs into ICL codec IDs. WAV-only speaker conditioning is supported; automatic generation of the optional ICL codes is separate work. Speaker caching uses exact PCM bytes, and expression offsets do not mutate the cached base voice.

## Quantization and package assembly

The current package retains the custom Q4_1 brain and uses Q8_0 for supported Parakeet and Qwen3-TTS matrices. Spatial convolutions, normalization and other unsupported or sensitive tensors retain their original types. Vision remains F16. Q8 describes the quantized matrices, not every tensor in the file.

Use the existing `llama-quantize` for the Qwen talker backbone. `tools/mtmd/quantize-audio.py` uses gguf-py's Q8 implementation for the audio projectors:

```sh
python tools/mtmd/quantize-audio.py encoder-f32.gguf encoder-q8.gguf --model parakeet
python tools/mtmd/quantize-audio.py mmproj-f16.gguf mmproj-q8.gguf --model qwen3tts
```

`tools/frankie/pack.py` can assemble the first package directly from components, without an existing Frankie GGUF:

```sh
python tools/frankie/pack.py --output frankie.gguf \
  --component brain=brain-q4_1.gguf --component vision=vision-f16.gguf \
  --component ear=ear-q8.gguf --component bridge=bridge-f32.gguf \
  --component talker=talker-q8.gguf --component mouth=mouth-q8.gguf \
  --component side=side-f32.gguf --component vad=vad-f32.gguf \
  --voice voice.wav --voice-text-file voice.txt --voice-codes voice.i32 \
  --expression expression.gguf --vap vap.gguf --bc bc.gguf
```

The custom MLX brain, vision and Parakeet exporters are in `examples/model-conversion/convert_frankie_{brain,vision,ear}.py`. The ear exporter accepts `MODEL OUTPUT` and uses the installed `parakeet_mlx` frontend; its optional middle `FIXTURE` argument preserves frozen filter/window exports. Use the existing `convert_hf_to_gguf.py` for Qwen3-TTS, once for the talker and once with `--mmproj` for the speaker/code/codec component. These source conversions need the Python dependencies of their reference models; native inference does not.

Bridge, tone, optional Side, VAD, expression, turn checkpoints and matching reference codes must come from the same source configuration. In particular, tone's emotion rows come from the original brain embedding, and the Studio's default Side has zero output. Do not substitute unrelated adapters or reference codec IDs.

To update an existing package, pass `--base existing.gguf` and only the replacement components/assets. The packer preserves unchanged tensors and metadata, verifies every written tensor against its input and writes a manifest. Failed assembly leaves an `.incomplete` file rather than a final GGUF.

The September 10 package is 20,326,799,488 bytes (18.93 GiB). That is slightly above 20 decimal GB. File size alone cannot establish whether the complete system fits a 24 GiB GPU: KV/recurrent state, compute buffers, vision, audio and turn models must be counted together during execution.

Do not substitute a stock Qwen GGUF without validating the trained bridge seams. This brain was converted from the custom MLX affine-four-bit checkpoint into Q4_1, including a requantization step.

## Runtime behavior and tuning

Defaults: 131072 context tokens, Q4_0 KV with flash attention, 4 CPU threads, 128-token logical and physical prompt batches on CPU, or 512 on GPU. `--batch-size` and `--ubatch-size` allow tuning memory and prompt throughput. Context supports 4096..262144 tokens; cache types are `q4_0`, `q8_0` and `f16`.

Brain sampling is temperature .7, top-p .8, top-k 20, presence penalty 0. Side defaults to 0. The Qwen talker/code predictor use temperature .6, top-k 50, top-p 1 and the reference implementation's repetition penalty. Speech cuts at sentence boundaries, joins short openers, and bounds chunks to 50 words. The expression head pools brain layer-16 states and applies gated speaker-vector offsets, a .9 peak ceiling and a reference-relative RMS ceiling.

The small VAD graph runs on CPU so GPU work cannot block microphone transport. Brain, ear, voice, vision and turn inference follow the selected backend.

During speech, the existing phrase worker gives the mouth priority when emitted audio has less than 160 ms of lead. Brain decoding resumes once that lead is restored or the phrase finishes. This avoids audio starvation from competing decode graphs without delaying the first PCM chunk or adding a separate inference implementation.

The phrase queue copies only the token states for each spoken span. Side conditioning adds offsets through the existing Qwen TTS embedding cache, without a second vocabulary matrix. Small component graphs validate backend support once when allocated, and release their host weight copies after loading.

Input uses VAD, in-speech ear prefill with a 20-frame lag, and speculative generation after short silence. VAP gates publication, with a 1500 ms silence ceiling and stale-prediction fallback. Resumed speech restores recurrent state. Short heard false starts can merge within 700 ms, without crossing published tool calls. Interruption history retains completed heard phrases rather than claiming exact word alignment.

Backchannels use the shared MaAI graph at 10 Hz, the actual rendered speaker PCM, and a temporary brain probe restricted to listener reactions. They have separate bounded audio events and never create an assistant turn or execute a tool. The probe shares prefix state through llama.cpp sequence operations and restores it after selection. Normal speech supersedes listener audio.

Default limits are 90 seconds per input utterance, 300 seconds of output audio per response, and 4096 answer tokens. Use `--max-utterance-seconds` (2..120), `--max-output-audio-seconds` (1..3600), and `--max-output-tokens N|inf` to change them. Realtime `session.max_output_tokens` also accepts a positive integer or `"inf"`. Reasoning has its separate budget. An overlong VAD capture is cleared, then capture resumes after silence; no partial user turn is published. Output exhaustion reports an incomplete response and keeps capture live. Immutable input checkpoints are shared with speculative work; recovery preserves the input prefix and excludes unspoken generation. There are at most 4096 history messages and 128 audio segments; history rolls over at safe boundaries within the actual context budget.

Inline PNG/JPEG input uses the existing vision and M-RoPE path: at most 512 KiB, four megapixels, a 4096-pixel edge and four images in a session. Audio is mono PCM16 at 24 kHz.

`FRANKIE_VERIFY_ROLLBACK=1` enables expensive byte-for-byte recurrent-state diagnostics. `FRANKIE_REPORT_MEMORY=1` reports retained model and compute buffers at session close. These are diagnostics, not substitutes for peak device-memory measurements under CUDA. The ordinary startup log also reports component memory and the allocated context.
