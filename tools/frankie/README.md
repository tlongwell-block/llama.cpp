# Frankie Realtime

This opt-in server runs Frankie's Qwen brain, Parakeet ear, bridges, Qwen3-TTS or Breeze voice, vision and turn-taking models through llama.cpp, mtmd and ggml. It accepts a single Frankie GGUF and binds to loopback. Buzz owns tools, permissions and conversation policy; the native server performs inference.

## Build

ICU development headers and the i18n, uc and data libraries are required for NFC speech alignment and spoken-number normalization. No model assets are downloaded by the build.

```sh
# Metal on macOS; use -DICU_ROOT=<prefix> if ICU is outside CMake's search paths.
cmake -S . -B build-frankie -DLLAMA_BUILD_FRANKIE=ON -DGGML_METAL=ON
cmake --build build-frankie --target llama-frankie-realtime -j4

# CPU-only: add -DGGML_METAL=OFF -DGGML_CUDA=OFF, then run --device cpu.
# CUDA: add -DGGML_CUDA=ON -DGGML_METAL=OFF, then run --device gpu.
# A portable macOS build also needs -DGGML_NATIVE=OFF.
```

CPU, Metal and CUDA use the same model graphs and runtime options. Measure peak device memory with the complete voice pipeline and intended context; a successful brain-only load does not establish full-system capacity.

## Start and choose thinking

```sh
export FRANKIE_REALTIME_TOKEN="$(openssl rand -hex 32)"
build-frankie/bin/llama-frankie-realtime /path/to/frankie.gguf 18793 \
  --device gpu --ctx-size 131072 --cache-type q4_0 --thinking none
```

The authenticated WebSocket endpoint is `/v1/realtime`. `/health` becomes ready after the mouth warm-up. Only one connection owns the voice session at a time. There is no automatic reconnect or replay of tool effects. `--host` defaults to `127.0.0.1`; set `--host 0.0.0.0` to accept LAN connections. The same bearer token authenticates HTTP and WebSocket requests.

For a mouth with a separate text encoder, such as Breeze, `--text-encoder-device cpu` keeps that encoder on the CPU while `--device gpu` runs the brain and acoustic models on the GPU. This leaves more VRAM for context and live-audio buffers without changing quantization. The default is `gpu`; measure the additional speech latency when selecting CPU placement. Qwen3-TTS has no separate text encoder, so this option does not change its execution.

### Experimental concurrent text and image requests

Add `--http-slots 2 --http-ctx-size 4096` to expose `/v1/chat/completions`,
`/v1/completions`, and `/v1/models` on the same port. HTTP requests share the
loaded brain and vision weights; each slot owns its context and sampler.
The default is zero HTTP slots. The HTTP context limit includes prompt and
output tokens and is separate from `--ctx-size`, which still applies to voice.
The additional slots use the selected `--cache-type`. `--mtp-tokens` applies
to both voice and HTTP requests. They share the existing MTP head, with
independent draft histories and target verification for each request.
The batch size must exceed the HTTP slot count times `1 + --mtp-tokens`.

Use a standard OpenAI client with this server's base URL and bearer token.
Chat messages accept ordered `text` and `image_url` content parts, including
multiple images, inline base64 data URLs, and HTTP(S) image URLs. The legacy
`/v1/completions` endpoint accepts a text prompt. Both support streamed SSE
and non-streamed results. `stream_options.include_usage` returns final usage.
Each chat request accepts at most four images, with a 10 MiB limit per image
and an 18 MiB request-body limit. Bitmap dimension limits still apply.

Realtime uses `conversation.item.create` user messages containing
`{"type":"input_image","image_url":"data:image/png;base64,..."}` alongside
`input_text` parts. Realtime image URLs must contain inline PNG or JPEG data.
Both image interfaces accept `detail: "auto"`, `"low"`, or `"high"`; like the
ordinary llama.cpp handler, preprocessing uses the model's configured image
budget rather than OpenAI-specific resolution tiers.

HTTP work advances between bounded brain steps. While requests are pending,
the speech producer may keep up to 800 ms of audio ahead to absorb compute
bursts; voice-only buffering stays unchanged. Long prompts and images can
still take time to produce their first text token. Contexts and cancellation
are independent; cancelling a voice reply does not cancel HTTP requests.
This is an experimental subset of the completion APIs. Unsupported options
return explicit errors; the Responses API is not exposed.

### Optional MTP

`--mtp-tokens N` enables the existing llama.cpp Qwen MTP draft head, with one to four proposed tokens per step. The default is zero (disabled). CPU, Metal and CUDA use the same GGUF, graphs and common speculative sampler. Both `--batch-size` and `--ubatch-size` must exceed N. More draft tokens require more verification work; measure throughput and first-audio latency for your workload before increasing N.

The brain component must include a matching MTP head. A package without one still works with MTP disabled and fails explicitly if MTP is requested. Append the original model's unquantized MTP safetensors and configuration during brain export, then replace only that component in the package:

```sh
python examples/model-conversion/convert_frankie_brain.py /path/to/mlx-brain brain-mtp.gguf \
  --mtp-model /path/to/matching-hf-mtp
python tools/frankie/pack.py --base frankie.gguf --component brain=brain-mtp.gguf --output frankie-mtp.gguf
... /path/to/frankie-mtp.gguf 18793 --device gpu --mtp-tokens 2
```

The exporter reuses the Qwen converter's tensor mapping and defaults MTP matrices to Q8_0. `--mtp-type q4_1` reduces the draft head's memory without requantizing the brain or voice components; measure draft acceptance when choosing it. Matching dimensions alone do not establish weight compatibility; use the head from the brain's original checkpoint and check acceptance. MTP adds weights, its attention cache and one recurrent rollback slot per proposed token on the selected device. A rejected draft uses llama.cpp sequence rollback to retain the accepted prefix, including its verified hidden rows. Raw audio and image embeddings retain their separate target hidden states through draft prefill. Only verified answer rows reach the voice bridge; thinking and tool behavior use the existing parser and sampler.

For smaller devices, the standard quantizer can also reduce a vision component before packaging:

```sh
build-frankie/bin/llama-quantize vision.gguf vision-q8.gguf Q8_0
python tools/frankie/pack.py --base frankie-mtp.gguf --component vision=vision-q8.gguf --output frankie-mtp-q8-media.gguf
```

The quantizer retains tensors that require floating-point storage or have dimensions incompatible with the quantization block size. This changes only the selected vision component; the packer verifies that all other tensors and the packaged voice remain unchanged.

### Replace the brain with a fine-tune

For an ordinary HF BF16/FP16 checkpoint, use the standard converter rather than the MLX-specific exporter above. Pin the checkpoint revision, preserve its tokenizer and configuration, and merge any adapters into the intended base before conversion. Keep the checkpoint's own MTP tensors; do not pass `--no-mtp` or request a head-only export.

```sh
python convert_hf_to_gguf.py /path/to/hf-checkpoint --outtype bf16 --outfile brain-bf16.gguf
build-frankie/bin/llama-quantize --tensor-type '^blk\.64\..*=q8_0' \
  brain-bf16.gguf brain-q4_k_s-mtp-q8.gguf Q4_K_S
python tools/frankie/pack.py --base frankie.gguf \
  --component brain=brain-q4_k_s-mtp-q8.gguf --output frankie-finetune.gguf
```

The head pattern above is specific to this 64-layer brain. Q4_K_S leaves more device memory for context and streaming audio than Q4_K_M; choose the type using complete-system memory and quality measurements. The packer preserves the other components and packaged reference byte for byte.

A replacement must retain the supported architecture, vocabulary and token IDs, position encoding, 5120-dimensional hidden states and the trained bridge tap after layer 16. Compatible shapes alone do not establish speech quality or MTP acceptance. Test raw audio, images, speech, thinking, tools and interrupted turns, then compare MTP off and on at the intended context. A different Qwen architecture or changed hidden representation can require new runtime support or retrained bridges. Retain source and final-file hashes with the build commands for each package.

### Restore conversation history

Clients can restore conversation history with ordered `conversation.item.create` events before requesting a response. User messages accept `input_text` and inline `input_image`; historical assistant messages accept `output_text` or `output_audio` with a transcript. Assistant audio bytes are not imported. To preserve learned audio inputs across a new connection, replay the stored user PCM through `input_audio_buffer.append` and `input_audio_buffer.commit`, with automatic responses disabled, in its original history position. A transcript alone does not restore the learned audio rows. Live sessions retain their existing audio rows.

Historical `function_call` items require `name`, JSON-object `arguments`, and a unique `call_id`. Import each matching `function_call_output` before the next message or response. Consecutive calls belong to one assistant message, including parallel historical calls. Importing a call only stores history and acknowledges `conversation.item.created`; it never publishes a response tool-call event or executes a tool. Clients must not execute imported history acknowledgments.

```json
{"type":"conversation.item.create","item":{"id":"history_assistant","type":"message","role":"assistant","content":[{"type":"output_text","text":"I checked the result."}]}}
{"type":"conversation.item.create","item":{"id":"history_call","type":"function_call","call_id":"previous_call","name":"calculate","arguments":"{\"expression\":\"2+2\"}"}}
{"type":"conversation.item.create","item":{"type":"function_call_output","call_id":"previous_call","output":"4"}}
{"type":"conversation.item.retrieve","item_id":"history_assistant"}
```

Optional item IDs must be unique within the connection and contain at most 256 ASCII letters, digits, underscores or hyphens. Items append to the tail; `previous_item_id` may be omitted, null, or the current tail ID. Other insertion positions are rejected. The existing limits remain 16 KiB instructions, 32 tools, 16 KiB message text, 256 KiB tool arguments/results, 4096 retained messages, and 18 MiB per event. A connection admits at most 65536 item IDs. A new connection starts with empty history; explicit client compression/reset should reconnect and import the chosen history.

Clients that own conversation history can set `session.truncation` to `"disabled"`. A response that would exceed context, message, or audio-history capacity then fails before any old turn or tool output is removed. The default `"auto"` preserves the existing rollover behavior. After a capacity failure, reconnect and import the history selected by the client.

`conversation.item.retrieve` returns `conversation.item.retrieved` with `item`. After `conversation.item.truncate`, retrieve the assistant item to obtain the same heard-prefix transcript used by the model, including its interruption marker. This also works while audio is being cancelled. Completed transcripts reflect retained conversation state; trimmed or deleted items are no longer available. Audio bytes are not returned. Retain caller-owned user PCM separately when reconnect support is needed.

### Thinking

Thinking defaults to off. `--thinking` accepts `none` (or `off`), `minimal`, `low`, `medium`, `high`, `xhigh`, and `max`. Their reasoning budgets are 0, 128, 512, 2048, 8192, 16384, and 32768 tokens. The existing llama.cpp chat template and common sampler enforce the budget; reasoning does not feed the speech bridge or appear in spoken transcripts. More thinking can delay the first spoken answer substantially.

`--http-thinking` sets an independent default for `/v1/chat/completions`, with the same levels and budgets. It defaults to `none`. For example, `--thinking none --http-thinking high` keeps voice thinking off and enables it for HTTP chat. A request can override this default with `"reasoning_effort": "low"` or `"reasoning_effort": "none"`. The response separates reasoning into `reasoning_content`; `max_completion_tokens` (or `max_tokens`) still caps total generated tokens, including reasoning, so allow room for the answer. Explicit `enable_thinking: false` disables thinking. The existing boolean-only `enable_thinking: true` keeps its prior behavior when the server default is off: thinking is limited by the total output cap. Conflicting `reasoning_effort` and `enable_thinking` values are rejected. `/v1/completions` accepts a raw prompt and does not apply a chat thinking template.

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

A non-silent WAV up to 20 seconds replaces the packaged voice. Qwen3-TTS uses its existing native speaker encoder and clears the packaged ICL text and codes. Breeze uses the shared Mimi reference encoder and RVQ codebooks, plus the packaged Parakeet ear to transcribe the recording. It builds the reference prefix once at startup in the same process. A matching `--voice-text-file` can override Breeze's automatic reference transcription; omit `--voice-codes` for Breeze.

The GGUF must include its mouth's reference encoder. Older Breeze packages containing only cached conditioning need their mouth and conditioning assets rebuilt with the current converter before `--voice` can work. Breeze releases the reference encoder's weights and compute buffers after startup; only its voice prefix stays in memory. A complete package does not need rebuilding for each new reference, and native inference requires no Python service.

For Qwen3-TTS reference-text/codec conditioning as well as the speaker embedding, supply all three matching inputs:

```sh
# I32 file: little-endian, frame-major [frames,16], codec IDs 0..2047.
# These codes must come from this WAV, and the UTF-8 text must transcribe it.
... --voice my-voice.wav --voice-text-file my-voice.txt --voice-codes my-voice.i32
```

The Qwen3-TTS path does not yet encode arbitrary WAVs into ICL codec IDs. WAV-only speaker conditioning is supported; automatic generation of those optional Qwen ICL codes is separate work. Speaker caching uses exact PCM bytes, and expression offsets do not mutate the cached base voice.

### Breeze sentence continuity

Breeze retains the acoustic backbone's evaluated text/audio KV rows between completed speech chunks in one response. Each continuation appends an end-of-speech row and the new text, then uses the existing depth decoder and streaming codec. Previous speech is neither re-encoded nor replayed. This is a runtime change; compatible GGUF weights and custom voice references work without conversion. Qwen3-TTS keeps its existing behavior.

The default `--speech-context-words 100` bounds the retained speech history. Set it to `0` to disable reuse, or choose up to `1000` words. Before a chunk would exceed that word count or the existing 2048-row mouth context, the runtime starts again from the voice reference. It reserves room for the current chunk's maximum audio length. This refreshes whole chunks of history rather than shifting old KV rows. Completed history above 100,000,000 bytes of logical KV state is discarded. The native mouth reuses its fixed KV allocation, so enabling continuity does not allocate another cache; temporary inference buffers are separate from this retained-state limit.

New responses, interruptions, errors and backchannels clear the history. Packages without the Breeze end-of-speech conditioning row use independent chunks. Breeze expands short integer text into English words before its text encoder; conversation transcripts and interruption offsets retain the brain's original text.

## Quantization and package assembly

Each component can use the quantization supported by its existing converter and runtime. For example, a Q4 brain can be packaged with Q8_0 Parakeet, voice and vision matrices. Spatial convolutions, normalization and other unsupported or sensitive tensors retain their original types. Q8 describes the quantized matrices, not every tensor in the file; retain the package manifest to identify the exact types.

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
  --expression expression.gguf --turn turn.gguf
```

The custom MLX brain, vision and Parakeet exporters are in `examples/model-conversion/convert_frankie_{brain,vision,ear}.py`. The ear exporter accepts `MODEL OUTPUT` and uses the installed `parakeet_mlx` frontend; its optional middle `FIXTURE` argument preserves frozen filter/window exports. Use the existing `convert_hf_to_gguf.py` for Qwen3-TTS, once for the talker and once with `--mmproj` for the speaker/code/codec component. These source conversions need the Python dependencies of their reference models; native inference does not.

Bridge, tone, optional Side, VAD, expression, turn checkpoints and matching reference codes must come from the same source configuration. In particular, tone's emotion rows come from the original brain embedding. Do not substitute unrelated adapters or reference codec IDs.

Turn projection and backchannel prediction share one VAP-BC checkpoint and one encoder/attention pass. Convert the complete `maai-kyoto/vap_bc_en` state dictionary with both its VAP and BC heads; its five-sample downsampling kernel differs from plain VAP. The verified MIT checkpoint revision is `3ff203ce14de279045eb1b145a1dd24caa8f1c9d`, with SHA256 `54e5d19456c0ec7a6fb54ebbbceca837257d827bc86fd7820b0669f961cd11bc`. Keep its MIT license with the package. The converter requires PyTorch for `.pt` input and also accepts a complete F32 NPZ state dictionary.

```sh
python tools/frankie/convert-turn.py vap-bc_state_dict_en_10hz_20000msec.pt turn.gguf
python tools/frankie/pack.py --base existing.gguf --turn turn.gguf --output updated.gguf
```

The packer removes both legacy split turn assets when replacing them with `assets.turn.gguf`; it refuses to carry them into a new package without a combined replacement. The runtime can still read old packages, but those weights are not a source for new turn assets. To replace other components, pass `--base existing.gguf` and their replacement paths. The packer preserves unchanged tensors and metadata, verifies every written tensor against its input and writes a manifest. Failed assembly leaves an `.incomplete` file rather than a final GGUF.

### Breeze components

Breeze consumes the generated spoken text through its native Gemma text encoder. The brain's hidden states select expression; they do not supply the words. The implementation reuses the Qwen3 backbone, Gemma encoder, Qwen speech depth/decoder and Mimi reference-encoder graphs. Model metadata selects Breeze's depth normalization, rotary frequencies and RVQ configuration.

Convert the Breeze checkpoint, including its `audio_tokenizer/` directory, and supply its cached default voice conditioning:

```sh
python tools/frankie/convert-breeze.py /path/to/breeze-source breeze-q8 \
  --voice-conditioning /path/to/default-voice.npz --reference-rms "$reference_rms"
python tools/frankie/pack.py --base frankie.gguf --output frankie-breeze.gguf \
  --component talker=breeze-q8/talker.gguf \
  --component side=breeze-q8/encoder.gguf \
  --component mouth=breeze-q8/acoustic.gguf --breeze breeze-q8/conditioning.gguf
```

The NPZ contains the source model's `voice_prefix` array with shape `[rows, 2048]`; `reference_rms` is the measured linear RMS of that reference, in `(0, 1]`. These are model-building inputs, not requirements for a user supplying `--voice`. The converter includes the WAV encoder, codebooks and reference EOS embedding needed to replace that cached default. Eligible matrices use Q8_0; codec tensors retain source precision. Keep Breeze and Qwen-TTS packages separate and retain the source checkpoint's model license with the artifact.

File size alone cannot establish whether the complete system fits a 24 GiB GPU: KV/recurrent state, compute buffers, vision, audio and turn models must be counted together during execution. Validate the trained bridge seams when replacing the brain, including raw audio and image inputs, expression and MTP acceptance.

## Runtime behavior and tuning

Defaults: 131072 context tokens, Q4_0 KV with flash attention, 4 CPU threads, 128-token logical and physical prompt batches on CPU, or 512 on GPU. `--batch-size` and `--ubatch-size` allow tuning memory and prompt throughput. Context supports 4096..262144 tokens; cache types are `q4_0`, `q8_0` and `f16`.

Brain sampling is temperature .7, top-p .8, top-k 20, presence penalty 0. Side defaults to 0. The Qwen talker/code predictor use temperature .6, top-k 50, top-p 1 and the reference implementation's repetition penalty. Speech cuts at sentence boundaries, joins short openers, and bounds chunks to 50 words. The expression head pools brain layer-16 states and applies gated speaker-vector offsets, a .9 peak ceiling and a reference-relative RMS ceiling.

The small VAD graph runs on CPU so GPU work cannot block microphone transport. Brain, ear, voice, vision and turn inference follow the selected backend.

During speech, the existing phrase worker gives the mouth priority when emitted audio has less than 160 ms of lead. Brain decoding resumes once that lead is restored or the phrase finishes. This avoids audio starvation from competing decode graphs without delaying the first PCM chunk or adding a separate inference implementation.

The phrase queue copies only the token states for each spoken span. Side conditioning adds offsets through the existing Qwen TTS embedding cache, without a second vocabulary matrix. Small component graphs validate backend support once when allocated, and release their host weight copies after loading.

Input uses VAD, in-speech ear prefill with a 20-frame lag, and speculative generation after short silence. VAP gates publication, with a 1500 ms silence ceiling and stale-prediction fallback. Resumed speech restores recurrent state. False-start recovery can merge speech only when the input pause, reply age and heard audio are each below 700 ms, without crossing published tool calls. A later utterance stays separate even if the previous reply just began. Interruption history retains completed heard phrases rather than claiming exact word alignment.

Backchannels use the shared MaAI graph at 10 Hz, the actual rendered speaker PCM, and a temporary brain probe restricted to listener reactions. They have separate bounded audio events and never create an assistant turn or execute a tool. The probe uses an existing llama.cpp device checkpoint and attention-suffix rollback, then restores it after selection. Normal speech supersedes listener audio.

Default limits are 90 seconds per input utterance, 300 seconds of output audio per response, and 4096 answer tokens. Use `--max-utterance-seconds` (2..120), `--max-output-audio-seconds` (1..3600), and `--max-output-tokens N|inf` to change them. Realtime `session.max_output_tokens` also accepts a positive integer or `"inf"`. Reasoning has its separate budget. An overlong VAD capture is cleared, then capture resumes after silence; no partial user turn is published. Output exhaustion reports an incomplete response and keeps capture live. Immutable input checkpoints are shared with speculative work; recovery preserves the input prefix and excludes unspoken generation. There are at most 4096 history messages and 128 audio segments; history rolls over at safe boundaries within the actual context budget.

Inline PNG/JPEG input uses the existing vision and M-RoPE path: at most 512 KiB, four megapixels, a 4096-pixel edge and four images in a session. Audio is mono PCM16 at 24 kHz.

`FRANKIE_VERIFY_ROLLBACK=1` enables expensive byte-for-byte recurrent-state diagnostics. `FRANKIE_REPORT_MEMORY=1` reports retained model and compute buffers at session close. These are diagnostics, not substitutes for peak device-memory measurements under CUDA. The ordinary startup log also reports component memory and the allocated context.
