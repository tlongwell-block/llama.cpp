# Experimental Frankie Realtime server

Local research server, not a stock llama-server mode. No model assets are downloaded or exported by the build.

```sh
# ICU development headers/library are required for NFC speech normalization.
cmake -S . -B build-frankie-metal -DLLAMA_BUILD_FRANKIE=ON # add -DICU_ROOT=<prefix> if needed
cmake --build build-frankie-metal --target llama-frankie-realtime -j2
FRANKIE_REALTIME_TOKEN=<local-secret-at-least-16-bytes> build-frankie-metal/bin/llama-frankie-realtime /path/to/frankie.gguf 18793
```

Binds loopback only. One connection owns transient session configuration, conversation items, audio buffers, and pending function calls. ACP permissions and MCP tool execution remain in the client. Native model objects execute explicit requests. No automatic reconnect or tool replay.

The experimental server supports VAD-driven input, in-speech ear processing and brain prefill, streamed speech, cancellation/barge-in, and cached-prefix reuse. Short silence starts speculative generation (80 ms rounded to a VAD frame); publication waits for ACP authorization. Resumed speech restores a saved recurrent state. Early rendered interruption can merge input within 700 ms, but never across a published tool call/result. History retains only completed heard phrases on interruption, not partial-phrase word alignment. FRANKIE_VERIFY_ROLLBACK=1 enables byte-for-byte state readback for diagnostics. Brain/talker alignment uses UTF-8 character spans and NFC-normalized talker text with original-input offsets, including combining marks and byte-split tokens. ICU is required only for this opt-in target; it does not change the generic llama tokenizer.

Bounds: one active connection, 200000 brain context tokens (padded by llama.cpp), 30-second output audio/item, 512 generated tokens/response, 4096 conversation messages, and 128 retained input audio segments. History is trimmed to the actual prompt budget at safe conversation boundaries. Audio/token exhaustion emits an incomplete response with no tool calls; the live ACP client keeps capture open and reports the limit. Other inference failures remain errors. Capture is bounded to 19 seconds per utterance. Full-tools audio understanding and long-history equivalence still need acceptance. Inline PNG/JPEG input uses the packaged vision model and mtmd M-RoPE evaluation; 512 KiB/image, 4 MP, 4096-pixel edge, four images/session. Existing real-package probes are integration evidence, not a full protocol-conformance claim.

The package reader presents each namespaced tensor set as its original GGUF without extracting component files. `general.tensor_inventory_complete` opts the user-model loader into respecting that complete tensor inventory: otherwise synthetic user-model initialization creates absent optional biases and asks the callback to load tensors the actual checkpoint never contained. Required tensors still fail closed. The hook is limited to user-model initialization, not normal file loading.

Brain sampling uses the non-thinking recipe (temperature 0.7, top-p 0.8, top-k 20, min-p 0, presence penalty 1.5, seed 42) through common_sampler. Serving requests GPU execution for ear, bridges, Side, VAD, brain, vision and mouth. Frozen component tests ran on CPU and Metal; this does not establish zero fallback for every graph or CUDA validation. Reference voice decoding and generated PCM reuse mtmd/miniaudio; output is mono PCM16 at 24 kHz.
