# Experimental Frankie Realtime server

Local research server, not a stock llama-server mode. No model assets are downloaded or exported by the build.

```sh
cmake -S . -B build-frankie-metal -DLLAMA_BUILD_FRANKIE=ON
cmake --build build-frankie-metal --target llama-frankie-realtime -j2
FRANKIE_REALTIME_TOKEN=<local-secret-at-least-16-bytes> build-frankie-metal/bin/llama-frankie-realtime /path/to/frankie.gguf 18793
```

Binds loopback only. One connection owns transient session configuration, conversation items, audio buffers, and pending function calls. ACP permissions and MCP tool execution remain in the client. Native model objects execute explicit requests. No automatic reconnect or tool replay.

Current limitations: manual turns, full prefill each response, complete-clip audio generation, ASCII-only brain/talker alignment. Inline PNG/JPEG input uses the packaged vision model and mtmd M-RoPE evaluation; 512 KiB/image, 4 MP, 4096-pixel edge, four images/session. Input/session mutation during a response is rejected. This is not streaming playback, VAD or barge-in support. Existing real-package probes are integration evidence, not a full protocol-conformance claim.

The package reader presents each namespaced tensor set as its original GGUF without extracting component files. `general.tensor_inventory_complete` opts the user-model loader into respecting that complete tensor inventory: otherwise synthetic user-model initialization creates absent optional biases and asks the callback to load tensors the actual checkpoint never contained. Required tensors still fail closed. The hook is limited to user-model initialization, not normal file loading.

Brain sampling uses the non-thinking recipe (temperature 0.7, top-p 0.8, top-k 20, min-p 0, presence penalty 1.5, seed 42) through common_sampler. Ear/bridges/Side use CPU; brain, vision and mouth use the available GPU backend. The executable does not claim strict Metal ear parity or CUDA validation. Reference voice decoding and generated PCM reuse mtmd/miniaudio; output is mono PCM16 at 24 kHz.
