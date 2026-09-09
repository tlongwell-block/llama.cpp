#pragma once
#include "brain-output.h"
#include "chat.h"
#include "clip-impl.h"
#include "component.h"
#include "llama-cpp.h"
#include "mtmd-ear.h"
#include "mtmd-helper.h"

#include <array>
#include <atomic>
#include <map>

// One worker owns this object; cancellation is the only cross-thread operation.
class brain_session {
    component                                           encoder_source, bridge_source, brain_source, vision_source;
    std::unique_ptr<clip_ctx, decltype(&clip_free)>     encoder{ nullptr, clip_free };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> weights{ nullptr, ggml_free };
    std::unique_ptr<mtmd_ear>                           ear;
    llama_model_ptr                                     model;
    mtmd::context_ptr                                   vision;

    struct capture {
        bool                    enabled = false;
        size_t                  calls   = 0;
        std::array<float, 5120> row{};
    } tap;

    llama_context_ptr         ctx;
    common_chat_templates_ptr templates;
    void                      decode_text(const std::string & text, int & pos, size_t & used);
  public:
    std::atomic<bool> cancelled{ false };
    explicit brain_session(const std::string & package);
    std::vector<float> encode_audio(const std::vector<float> & pcm16k);

    using image_ptr = std::shared_ptr<mtmd_bitmap>;
    image_ptr decode_image(const std::vector<unsigned char> & bytes);

    struct request {
        common_chat_templates_inputs              chat;
        std::map<std::string, image_ptr>          images;
        std::map<std::string, std::vector<float>> audio_rows;
    };

    struct response {
        brain_output    raw;
        common_chat_msg message;
    };

    response generate(const request & request);
};
