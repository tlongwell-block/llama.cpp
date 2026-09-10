#pragma once
#include "brain-session.h"
#include "mtmd-helper.h"
#include "mtmd-side.h"
#include "mtmd.h"
#include "sampling.h"

#include <functional>

class mouth_session {
    component                                           talker_source, mouth_source, side_source;
    llama_model_ptr                                     model;
    llama_context_ptr                                   ctx;
    mtmd::context_ptr                                   mctx;
    std::unique_ptr<mtmd_helper::gen_audio>               generator;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> side_weights{ nullptr, ggml_free };
    std::unique_ptr<mtmd_side>                          side;
    std::vector<float>                                  embeddings;
    mtmd::bitmap_ptr                                    speaker;
    std::vector<int32_t>                                codes;
    std::string                                         reference;
  public:
    std::atomic<bool> cancelled{ false };
    explicit mouth_session(const std::string & package);
    using audio_callback = std::function<void(const float *, size_t)>;
    std::vector<float> speak(const brain_session::response & response, const audio_callback & on_audio = {}, const brain_session::stage_callback & on_stage = {});
};
