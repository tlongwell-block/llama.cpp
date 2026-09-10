#pragma once
#include "brain-session.h"
#include "mtmd-helper.h"
#include "mtmd-side.h"
#include "mtmd.h"
#include "sampling.h"

#include <functional>
#include <mutex>

class mouth_session {
    component                                           talker_source, mouth_source, side_source;
    llama_model_ptr                                     model;
    llama_context_ptr                                   ctx;
    mtmd::context_ptr                                   mctx;
    std::unique_ptr<mtmd_helper::gen_audio>               generator;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> side_weights{ nullptr, ggml_free };
    std::unique_ptr<mtmd_side>                          side;
    std::unique_ptr<mtmd_expression>                    expression;
    std::vector<float>                                  embeddings;
    mtmd::bitmap_ptr                                    speaker;
    std::vector<int32_t>                                codes;
    std::string                                         reference;
    frankie_options                                     options;
    float                                               reference_rms = 0.0f;
    std::mutex execution;
    struct plain_style { float gain; float rms_db; int frames; int emotion; bool aside; };
    std::vector<float> speak_impl(const brain_session::response &, const std::function<void(const float *, size_t)> &,
                                  const brain_session::stage_callback &, const plain_style *);
  public:
    std::atomic<bool> aside_cancelled{false};
    void warmup();
    void report_memory() const;
    void speak_backchannel(int reaction, bool strong, const std::function<void(const float *, size_t)> & on_audio);
    std::atomic<bool> cancelled{ false };
    explicit mouth_session(const std::string & package, const frankie_options & options = {});
    using audio_callback = std::function<void(const float *, size_t)>;
    std::vector<float> speak(const brain_session::response & response, const audio_callback & on_audio = {}, const brain_session::stage_callback & on_stage = {});
};
