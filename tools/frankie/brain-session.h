#pragma once
#include "brain-output.h"
#include "chat.h"
#include "clip-impl.h"
#include "component.h"
#include "llama-cpp.h"
#include "mtmd-ear.h"
#include "mtmd-helper.h"
#include "runtime-options.h"
#include "speculative.h"

#include <array>
#include <atomic>
#include <functional>
#include <map>
#include <mutex>

// Compute is serialized; HTTP sequences share weights with the voice worker.
class brain_session {
    friend class frankie_completions;
    friend class speech_stream;
  public:
    struct output_limit : std::runtime_error {
        explicit output_limit(const char * reason) : std::runtime_error(reason) {}
    };
    struct request;
  private:
    mutable std::recursive_mutex compute_mutex;
    mutable std::atomic<unsigned> compute_waiters{0};
    std::unique_lock<std::recursive_mutex> lock_compute() const {
        std::unique_lock<std::recursive_mutex> lock(compute_mutex, std::defer_lock);
        if (!lock.try_lock()) {
            ++compute_waiters;
            try { lock.lock(); }
            catch (...) { --compute_waiters; throw; }
            --compute_waiters;
        }
        return lock;
    }
    std::function<void()> background_step;
    std::atomic<bool> speech_active{false};
    std::atomic<int64_t> speech_buffer_until_us{0};
    std::atomic<bool> http_pending{false};
    component                                           encoder_source, bridge_source, brain_source, vision_source;
    std::unique_ptr<clip_ctx, decltype(&clip_free)>     encoder{ nullptr, clip_free };
    std::unique_ptr<mtmd_ear>                           ear;
    llama_model_ptr                                     model;
    mtmd::context_ptr                                   vision;

    llama_context_ptr         ctx;
    llama_context_ptr         draft_ctx;
    common_speculative_ptr    speculative;
    common_chat_templates_ptr templates;
    struct sequence_state {
        std::vector<uint8_t> target, draft, boundary;
        size_t size() const { return target.size() + draft.size() + boundary.size(); }
    };
    static constexpr llama_state_seq_flags branch_flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE;
    // Device storage is reused: one active branch, with its prefix retained.
    struct branch_state {
        std::vector<uint8_t> recurrent, boundary;
        int pos = 0;
    };
    void capture_branch(branch_state & state, int pos, llama_seq_id seq = 0);
    void restore_branch(const branch_state & state, llama_seq_id seq = 0);
    sequence_state capture_sequence();
    void restore_sequence(const sequence_state & state);
    void clear_sequence();
    int decode(const llama_batch & batch);
    void configure_thinking(common_params_sampling & sampling, const common_chat_params & formatted, int budget) const;
    std::shared_ptr<const sequence_state> checkpoint;
    // Only system/tool context may survive a connection; never cache user turns here.
    std::shared_ptr<const sequence_state> warm_checkpoint;
    std::string warm_prefix_text;
    int warm_pos = 0;
    size_t warm_used = 0;
    std::string checkpoint_prefix;
    int checkpoint_pos = 0;
    size_t checkpoint_used = 0;
    std::string cached_prefix;
    int cached_pos = 0;
    size_t cached_used = 0;
    bool cache_valid = false;
    frankie_options options;
    std::string partial_prefix, partial_marker;
    std::vector<float> partial_rows;
    int partial_pos = 0;
    size_t partial_used = 0;
    void decode_rows(const std::vector<float> & rows, size_t start, size_t end, int & pos, size_t & used);
    void prefill(const request & input, const std::string & prompt, int & pos, size_t & used,
                 const std::string & stop_marker, size_t stop_rows, const std::function<void(const char *)> & on_stage);
    void save_checkpoint(const std::string & prefix, int pos, size_t used);
    void                      decode_text(const std::string & text, int & pos, size_t & used);
  public:
    int64_t audio_lead_ms() const { return http_pending.load() ? 800 : 240; }
    size_t audio_row_limit() const { return (options.max_utterance_seconds * 1000 + 79) / 80 + 2; }
    size_t context_tokens() const { return options.voice_context_tokens(); }
    static constexpr size_t max_messages = 4096;
    static constexpr size_t max_audio_segments = 128;
    std::atomic<bool> cancelled{ false };
    explicit brain_session(const std::string & package, const frankie_options & options = {});
    std::vector<float> encode_audio(const std::vector<float> & pcm16k, std::string * transcript = nullptr);

    using image_ptr = std::shared_ptr<mtmd_bitmap>;
    image_ptr decode_image(const std::vector<unsigned char> & bytes);

    struct request {
        int                                      reasoning_budget = 0;
        uint32_t                                 answer_limit = frankie_options::default_output_tokens;
        size_t generation_tokens() const { return std::min(answer_limit ? answer_limit : 512u, 512u) + (reasoning_budget ? reasoning_budget + 16 : 0); }
        common_chat_templates_inputs              chat;
        std::map<std::string, image_ptr>          images;
        std::map<std::string, std::vector<float>> audio_rows;
    };

    struct response {
        brain_output    raw;
        common_chat_msg message;
        size_t content_offset = std::string::npos;
    };

    // Count formatted text and media rows before choosing a bounded history window.
    size_t prompt_tokens(const request & request, const std::map<std::string, size_t> & pending_audio) const;
    void reset(bool preserve_checkpoint = false);
    void report_memory() const;
    struct saved_state {
        sequence_state sequence;
        std::shared_ptr<const sequence_state> checkpoint;
        std::string checkpoint_prefix, cached_prefix, partial_prefix, partial_marker;
        std::vector<float> partial_rows;
        int checkpoint_pos, cached_pos, partial_pos;
        size_t checkpoint_used, cached_used, partial_used;
        bool cache_valid;
    };
    saved_state suspend_state();
    void restore_state(saved_state state);

    // Append older ear rows while capture is open. The final request keeps these exact rows.
    void precommit(const request & input, const std::string & marker, const std::vector<float> & rows, size_t count);
    struct listener_reaction { int index = 4; std::array<float, 5> probabilities{}; };
    listener_reaction probe_listener(const request & input);
    void finish_audio(const request & input, const std::string & marker, std::vector<float> & rows);
    // Prefill static system/tools without an assistant generation header.
    void warm_prefix(common_chat_templates_inputs input);
    // Called after parsing each decoded token. The callback must copy retained rows.
    using stage_callback = std::function<void(const char *)>;
    using stream_callback = std::function<void(const response &, bool)>;
    response generate(const request & request, const stream_callback & on_text = {}, const stage_callback & on_stage = {});
};
