#pragma once
#include "mtmd-watermark.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// Frankie v9's AudioSeal watermark on final speech PCM, ported from MTPLX
// frankie/demo/{watermark,audioseal_stream}.py ("residual16", payload 0xF601).
// The 16 kHz generator never touches the 24 kHz speech itself: a causal copy is
// resampled down, watermarked, and only the learned residual is resampled up and
// added to the original float PCM, held back 30 samples (1.25 ms) to align.

// Public experiment version tag, never an account, user or voice identifier.
constexpr uint16_t frankie_watermark_payload = 0xF601;

// Causal rational FIR resampler, identical to upfirdn(firwin(61, 1/max(up, down),
// kaiser 5.0) * up) evaluated only where every input it needs has arrived.
class frankie_causal_resampler {
    int up, down;
    std::vector<double> taps, history;
    int64_t start = 0, consumed = 0, produced = 0;

    static double bessel_i0(double x) {
        double sum = 1, term = 1;
        for (int k = 1; k < 64; ++k) {
            term *= (x / (2 * k)) * (x / (2 * k));
            sum += term;
            if (term < sum * 1e-17) { break; }
        }
        return sum;
    }

  public:
    frankie_causal_resampler(int up, int down) : up(up), down(down), taps(61) {
        // scipy.signal.firwin(61, cutoff, window=("kaiser", 5.0)), scaled to unit DC gain.
        const double cutoff = 1.0 / std::max(up, down), beta = 5.0;
        double sum = 0;
        for (int n = 0; n < 61; ++n) {
            const double m = n - 30.0, x = cutoff * m;
            const double sinc = x == 0 ? 1.0 : std::sin(M_PI * x) / (M_PI * x);
            const double r = 2.0 * n / 60.0 - 1.0;
            taps[n] = cutoff * sinc * bessel_i0(beta * std::sqrt(1.0 - r * r)) / bessel_i0(beta);
            sum += taps[n];
        }
        for (auto & h : taps) { h = h / sum * up; }
    }

    const std::vector<double> & filter() const { return taps; }

    std::vector<double> push(const double * x, size_t n) {
        history.insert(history.end(), x, x + n);
        consumed += int64_t(n);
        const int64_t target = (consumed * up + down - 1) / down;
        std::vector<double> y;
        y.reserve(size_t(target - produced));
        for (int64_t m = produced; m < target; ++m) {
            const int64_t phase = m * down;
            double acc = 0;
            for (int64_t j = std::max<int64_t>(0, (phase - 60 + up - 1) / up); j * up <= phase; ++j) {
                if (j < start || j >= consumed) { throw std::runtime_error("resampler history underflow"); }
                acc += taps[size_t(phase - j * up)] * history[size_t(j - start)];
            }
            y.push_back(acc);
        }
        produced = target;
        // Keep the 61 newest inputs; no later output reaches further back.
        const int64_t keep = std::max<int64_t>(start, consumed - 61);
        history.erase(history.begin(), history.begin() + (keep - start));
        start = keep;
        return y;
    }
};

// Loads the converted generator once (convert-watermark.py) and keeps it on CPU,
// like MTPLX, so it never queues behind the brain or mouth on the GPU.
class frankie_watermark {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> weights{nullptr, ggml_free};
    std::unique_ptr<mtmd_watermark> model;
    std::vector<float> payload;

  public:
    explicit frankie_watermark(const std::string & path) {
        ggml_context * raw = nullptr;
        std::unique_ptr<gguf_context, decltype(&gguf_free)> meta(gguf_init_from_file(path.c_str(), {false, &raw}), gguf_free);
        weights.reset(raw);
        const auto frame = meta ? gguf_find_key(meta.get(), "wm.frame") : -1;
        if (!meta || !raw || frame < 0 || gguf_get_kv_type(meta.get(), frame) != GGUF_TYPE_UINT32 ||
            gguf_get_val_u32(meta.get(), frame) != mtmd_watermark::frame) {
            throw std::runtime_error("invalid AudioSeal watermark GGUF: " + path);
        }
        model = std::make_unique<mtmd_watermark>(raw, false);
        payload = model->message(frankie_watermark_payload);
    }

    // One response's stream (MTPLX ResidualResponse). Output never exceeds input; normal
    // completion conserves every sample; cancellation discards whatever is still held.
    class stream {
        mtmd_watermark & model;
        const std::vector<float> & payload;
        std::vector<float> state, pending, original, low_original;
        frankie_causal_resampler down{2, 3}, up{3, 2};
        size_t skip = 30;
        uint64_t received = 0, emitted = 0;
        bool closed = false;

        std::vector<float> generate(std::vector<float> low) {
            low_original.insert(low_original.end(), low.begin(), low.end());
            pending.insert(pending.end(), low.begin(), low.end());
            const size_t n = pending.size() / mtmd_watermark::frame * mtmd_watermark::frame;
            std::vector<float> marked(n);
            for (size_t i = 0; i < n; i += mtmd_watermark::frame) {
                model.process(state, payload, pending.data() + i, marked.data() + i);
            }
            pending.erase(pending.begin(), pending.begin() + n);
            return marked;
        }

        std::vector<double> residual(const std::vector<float> & marked) {
            if (marked.size() > low_original.size()) { throw std::runtime_error("watermark residual overrun"); }
            std::vector<double> res(marked.size());
            for (size_t i = 0; i < marked.size(); ++i) { res[i] = double(marked[i] - low_original[i]); }
            low_original.erase(low_original.begin(), low_original.begin() + marked.size());
            return up.push(res.data(), res.size());
        }

        std::vector<float> lower(const float * x, size_t n) {
            std::vector<double> in(x, x + n);
            const auto low = down.push(in.data(), in.size());
            return generate(std::vector<float>(low.begin(), low.end()));
        }

        std::vector<float> add(const std::vector<double> & res) {
            const size_t drop = std::min(skip, res.size());
            skip -= drop;
            const size_t n = std::min(res.size() - drop, original.size());
            std::vector<float> out(n);
            for (size_t i = 0; i < n; ++i) { out[i] = original[i] + float(res[drop + i]); }
            original.erase(original.begin(), original.begin() + n);
            emitted += n;
            return out;
        }

      public:
        explicit stream(frankie_watermark & owner) : model(*owner.model), payload(owner.payload), state(model.initial_state()) {}

        uint64_t input_samples() const { return received; }
        uint64_t output_samples() const { return emitted; }

        std::vector<float> push(const float * pcm, size_t n) {
            if (closed) { throw std::runtime_error("watermark stream finished"); }
            if (!std::all_of(pcm, pcm + n, [](float v) { return std::isfinite(v); })) {
                throw std::runtime_error("non-finite speech PCM");
            }
            original.insert(original.end(), pcm, pcm + n);
            received += n;
            return add(residual(lower(pcm, n)));
        }

        // Normal completion: advance both filters past the last source sample and
        // pad the final partial frame, then release every remaining original sample.
        std::vector<float> finish() {
            if (closed) { throw std::runtime_error("watermark stream finished"); }
            closed = true;
            const std::vector<float> zeros(96, 0.0f);
            auto out = add(residual(lower(zeros.data(), zeros.size())));
            const size_t n = pending.size();
            std::vector<float> tail;
            if (n) {
                pending.resize(mtmd_watermark::frame, 0.0f);
                tail.resize(mtmd_watermark::frame);
                model.process(state, payload, pending.data(), tail.data());
                tail.resize(n);
            }
            pending.clear();
            for (float v : add(residual(tail))) { out.push_back(v); }
            const std::vector<double> flush(64, 0.0);
            for (float v : add(up.push(flush.data(), flush.size()))) { out.push_back(v); }
            if (!original.empty() || emitted != received) { throw std::runtime_error("watermark flush lost samples"); }
            return out;
        }
    };
};
