#pragma once
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Brain-led delivery: the mean of a phrase's final brain states, read against fixed
// directions (16 feelings, 24 words) with length calibration, chooses weights over
// learned Breeze instruction rows and a guidance strength. Port of MTPLX demo/delivery.py.
class frankie_delivery {
    struct axes {
        size_t n = 0;
        std::vector<float> V, mu, center, spread;
        std::vector<double> a, b, c, e;
        double ref = 0;
    };
    size_t width_ = 0;
    axes feelings, words;
    bool scaled = false;
    std::vector<double> fa, fb, fc, fe, wa, wb, wc, we, W, A, mu, sd;
    std::vector<float> rows; // [bank][16][2048]

    // Calibrated readings of n-token phrases: z = ((mean - mu) V^T - center) / spread.
    static std::vector<double> read_axes(const axes & d, const std::vector<float> & mean, size_t count) {
        const double r0 = 1.0 / std::sqrt(d.ref);
        const double r = std::max(1.0 / std::sqrt(double(std::max<size_t>(count, 1))), r0);
        std::vector<float> centred(mean.size());
        for (size_t j = 0; j < mean.size(); ++j) { centred[j] = mean[j] - d.mu[j]; }
        std::vector<double> z(d.n);
        for (size_t i = 0; i < d.n; ++i) {
            const double m = d.a[i] + d.b[i] * r, s = std::max(d.c[i] + d.e[i] * r, 1e-3);
            const double m0 = d.a[i] + d.b[i] * r0, s0 = d.c[i] + d.e[i] * r0;
            const double spread = d.spread[i] * s / s0;
            const double center = d.center[i] + d.spread[i] * m - spread * m0;
            const float * v = d.V.data() + i * mean.size();
            double dot = 0;
            for (size_t j = 0; j < mean.size(); ++j) { dot += double(centred[j]) * v[j]; }
            z[i] = (double(float(dot)) - center) / spread;
        }
        return z;
    }

  public:
    std::vector<std::string> bank, names;
    static constexpr size_t n_rows = 16;

    struct reading {
        std::vector<std::pair<size_t, double>> weights; // bank index, weight; heaviest first
        std::vector<float> rows;                        // n_rows x 2048
        float strength = 1.0f;
        std::vector<double> feelings;                   // 16 readings, rounded to 0.01
    };

    explicit frankie_delivery(std::vector<char> & bytes) {
        ggml_context * raw = nullptr;
        std::unique_ptr<gguf_context, decltype(&gguf_free)> meta(
            gguf_init_from_buffer(bytes.data(), bytes.size(), {false, &raw}), gguf_free);
        std::unique_ptr<ggml_context, decltype(&ggml_free)> weights(raw, ggml_free);
        if (!meta || !weights) { throw std::runtime_error("delivery asset"); }
        auto strings = [&](const char * key) {
            const auto id = gguf_find_key(meta.get(), key);
            if (id < 0 || gguf_get_kv_type(meta.get(), id) != GGUF_TYPE_ARRAY || gguf_get_arr_type(meta.get(), id) != GGUF_TYPE_STRING) {
                throw std::runtime_error(std::string("delivery metadata: ") + key);
            }
            std::vector<std::string> out;
            for (size_t i = 0; i < gguf_get_arr_n(meta.get(), id); ++i) { out.emplace_back(gguf_get_arr_str(meta.get(), id, i)); }
            return out;
        };
        bank = strings("delivery.bank");
        names = strings("delivery.axes");
        const auto key = gguf_find_key(meta.get(), "delivery.width");
        if (key < 0 || gguf_get_kv_type(meta.get(), key) != GGUF_TYPE_UINT32) { throw std::runtime_error("delivery width"); }
        width_ = gguf_get_val_u32(meta.get(), key);
        if (width_ != 5120 && width_ != 10240) { throw std::runtime_error("delivery width"); }
        auto tensor = [&](const std::string & name, ggml_type type, size_t count, bool required = true) -> const void * {
            auto * t = ggml_get_tensor(raw, ("delivery." + name).c_str());
            if (!t && !required) { return nullptr; }
            if (!t || t->type != type || size_t(ggml_nelements(t)) != count) { throw std::runtime_error("delivery tensor: " + name); }
            return t->data;
        };
        auto f32 = [&](const std::string & name, size_t count) {
            const auto * v = static_cast<const float *>(tensor(name, GGML_TYPE_F32, count));
            return std::vector<float>(v, v + count);
        };
        auto f64 = [&](const std::string & name, size_t count) {
            const auto * v = static_cast<const double *>(tensor(name, GGML_TYPE_F64, count));
            return std::vector<double>(v, v + count);
        };
        for (auto [d, prefix, n] : {std::make_tuple(&feelings, "feelings", size_t(16)), std::make_tuple(&words, "words", size_t(24))}) {
            const std::string p = prefix;
            d->n = n;
            d->V = f32(p + ".V", n * width_);
            d->mu = f32(p + ".mu", width_);
            d->center = f32(p + ".center", n);
            d->spread = f32(p + ".spread", n);
            d->a = f64(p + ".cal_a", n); d->b = f64(p + ".cal_b", n);
            d->c = f64(p + ".cal_c", n); d->e = f64(p + ".cal_e", n);
            d->ref = f64(p + ".cal_ref", 1)[0];
            if (!(d->ref >= 1)) { throw std::runtime_error("delivery calibration"); }
        }
        if (names.size() != 16 || bank.empty() || bank.size() > 64) { throw std::runtime_error("delivery bank"); }
        if (tensor("scale.fa", GGML_TYPE_F64, 16, false)) {
            scaled = true;
            fa = f64("scale.fa", 16); fb = f64("scale.fb", 16); fc = f64("scale.fc", 16); fe = f64("scale.fe", 16);
            wa = f64("scale.wa", 24); wb = f64("scale.wb", 24); wc = f64("scale.wc", 24); we = f64("scale.we", 24);
        }
        W = f64("adapter.W", 41 * bank.size());
        A = f64("adapter.A", 41);
        mu = f64("adapter.mu", 40);
        sd = f64("adapter.sd", 40);
        rows = f32("rows", bank.size() * n_rows * 2048);
        for (double x : sd) { if (!(x > 0)) { throw std::runtime_error("delivery adapter sd"); } }
    }

    size_t width() const { return width_; }

    // mean: the phrase's final brain states averaged over its count rows.
    reading read(const std::vector<float> & mean, size_t count) const {
        if (mean.size() != width_) { throw std::runtime_error("delivery state width"); }
        auto zf = read_axes(feelings, mean, count);
        auto zw = read_axes(words, mean, count);
        if (scaled) {
            for (size_t i = 0; i < 16; ++i) { zf[i] = fa[i] + (zf[i] - fc[i]) / fe[i] * fb[i]; }
            for (size_t i = 0; i < 24; ++i) { zw[i] = wa[i] + (zw[i] - wc[i]) / we[i] * wb[i]; }
        }
        std::vector<double> f(41, 1.0);
        for (size_t i = 0; i < 40; ++i) { f[i] = ((i < 16 ? zf[i] : zw[i - 16]) - mu[i]) / sd[i]; }
        const size_t n = bank.size();
        std::vector<double> p(n, 0.0);
        for (size_t i = 0; i < 41; ++i) { for (size_t j = 0; j < n; ++j) { p[j] += f[i] * W[i * n + j]; } }
        const double top = *std::max_element(p.begin(), p.end());
        double sum = 0;
        for (auto & x : p) { x = std::exp(x - top); sum += x; }
        for (auto & x : p) { x /= sum; }
        std::vector<size_t> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return p[a] > p[b]; });
        reading out;
        double total = 0;
        for (size_t j : order) { if (p[j] >= 0.05) { total += p[j]; } }
        out.rows.assign(n_rows * 2048, 0.0f);
        for (size_t j : order) {
            if (p[j] < 0.05) { continue; }
            const double w = std::nearbyint(p[j] / total * 1000) / 1000;
            out.weights.emplace_back(j, w);
            const float * r = rows.data() + j * n_rows * 2048;
            for (size_t k = 0; k < out.rows.size(); ++k) { out.rows[k] += float(w) * r[k]; }
        }
        double s = 0;
        for (size_t i = 0; i < 41; ++i) { s += f[i] * A[i]; }
        out.strength = float(std::nearbyint(std::clamp(s, 3.0, 4.0) * 2) / 2);
        for (double z : zf) { out.feelings.push_back(std::nearbyint(z * 100) / 100); }
        return out;
    }
};

// When the brain's feeling is extreme it outranks the held voice (readings on the adapter's scale):
//   enter: a phrase reads a feeling at hi or more that the held speech reads under hi - gap;
//   leave: a reply's first phrase reads under plain a feeling the held speech reads at hi or more.
// The held reading is the word-weighted mean over the phrases still in Breeze's window.
class frankie_hold {
    std::deque<std::pair<size_t, std::vector<double>>> held; // words, feelings; oldest first
  public:
    double hi = 5.5, gap = 2.0, plain = 3.0;
    struct reason { const char * rule; size_t axis; double reading, held; };

    void clear() { held.clear(); }

    // Forget phrases Breeze has evicted, oldest first.
    void follow(size_t context_words) {
        auto total = [&] { size_t n = 0; for (const auto & h : held) { n += h.first; } return n; };
        while (!held.empty() && total() > context_words) { held.pop_front(); }
    }

    bool why(const std::vector<double> & z, bool start, reason & out) const {
        if (held.empty()) { return false; }
        std::vector<double> c(z.size(), 0.0);
        size_t n = 0;
        for (const auto & [w, h] : held) { n += w; for (size_t i = 0; i < c.size(); ++i) { c[i] += double(w) * h[i]; } }
        for (auto & x : c) { x /= double(std::max<size_t>(1, n)); }
        const size_t j = std::max_element(z.begin(), z.end()) - z.begin();
        const size_t k = std::max_element(c.begin(), c.end()) - c.begin();
        if (z[j] >= hi && c[j] < hi - gap) { out = {"enter", j, z[j], c[j]}; return true; }
        if (start && c[k] >= hi && z[k] < plain) { out = {"leave", k, c[k], z[k]}; return true; }
        return false;
    }

    void add(size_t words, const std::vector<double> & z) { held.emplace_back(words, z); }
};
