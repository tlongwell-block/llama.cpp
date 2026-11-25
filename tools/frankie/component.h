#pragma once
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// Read a component's virtual GGUF directly from the parent tensor table.
class component {
    struct region {
        uint64_t begin, end, parent;
    };

    std::ifstream                                       file;
    std::ifstream                                       standalone;
    std::vector<char>                                   header;
    std::vector<region>                                 regions;
    std::unique_ptr<gguf_context, decltype(&gguf_free)> parent{ nullptr, gguf_free };
    std::unique_ptr<gguf_context, decltype(&gguf_free)> meta{ nullptr, gguf_free };
    uint64_t                                            file_size = 0, size_ = 0;

    void read_file(uint64_t offset, void * out, size_t n) {
        if (offset > file_size || n > file_size - offset) {
            throw std::runtime_error("package read outside file");
        }
        file.clear();
        file.seekg(offset);
        if (!file.read(static_cast<char *>(out), n)) {
            throw std::runtime_error("short package read");
        }
    }
  public:
    component(const std::string & path, const std::string & name, const std::string & override_path = "") : file(path, std::ios::binary | std::ios::ate) {
        if (!file || file.tellg() <= 0) {
            throw std::runtime_error("missing package");
        }
        file_size = file.tellg();
        parent.reset(gguf_init_from_file(path.c_str(), { true, nullptr }));
        if (!parent) {
            throw std::runtime_error("invalid package");
        }
        const auto version = gguf_find_key(parent.get(), "frankie.version");
        if (version < 0 || gguf_get_kv_type(parent.get(), version) != GGUF_TYPE_UINT32 ||
            gguf_get_val_u32(parent.get(), version) != 1) {
            throw std::runtime_error("package version");
        }
        if (!override_path.empty()) {
            standalone.open(override_path, std::ios::binary | std::ios::ate);
            if (!standalone || standalone.tellg() <= 0) { throw std::runtime_error("missing component override"); }
            size_ = standalone.tellg();
            meta.reset(gguf_init_from_file(override_path.c_str(), {true, nullptr}));
            if (!meta || gguf_get_data_offset(meta.get()) > size_ || gguf_get_data_offset(meta.get()) > 64 * 1024 * 1024) {
                throw std::runtime_error("invalid component override");
            }
            for (int64_t i = 0; i < gguf_get_n_tensors(meta.get()); ++i) {
                const uint64_t offset = gguf_get_tensor_offset(meta.get(), i);
                const uint64_t extent = size_ - gguf_get_data_offset(meta.get());
                if (offset > extent || gguf_get_tensor_size(meta.get(), i) > extent - offset) {
                    throw std::runtime_error("component override tensor out of bounds");
                }
            }
            return;
        }
        const std::string prefix   = "frankie." + name + ".";
        const auto        size_key = gguf_find_key(parent.get(), (prefix + "size").c_str());
        if (size_key < 0 || gguf_get_kv_type(parent.get(), size_key) != GGUF_TYPE_UINT64) {
            throw std::runtime_error("component size missing");
        }
        size_    = gguf_get_val_u64(parent.get(), size_key);
        auto hid = gguf_find_tensor(parent.get(), ("assets." + name + ".header").c_str());
        if (hid < 0 || gguf_get_tensor_type(parent.get(), hid) != GGML_TYPE_I8) {
            throw std::runtime_error("component header missing");
        }
        auto bytes = gguf_get_tensor_size(parent.get(), hid);
        if (!bytes || bytes > 64 * 1024 * 1024 || bytes > size_) {
            throw std::runtime_error("component header bounds");
        }
        header.resize(bytes);
        read_file(gguf_get_data_offset(parent.get()) + gguf_get_tensor_offset(parent.get(), hid), header.data(), bytes);
        meta.reset(gguf_init_from_buffer(header.data(), header.size(), { true, nullptr }));
        if (!meta || gguf_get_data_offset(meta.get()) != header.size()) {
            throw std::runtime_error("component header invalid");
        }
        for (int64_t i = 0; i < gguf_get_n_tensors(meta.get()); ++i) {
            auto id = gguf_find_tensor(parent.get(), (prefix + gguf_get_tensor_name(meta.get(), i)).c_str());
            if (id < 0 || gguf_get_tensor_type(parent.get(), id) != gguf_get_tensor_type(meta.get(), i) ||
                gguf_get_tensor_size(parent.get(), id) != gguf_get_tensor_size(meta.get(), i) ||
                !std::equal(gguf_get_tensor_ne(meta.get(), i), gguf_get_tensor_ne(meta.get(), i) + 4,
                            gguf_get_tensor_ne(parent.get(), id))) {
                throw std::runtime_error("component tensor schema mismatch");
            }
            uint64_t begin  = gguf_get_data_offset(meta.get()) + gguf_get_tensor_offset(meta.get(), i),
                     n      = gguf_get_tensor_size(meta.get(), i);
            uint64_t target = gguf_get_data_offset(parent.get()) + gguf_get_tensor_offset(parent.get(), id);
            if (begin < size_t(header.size()) || begin > size_ || n > size_ - begin || target > file_size ||
                n > file_size - target) {
                throw std::runtime_error("component tensor out of bounds");
            }
            regions.push_back({ begin, begin + n, target });
        }
        std::sort(regions.begin(), regions.end(), [](auto & a, auto & b) { return a.begin < b.begin; });
        uint64_t end = header.size();
        for (auto & r : regions) {
            if (r.begin < end) {
                throw std::runtime_error("overlapping component tensors");
            }
            end = r.end;
        }
    }

    bool has_asset(const std::string & name) const { return gguf_find_tensor(parent.get(), name.c_str()) >= 0; }

    std::vector<char> asset(const std::string & name, size_t limit = 16 * 1024 * 1024) {
        auto id = gguf_find_tensor(parent.get(), name.c_str());
        if (id < 0) {
            throw std::runtime_error("missing asset");
        }
        size_t n = gguf_get_tensor_size(parent.get(), id);
        if (n > limit) {
            throw std::runtime_error("asset too large");
        }
        std::vector<char> result(n);
        read_file(gguf_get_data_offset(parent.get()) + gguf_get_tensor_offset(parent.get(), id), result.data(), n);
        return result;
    }

    std::string string_value(const char * key) {
        auto i = gguf_find_key(parent.get(), key);
        if (i < 0 || gguf_get_kv_type(parent.get(), i) != GGUF_TYPE_STRING) {
            throw std::runtime_error("missing string");
        }
        return gguf_get_val_str(parent.get(), i);
    }

    gguf_context * metadata() {
        gguf_set_val_bool(meta.get(), "general.tensor_inventory_complete", true);
        return meta.get();
    }

    uint64_t size() const { return size_; }

    size_t read(void * out, uint64_t offset, size_t n) {
        if (offset > size_ || n > size_ - offset) {
            return 0;
        }
        if (standalone.is_open()) {
            standalone.clear();
            standalone.seekg(offset);
            return standalone.read(static_cast<char *>(out), n) ? n : 0;
        }
        auto *       dst   = static_cast<char *>(out);
        const size_t total = n;
        if (offset < header.size()) {
            size_t k = std::min<uint64_t>(n, header.size() - offset);
            std::memcpy(dst, header.data() + offset, k);
            dst += k;
            offset += k;
            n -= k;
        }
        while (n) {
            auto it = std::upper_bound(regions.begin(), regions.end(), offset,
                                       [](uint64_t p, const region & r) { return p < r.end; });
            if (it == regions.end()) {
                std::memset(dst, 0, n);
                break;
            }
            if (offset < it->begin) {
                size_t k = std::min<uint64_t>(n, it->begin - offset);
                std::memset(dst, 0, k);
                dst += k;
                offset += k;
                n -= k;
                continue;
            }
            size_t k = std::min<uint64_t>(n, it->end - offset);
            read_file(it->parent + offset - it->begin, dst, k);
            dst += k;
            offset += k;
            n -= k;
        }
        return total;
    }

    static size_t callback(void * user, void * out, uint64_t offset, size_t n) {
        try {
            return static_cast<component *>(user)->read(out, offset, n);
        } catch (...) {
            return 0;
        }
    }

    static void set_tensor(ggml_tensor * t, void * user) {
        auto & c = *static_cast<component *>(user);
        auto   i = gguf_find_tensor(c.meta.get(), t->name);
        if (i < 0 || gguf_get_tensor_size(c.meta.get(), i) != ggml_nbytes(t) ||
            gguf_get_tensor_type(c.meta.get(), i) != t->type) {
            throw std::runtime_error(std::string("tensor load mismatch: ") + t->name + " id=" + std::to_string(i) +
                                     " bytes=" + std::to_string(ggml_nbytes(t)) + " type=" + std::to_string(t->type));
        }
        uint64_t          offset = gguf_get_data_offset(c.meta.get()) + gguf_get_tensor_offset(c.meta.get(), i);
        std::vector<char> buf(std::min<size_t>(ggml_nbytes(t), 1024 * 1024));
        for (size_t done = 0; done < ggml_nbytes(t);) {
            size_t n = std::min(buf.size(), ggml_nbytes(t) - done);
            if (c.read(buf.data(), offset + done, n) != n) {
                throw std::runtime_error("tensor read failed");
            }
            ggml_backend_tensor_set(t, buf.data(), done, n);
            done += n;
        }
    }
};
