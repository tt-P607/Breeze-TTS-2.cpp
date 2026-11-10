#pragma once

#include "breeze/config.h"
#include "breeze/common.h"
#include "breeze/gguf_loader.h"
#include "breeze/tokenizer.h"

#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace breeze {

struct QkvProjectionCache {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor * a = nullptr;
    ggml_tensor * b = nullptr;
    std::array<int64_t, 3> rows{};
    int64_t k = 0;
    bool initialized = false;
    bool supported = false;

    ~QkvProjectionCache() { release(); }
    void release() {
        if (buffer) ggml_backend_buffer_free(buffer);
        if (ctx) ggml_free(ctx);
        buffer = nullptr;
        ctx = nullptr;
        a = nullptr;
        b = nullptr;
    }
};

struct LoraAdapter {
    GGUFModel gg;
    float scale = 1.0f;
    std::map<std::string, std::unique_ptr<QkvProjectionCache>> qkv_cache;

    ~LoraAdapter() { qkv_cache.clear(); gg.free(); }
};

struct LoraWeight {
    std::string id;
    float weight = 1.0f;
};

struct ActiveLora {
    LoraAdapter * adapter;
    float weight;
};

struct BreezeModel {
    Backend backend;
    // only used to offload text-encoder weights or embedding lookup
    Backend cpu;
    bool te_cpu = false;
    bool te_embedding_cpu = false;
    GGUFModel gg;
    BreezeConfig cfg;
    Tokenizer tok;
    std::map<std::string, std::unique_ptr<LoraAdapter>> loras;
    std::map<std::string, std::vector<LoraWeight>> lora_stacks;
    std::vector<ActiveLora> active_loras;
    std::map<std::string, std::unique_ptr<QkvProjectionCache>> qkv_base_cache;
    bool fuse_qkv = false;

    // BREEZE_TE_CPU=1 opts into CPU text encoding for the C API and bindings too.
    bool load(const std::string & path, bool prefer_gpu, bool te_cpu = false);
    bool load_lora(const std::string & id, const std::string & path);
    bool set_lora(const std::string & id);
    void add_lora_stack(const std::string & id, const std::vector<LoraWeight> & weights);
    void unload_lora(const std::string & id);
    void clear_lora();
    void free();

    ggml_tensor * w(const std::string & name) const { return gg.get(name); }
    ggml_tensor * wopt(const std::string & name) const { return gg.find(name); }
    // 计算底模投影，并叠加当前 LoRA 在这一层的增量。
    ggml_tensor * project(ggml_context * ctx, const std::string & weight_name, ggml_tensor * x) const;
    bool project_qkv(ggml_context * ctx, const std::string & prefix, ggml_tensor * x,
                     std::array<ggml_tensor *, 3> & outputs);
};

// Generation runs on the scheduler thread; a scope selects only that job's adapter.
class ScopedLora {
public:
    ScopedLora(BreezeModel & model, const std::string & id);
    ~ScopedLora();
    ScopedLora(const ScopedLora &) = delete;
    ScopedLora & operator=(const ScopedLora &) = delete;

private:
    BreezeModel & model;
    std::vector<ActiveLora> previous;
};

}
