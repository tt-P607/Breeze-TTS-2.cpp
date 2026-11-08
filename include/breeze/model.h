#pragma once

#include "breeze/config.h"
#include "breeze/common.h"
#include "breeze/gguf_loader.h"
#include "breeze/tokenizer.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace breeze {

struct LoraAdapter {
    GGUFModel gg;
    float scale = 1.0f;

    ~LoraAdapter() { gg.free(); }
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
    // only used for the text encoder with te_cpu
    Backend cpu;
    bool te_cpu = false;
    GGUFModel gg;
    BreezeConfig cfg;
    Tokenizer tok;
    std::map<std::string, std::unique_ptr<LoraAdapter>> loras;
    std::map<std::string, std::vector<LoraWeight>> lora_stacks;
    std::vector<ActiveLora> active_loras;

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
