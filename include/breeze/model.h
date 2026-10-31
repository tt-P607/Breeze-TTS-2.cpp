#pragma once

#include "breeze/config.h"
#include "breeze/common.h"
#include "breeze/gguf_loader.h"
#include "breeze/tokenizer.h"

#include <map>
#include <memory>
#include <string>

namespace breeze {

struct LoraAdapter {
    GGUFModel gg;
    float scale = 1.0f;

    ~LoraAdapter() { gg.free(); }
};

struct BreezeModel {
    Backend backend;
    GGUFModel gg;
    BreezeConfig cfg;
    Tokenizer tok;
    std::map<std::string, std::unique_ptr<LoraAdapter>> loras;
    LoraAdapter * active_lora = nullptr;

    bool load(const std::string & path, bool prefer_gpu);
    bool load_lora(const std::string & id, const std::string & path);
    bool set_lora(const std::string & id);
    void unload_lora(const std::string & id);
    void clear_lora();
    void free();

    ggml_tensor * w(const std::string & name) const { return gg.get(name); }
    ggml_tensor * wopt(const std::string & name) const { return gg.find(name); }
    // 计算底模投影，并叠加当前 LoRA 在这一层的增量。
    ggml_tensor * project(ggml_context * ctx, const std::string & weight_name, ggml_tensor * x) const;
};

}
