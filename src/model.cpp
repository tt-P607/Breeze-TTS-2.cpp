#include "breeze/model.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <stdexcept>
#include <utility>
#include <algorithm>

namespace breeze {

bool BreezeModel::load(const std::string & path, bool prefer_gpu) {
    backend.init(prefer_gpu);
    if (!gg.load(path, backend)) return false;
    cfg = parse_config(gg);
    if (!tok.load(gg)) return false;
    return true;
}

bool BreezeModel::load_lora(const std::string & id, const std::string & path) {
    if (id.empty() || loras.count(id)) {
        fprintf(stderr, "invalid or duplicate LoRA ID: %s\n", id.c_str());
        return false;
    }

    auto adapter = std::make_unique<LoraAdapter>();
    if (!adapter->gg.load(path, backend)) {
        fprintf(stderr, "failed to load LoRA adapter: %s\n", path.c_str());
        return false;
    }

    if (adapter->gg.kv_str("breeze.lora.format") != "breeze-lora-v1") {
        fprintf(stderr, "unsupported LoRA adapter format: %s\n", path.c_str());
        return false;
    }
    adapter->scale = adapter->gg.kv_f32("breeze.lora.scale", 0.0f);
    if (!std::isfinite(adapter->scale) || adapter->scale <= 0.0f) {
        fprintf(stderr, "invalid LoRA adapter scale: %s\n", path.c_str());
        return false;
    }

    const std::string a_suffix = ".lora_a";
    const std::string b_suffix = ".lora_b";
    size_t count = 0;
    for (const auto & entry : adapter->gg.tensors) {
        const std::string & factor_name = entry.first;
        const bool is_a = factor_name.size() > a_suffix.size() &&
                          factor_name.compare(factor_name.size() - a_suffix.size(), a_suffix.size(), a_suffix) == 0;
        const bool is_b = factor_name.size() > b_suffix.size() &&
                          factor_name.compare(factor_name.size() - b_suffix.size(), b_suffix.size(), b_suffix) == 0;
        if (!is_a && !is_b) {
            fprintf(stderr, "unexpected tensor in LoRA adapter: %s\n", factor_name.c_str());
            return false;
        }

        const std::string base_name = factor_name.substr(
            0, factor_name.size() - (is_a ? a_suffix.size() : b_suffix.size()));
        ggml_tensor * base = gg.find(base_name);
        ggml_tensor * a = adapter->gg.find(base_name + a_suffix);
        ggml_tensor * b = adapter->gg.find(base_name + b_suffix);
        if (!base || !a || !b || !ggml_is_matrix(base) || !ggml_is_matrix(a) ||
            !ggml_is_matrix(b) || a->ne[0] != base->ne[0] || a->ne[1] != b->ne[0] ||
            b->ne[1] != base->ne[1] ||
            (a->type != GGML_TYPE_F16 && a->type != GGML_TYPE_F32) ||
            (b->type != GGML_TYPE_F16 && b->type != GGML_TYPE_F32)) {
            fprintf(stderr, "LoRA tensor shape or type mismatch for base tensor: %s\n", base_name.c_str());
            return false;
        }
        if (is_a) count++;
    }

    if (count == 0) {
        fprintf(stderr, "LoRA adapter contains no factor pairs: %s\n", path.c_str());
        return false;
    }

    printf("loaded LoRA %s (%zu projections, scale %.4g)\n", id.c_str(), count, adapter->scale);
    loras.emplace(id, std::move(adapter));
    return true;
}

bool BreezeModel::set_lora(const std::string & id) {
    const auto stack = lora_stacks.find(id);
    if (stack != lora_stacks.end()) {
        std::vector<ActiveLora> selected;
        for (const auto & item : stack->second)
            selected.push_back({loras.at(item.id).get(), item.weight});
        active_loras = std::move(selected);
        return true;
    }
    const auto it = loras.find(id);
    if (it == loras.end()) return false;
    active_loras = {{it->second.get(), 1.0f}};
    return true;
}

void BreezeModel::add_lora_stack(const std::string & id, const std::vector<LoraWeight> & weights) {
    for (const auto & item : weights) {
        if (!loras.count(item.id) || !std::isfinite(item.weight) || item.weight < 0)
            throw std::runtime_error("invalid LoRA stack member: " + item.id);
    }
    lora_stacks.emplace(id, weights);
}

ScopedLora::ScopedLora(BreezeModel & model, const std::string & id)
    : model(model), previous(model.active_loras) {
    if (id.empty()) model.clear_lora();
    else if (!model.set_lora(id)) throw std::runtime_error("unknown LoRA slot: " + id);
}

ScopedLora::~ScopedLora() {
    model.active_loras = std::move(previous);
}

void BreezeModel::unload_lora(const std::string & id) {
    lora_stacks.erase(id);
    const auto it = loras.find(id);
    if (it == loras.end()) return;
    active_loras.erase(std::remove_if(active_loras.begin(), active_loras.end(), [&](const ActiveLora & active) {
        return active.adapter == it->second.get();
    }), active_loras.end());
    loras.erase(it);
}

void BreezeModel::clear_lora() {
    active_loras.clear();
}

ggml_tensor * BreezeModel::project(ggml_context * ctx, const std::string & weight_name, ggml_tensor * x) const {
    ggml_tensor * out = ggml_mul_mat(ctx, gg.get(weight_name), x);
    for (const auto & active : active_loras) {
        if (active.weight == 0) continue;
        ggml_tensor * a = active.adapter->gg.find(weight_name + ".lora_a");
        if (!a) continue;
        ggml_tensor * b = active.adapter->gg.get(weight_name + ".lora_b");
        ggml_tensor * low_rank = ggml_mul_mat(ctx, a, x);
        ggml_tensor * delta = ggml_mul_mat(ctx, b, low_rank);
        out = ggml_add(ctx, out, ggml_scale(ctx, delta, active.adapter->scale * active.weight));
    }
    return out;
}

void BreezeModel::free() {
    clear_lora();
    lora_stacks.clear();
    loras.clear();
    gg.free();
    backend.free();
}

}
