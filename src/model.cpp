#include "breeze/model.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace breeze {

static bool qkv_tensor_bytes(ggml_tensor * tensor, std::vector<uint8_t> & bytes) {
    if (!tensor || !tensor->buffer) return false;
    bytes.resize(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size());
    return true;
}

static bool qkv_cache_alloc(QkvProjectionCache & cache, Backend & backend,
                            ggml_type type_a, int64_t a_ne0, int64_t a_ne1,
                            const std::vector<uint8_t> & bytes_a,
                            int64_t b_ne0 = 0, int64_t b_ne1 = 0,
                            const std::vector<uint8_t> * bytes_b = nullptr) {
    cache.release();
    ggml_init_params params{ggml_tensor_overhead() * (bytes_b ? 2 : 1) + 4096, nullptr, true};
    cache.ctx = ggml_init(params);
    if (!cache.ctx) return false;

    cache.a = ggml_new_tensor_2d(cache.ctx, type_a, a_ne0, a_ne1);
    if (bytes_b) cache.b = ggml_new_tensor_2d(cache.ctx, GGML_TYPE_F16, b_ne0, b_ne1);
    if (!cache.a || (bytes_b && !cache.b) || ggml_nbytes(cache.a) != bytes_a.size() ||
        (bytes_b && ggml_nbytes(cache.b) != bytes_b->size())) {
        cache.release();
        return false;
    }
    cache.buffer = ggml_backend_alloc_ctx_tensors(cache.ctx, backend.backend);
    if (!cache.buffer) {
        cache.release();
        return false;
    }
    ggml_backend_tensor_set(cache.a, bytes_a.data(), 0, bytes_a.size());
    if (bytes_b) ggml_backend_tensor_set(cache.b, bytes_b->data(), 0, bytes_b->size());
    return true;
}

bool BreezeModel::load(const std::string & path, bool prefer_gpu, bool want_te_cpu) {
    backend.init(prefer_gpu);
    const char * qkv_env = std::getenv("BREEZE_FUSE_QKV");
    fuse_qkv = backend.is_gpu && qkv_env && strcmp(qkv_env, "1") == 0 &&
               std::string(backend.name()).find("Vulkan") == 0;
    const char * te = std::getenv("BREEZE_TE_CPU");
    if (te && *te && strcmp(te, "0") != 0) want_te_cpu = true;
    te_cpu = backend.is_gpu && want_te_cpu;
    const char * embedding = std::getenv("BREEZE_TE_EMBED_CPU");
    te_embedding_cpu = backend.is_gpu && !te_cpu && embedding && *embedding && strcmp(embedding, "0") != 0;
    if (te_cpu || te_embedding_cpu) {
        cpu.init(false);
        const char * nt = std::getenv("BREEZE_TE_THREADS");
        int n = te_cpu ? (nt ? std::atoi(nt) : (int) std::max(1u, std::thread::hardware_concurrency() / 2)) : 1;
        ggml_backend_cpu_set_n_threads(cpu.backend, std::max(1, n));
        if (te_cpu) printf("text encoder: cpu, %d threads\n", std::max(1, n));
        else printf("text embedding lookup: cpu, 1 thread; text encoder: gpu\n");
    }
    const char * cpu_prefix = te_cpu ? "te." : te_embedding_cpu ? "te.token_embd.weight" : nullptr;
    if (!gg.load(path, backend, cpu_prefix ? &cpu : nullptr, cpu_prefix, fuse_qkv)) return false;
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
    if (!adapter->gg.load(path, backend, te_cpu ? &cpu : nullptr, te_cpu ? "te." : nullptr)) {
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

bool BreezeModel::project_qkv(ggml_context * ctx, const std::string & prefix, ggml_tensor * x,
                              std::array<ggml_tensor *, 3> & outputs) {
    if (!fuse_qkv || !ctx || !x || x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) ||
        x->ne[1] != 1 || x->ne[2] != 1 || x->ne[3] != 1) {
        return false;
    }

    auto & base_entry = qkv_base_cache[prefix];
    if (!base_entry) base_entry = std::make_unique<QkvProjectionCache>();
    QkvProjectionCache & base_cache = *base_entry;
    if (!base_cache.initialized) {
        base_cache.initialized = true;
        const std::array<std::string, 3> base_names = {
            prefix + ".attn_q.weight", prefix + ".attn_k.weight", prefix + ".attn_v.weight",
        };
        std::array<ggml_tensor *, 3> base_weights{};
        for (size_t i = 0; i < base_weights.size(); ++i) base_weights[i] = gg.find(base_names[i]);
        if (!base_weights[0] || !base_weights[1] || !base_weights[2]) return false;
        const int64_t k = base_weights[0]->ne[0];
        if (k <= 0) return false;
        std::array<int64_t, 3> rows{};
        int64_t sum_rows = 0;
        // Keep each row's workgroup and reduction identical to the separate projections.
        for (size_t i = 0; i < base_weights.size(); ++i) {
            ggml_tensor * w = base_weights[i];
            if (!w->buffer || w->type != GGML_TYPE_Q8_0 || !ggml_is_contiguous(w) ||
                w->ne[0] != k || w->ne[1] <= 0 || w->ne[2] != 1 || w->ne[3] != 1 ||
                w->ne[1] > 8192 - sum_rows) {
                return false;
            }
            rows[i] = w->ne[1];
            sum_rows += rows[i];
        }
        if (k % 32 != 0 || x->ne[0] != k) return false;

        const auto shared = gg.packed_qkv.find(prefix);
        if (shared != gg.packed_qkv.end()) {
            base_cache.a = shared->second;
        } else {
            size_t total_bytes = 0;
            for (ggml_tensor * w : base_weights) total_bytes += ggml_nbytes(w);
            std::vector<uint8_t> packed(total_bytes);
            size_t offset = 0;
            for (ggml_tensor * w : base_weights) {
                const size_t size = ggml_nbytes(w);
                ggml_backend_tensor_get(w, packed.data() + offset, 0, size);
                offset += size;
            }
            if (!qkv_cache_alloc(base_cache, backend, GGML_TYPE_Q8_0, k, sum_rows, packed)) return false;
        }
        base_cache.k = k;
        base_cache.rows = rows;
        base_cache.supported = true;
    }
    if (!base_cache.supported) return false;

    std::vector<std::pair<const ActiveLora *, QkvProjectionCache *>> lora_caches;
    lora_caches.reserve(active_loras.size());
    for (const ActiveLora & active : active_loras) {
        if (active.weight == 0.0f) continue;
        if (!active.adapter) return false;

        auto & lora_entry = active.adapter->qkv_cache[prefix];
        if (!lora_entry) lora_entry = std::make_unique<QkvProjectionCache>();
        QkvProjectionCache & lora_cache = *lora_entry;
        if (!lora_cache.initialized) {
            lora_cache.initialized = true;
            const std::array<std::string, 3> base_names = {
                prefix + ".attn_q.weight", prefix + ".attn_k.weight", prefix + ".attn_v.weight",
            };
            std::array<ggml_tensor *, 3> a{}, b{};
            bool any = false;
            bool complete = true;
            for (size_t i = 0; i < a.size(); ++i) {
                a[i] = active.adapter->gg.find(base_names[i] + ".lora_a");
                b[i] = active.adapter->gg.find(base_names[i] + ".lora_b");
                any |= a[i] != nullptr || b[i] != nullptr;
                complete &= a[i] != nullptr && b[i] != nullptr;
            }
            if (!any) {
                lora_cache.k = base_cache.k;
                lora_cache.rows = base_cache.rows;
                lora_cache.supported = true;
            } else if (!complete) {
                lora_cache.supported = false;
            } else {
                const int64_t rank = a[0]->ne[1];
                const bool valid_rank = rank >= 4 && rank <= 256 && (rank & (rank - 1)) == 0 && 3 * rank < 1024;
                bool valid = valid_rank;
                for (size_t i = 0; i < a.size() && valid; ++i) {
                    valid = a[i]->type == GGML_TYPE_F16 && b[i]->type == GGML_TYPE_F16 &&
                            ggml_is_contiguous(a[i]) && ggml_is_contiguous(b[i]) &&
                            a[i]->ne[0] == base_cache.k && a[i]->ne[1] == rank &&
                            a[i]->ne[2] == 1 && a[i]->ne[3] == 1 &&
                            b[i]->ne[0] == rank && b[i]->ne[1] == base_cache.rows[i] &&
                            b[i]->ne[2] == 1 && b[i]->ne[3] == 1;
                }
                if (!valid) {
                    lora_cache.supported = false;
                } else {
                    std::array<std::vector<uint8_t>, 3> a_bytes, b_bytes;
                    std::vector<uint8_t> packed_a;
                    for (size_t i = 0; i < a.size(); ++i) {
                        if (!qkv_tensor_bytes(a[i], a_bytes[i]) || !qkv_tensor_bytes(b[i], b_bytes[i])) {
                            valid = false;
                            break;
                        }
                        packed_a.insert(packed_a.end(), a_bytes[i].begin(), a_bytes[i].end());
                    }
                    if (!valid) {
                        lora_cache.supported = false;
                    } else {
                        const size_t scalar_bytes = ggml_type_size(GGML_TYPE_F16);
                        const size_t source_row_bytes = (size_t)rank * scalar_bytes;
                        const size_t packed_row_bytes = (size_t)(3 * rank) * scalar_bytes;
                        const size_t total_rows = (size_t)(base_cache.rows[0] + base_cache.rows[1] + base_cache.rows[2]);
                        std::vector<uint8_t> packed_b(packed_row_bytes * total_rows, 0);
                        size_t output_row = 0;
                        for (size_t i = 0; i < b.size(); ++i) {
                            for (int64_t row = 0; row < base_cache.rows[i]; ++row) {
                                const size_t dst_offset = (output_row + row) * packed_row_bytes + i * source_row_bytes;
                                const size_t src_offset = (size_t)row * source_row_bytes;
                                std::memcpy(packed_b.data() + dst_offset, b_bytes[i].data() + src_offset, source_row_bytes);
                            }
                            output_row += (size_t)base_cache.rows[i];
                        }
                        if (qkv_cache_alloc(lora_cache, backend, GGML_TYPE_F16, base_cache.k, 3 * rank, packed_a,
                                            3 * rank, total_rows, &packed_b)) {
                            lora_cache.k = base_cache.k;
                            lora_cache.rows = base_cache.rows;
                            lora_cache.supported = true;
                        } else {
                            lora_cache.supported = false;
                        }
                    }
                }
            }
        }
        if (!lora_cache.supported || lora_cache.k != base_cache.k || lora_cache.rows != base_cache.rows) return false;
        if (lora_cache.a) lora_caches.emplace_back(&active, &lora_cache);
    }

    ggml_tensor * combined = ggml_mul_mat(ctx, base_cache.a, x);
    for (const auto & [active, lora_cache] : lora_caches) {
        ggml_tensor * low_rank = ggml_mul_mat(ctx, lora_cache->a, x);
        ggml_tensor * delta = ggml_mul_mat(ctx, lora_cache->b, low_rank);
        combined = ggml_add(ctx, combined,
                            ggml_scale(ctx, delta, active->adapter->scale * active->weight));
    }

    const int64_t q_rows = base_cache.rows[0];
    const int64_t k_rows = base_cache.rows[1];
    outputs[0] = ggml_view_1d(ctx, combined, q_rows, 0);
    outputs[1] = ggml_view_1d(ctx, combined, k_rows, (size_t)q_rows * sizeof(float));
    outputs[2] = ggml_view_1d(ctx, combined, base_cache.rows[2], (size_t)(q_rows + k_rows) * sizeof(float));
    return true;
}

void BreezeModel::free() {
    clear_lora();
    lora_stacks.clear();
    loras.clear();
    qkv_base_cache.clear();
    gg.free();
    if (cpu.backend) cpu.free();
    backend.free();
    fuse_qkv = false;
    te_embedding_cpu = false;
}

}
