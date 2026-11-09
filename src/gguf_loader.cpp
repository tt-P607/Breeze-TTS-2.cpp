#include "breeze/gguf_loader.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>
#include <stdexcept>

namespace breeze {

#ifdef _WIN32
#define breeze_fseek _fseeki64
#else
#define breeze_fseek fseeko
#endif

bool GGUFModel::load(const std::string & path, Backend & be, Backend * cpu_be, const char * cpu_prefix, bool pack_qkv) {
    free();
    gguf_init_params gp{ /*no_alloc=*/true, /*ctx=*/&meta };
    gguf = gguf_init_from_file(path.c_str(), gp);
    if (!gguf) {
        free();
        return false;
    }

    // pointing the originals at cpu twins makes the gpu alloc below skip them, it only takes tensors with no data
    if (cpu_be && cpu_be->backend && cpu_prefix) {
        const size_t plen = strlen(cpu_prefix);
        const int64_t nt = gguf_get_n_tensors(gguf);
        ggml_init_params cp{ ggml_tensor_overhead() * (size_t) (nt + 1), nullptr, true };
        meta_cpu = ggml_init(cp);
        std::vector<std::pair<ggml_tensor *, ggml_tensor *>> twins;
        for (int64_t i = 0; i < nt; i++) {
            const char * name = gguf_get_tensor_name(gguf, i);
            if (strncmp(name, cpu_prefix, plen) != 0) continue;
            ggml_tensor * t = ggml_get_tensor(meta, name);
            ggml_tensor * d = ggml_dup_tensor(meta_cpu, t);
            ggml_set_name(d, name);
            twins.emplace_back(t, d);
        }
        if (!twins.empty()) {
            buffer_cpu = ggml_backend_alloc_ctx_tensors(meta_cpu, cpu_be->backend);
            if (!buffer_cpu) {
                free();
                return false;
            }
            for (auto & [t, d] : twins) { t->data = d->data; t->buffer = d->buffer; }
        }
    }

    if (pack_qkv) {
        struct QKVGroup {
            std::string prefix;
            std::array<ggml_tensor *, 3> parts;
            std::array<size_t, 3> offsets;
            ggml_tensor * packed = nullptr;
        };
        std::vector<QKVGroup> groups;
        const int64_t nt = gguf_get_n_tensors(gguf);
        const size_t alignment = std::max<size_t>(1, ggml_backend_get_alignment(be.backend));
        constexpr char q_suffix[] = ".attn_q.weight";

        for (int64_t i = 0; i < nt; i++) {
            const char * raw_name = gguf_get_tensor_name(gguf, i);
            const std::string q_name(raw_name);
            if (q_name.size() <= sizeof(q_suffix) - 1 ||
                q_name.compare(q_name.size() - (sizeof(q_suffix) - 1), sizeof(q_suffix) - 1, q_suffix) != 0) continue;
            const std::string prefix = q_name.substr(0, q_name.size() - (sizeof(q_suffix) - 1));
            if (prefix.rfind("bb.", 0) != 0 && prefix.rfind("dd.", 0) != 0) continue;

            std::array<ggml_tensor *, 3> parts = {
                ggml_get_tensor(meta, q_name.c_str()),
                ggml_get_tensor(meta, (prefix + ".attn_k.weight").c_str()),
                ggml_get_tensor(meta, (prefix + ".attn_v.weight").c_str()),
            };
            ggml_tensor * q = parts[0];
            if (!q || !parts[1] || !parts[2]) continue;
            bool eligible = true;
            int64_t rows = 0;
            for (ggml_tensor * t : parts) {
                eligible = eligible && t->type == GGML_TYPE_Q8_0 && ggml_n_dims(t) == 2 &&
                           ggml_is_contiguous(t) && t->ne[0] == q->ne[0] && t->ne[1] > 0 &&
                           t->data == nullptr && t->buffer == nullptr;
                if (t->ne[1] <= 0 || t->ne[1] > 8192 - rows) eligible = false;
                else rows += t->ne[1];
            }
            if (!eligible) continue;

            std::array<size_t, 3> sizes = { ggml_nbytes(parts[0]), ggml_nbytes(parts[1]), ggml_nbytes(parts[2]) };
            std::array<size_t, 3> offsets = { 0, sizes[0], sizes[0] + sizes[1] };
            if (offsets[1] % alignment != 0 || offsets[2] % alignment != 0) continue;
            groups.push_back({ prefix, parts, offsets, nullptr });
        }

        if (!groups.empty()) {
            ggml_init_params pp{ ggml_tensor_overhead() * (groups.size() + 1), nullptr, true };
            meta_packed = ggml_init(pp);
            if (!meta_packed) {
                free();
                return false;
            }
            for (QKVGroup & group : groups) {
                const int64_t rows = group.parts[0]->ne[1] + group.parts[1]->ne[1] + group.parts[2]->ne[1];
                group.packed = ggml_new_tensor_2d(meta_packed, GGML_TYPE_Q8_0, group.parts[0]->ne[0], rows);
                ggml_set_name(group.packed, (group.prefix + ".attn_qkv.weight").c_str());
                packed_qkv[group.prefix] = group.packed;
            }
            buffer_packed = ggml_backend_alloc_ctx_tensors(meta_packed, be.backend);
            if (buffer_packed) {
                for (QKVGroup & group : groups) {
                    for (size_t i = 0; i < group.parts.size(); i++) {
                        group.parts[i]->data = (uint8_t *) group.packed->data + group.offsets[i];
                        group.parts[i]->buffer = group.packed->buffer;
                    }
                }
                printf("QKV shared storage: %zu groups, %.1f MiB\n", groups.size(),
                       ggml_backend_buffer_get_size(buffer_packed) / (1024.0 * 1024.0));
            } else {
                ggml_free(meta_packed);
                meta_packed = nullptr;
                packed_qkv.clear();
            }
        }
    }

    buffer = ggml_backend_alloc_ctx_tensors(meta, be.backend);
    if (!buffer) {
        free();
        return false;
    }

    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        free();
        return false;
    }

    const size_t data_off = gguf_get_data_offset(gguf);
    const int64_t n = gguf_get_n_tensors(gguf);
    std::vector<uint8_t> buf;
    for (int64_t i = 0; i < n; i++) {
        const char * name = gguf_get_tensor_name(gguf, i);
        ggml_tensor * t = ggml_get_tensor(meta, name);
        const size_t off = data_off + gguf_get_tensor_offset(gguf, i);
        const size_t sz = ggml_nbytes(t);
        buf.resize(sz);
        if (breeze_fseek(f, (long long) off, SEEK_SET) != 0) {
            fclose(f);
            free();
            return false;
        }
        if (fread(buf.data(), 1, sz, f) != sz) {
            fclose(f);
            free();
            return false;
        }
        ggml_backend_tensor_set(t, buf.data(), 0, sz);
        tensors[name] = t;
    }
    fclose(f);
    return true;
}

void GGUFModel::free() {
    if (buffer) ggml_backend_buffer_free(buffer);
    if (buffer_cpu) ggml_backend_buffer_free(buffer_cpu);
    if (buffer_packed) ggml_backend_buffer_free(buffer_packed);
    if (meta_cpu) ggml_free(meta_cpu);
    if (meta_packed) ggml_free(meta_packed);
    buffer_cpu = nullptr;
    meta_cpu = nullptr;
    buffer_packed = nullptr;
    meta_packed = nullptr;
    if (meta) ggml_free(meta);
    if (gguf) gguf_free(gguf);
    tensors.clear();
    packed_qkv.clear();
    buffer = nullptr;
    meta = nullptr;
    gguf = nullptr;
}

ggml_tensor * GGUFModel::find(const std::string & name) const {
    auto it = tensors.find(name);
    return it == tensors.end() ? nullptr : it->second;
}

ggml_tensor * GGUFModel::get(const std::string & name) const {
    ggml_tensor * t = find(name);
    if (!t) throw std::runtime_error("missing tensor: " + name);
    return t;
}

bool GGUFModel::has(const std::string & name) const {
    return tensors.count(name) > 0;
}

int GGUFModel::kv_u32(const char * key, int def) const {
    const int64_t id = gguf_find_key(gguf, key);
    if (id < 0) return def;
    switch (gguf_get_kv_type(gguf, id)) {
        case GGUF_TYPE_UINT32: return (int) gguf_get_val_u32(gguf, id);
        case GGUF_TYPE_INT32:  return (int) gguf_get_val_i32(gguf, id);
        case GGUF_TYPE_UINT64: return (int) gguf_get_val_u64(gguf, id);
        case GGUF_TYPE_INT64:  return (int) gguf_get_val_i64(gguf, id);
        default: return def;
    }
}

float GGUFModel::kv_f32(const char * key, float def) const {
    const int64_t id = gguf_find_key(gguf, key);
    if (id < 0) return def;
    const gguf_type t = gguf_get_kv_type(gguf, id);
    if (t == GGUF_TYPE_FLOAT32) return gguf_get_val_f32(gguf, id);
    if (t == GGUF_TYPE_FLOAT64) return (float) gguf_get_val_f64(gguf, id);
    return def;
}

bool GGUFModel::kv_bool(const char * key, bool def) const {
    const int64_t id = gguf_find_key(gguf, key);
    if (id < 0) return def;
    if (gguf_get_kv_type(gguf, id) == GGUF_TYPE_BOOL) return gguf_get_val_bool(gguf, id);
    return def;
}

std::string GGUFModel::kv_str(const char * key, const std::string & def) const {
    const int64_t id = gguf_find_key(gguf, key);
    if (id < 0 || gguf_get_kv_type(gguf, id) != GGUF_TYPE_STRING) return def;
    return gguf_get_val_str(gguf, id);
}

std::vector<int> GGUFModel::kv_i32_array(const char * key) const {
    std::vector<int> out;
    const int64_t id = gguf_find_key(gguf, key);
    if (id < 0 || gguf_get_kv_type(gguf, id) != GGUF_TYPE_ARRAY) return out;
    const size_t n = gguf_get_arr_n(gguf, id);
    const void * data = gguf_get_arr_data(gguf, id);
    const gguf_type at = gguf_get_arr_type(gguf, id);
    out.resize(n);
    for (size_t i = 0; i < n; i++) {
        if (at == GGUF_TYPE_INT32) out[i] = ((const int32_t *) data)[i];
        else if (at == GGUF_TYPE_UINT32) out[i] = (int) ((const uint32_t *) data)[i];
    }
    return out;
}

std::vector<std::string> GGUFModel::kv_str_array(const char * key) const {
    std::vector<std::string> out;
    const int64_t id = gguf_find_key(gguf, key);
    if (id < 0 || gguf_get_kv_type(gguf, id) != GGUF_TYPE_ARRAY) return out;
    const size_t n = gguf_get_arr_n(gguf, id);
    out.reserve(n);
    for (size_t i = 0; i < n; i++) out.emplace_back(gguf_get_arr_str(gguf, id, i));
    return out;
}

}
