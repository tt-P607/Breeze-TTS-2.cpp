#include "breeze/model.h"
#include "breeze/text_encoder.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace breeze;

static void require(bool ok, const std::string & message) {
    if (!ok) throw std::runtime_error(message);
}

class TempDir {
public:
    TempDir() {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path_ = std::filesystem::temp_directory_path() /
                ("breeze-model-storage-test-" + std::to_string(stamp));
        std::filesystem::create_directories(path_);
    }

    ~TempDir() {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }

    const std::filesystem::path & path() const { return path_; }

private:
    std::filesystem::path path_;
};

struct ModelCleanup {
    BreezeModel & model;
    ~ModelCleanup() { model.free(); }
};

struct GGUFModelCleanup {
    GGUFModel & model;
    ~GGUFModelCleanup() { model.free(); }
};

struct FixtureTensor {
    ggml_tensor * tensor = nullptr;
    std::vector<uint8_t> bytes;
};

static FixtureTensor make_q8(ggml_context * ctx, const char * name, int k, int rows, int seed) {
    ggml_tensor * tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, k, rows);
    require(tensor != nullptr, std::string("could not allocate fixture tensor: ") + name);
    ggml_set_name(tensor, name);

    std::vector<float> source((size_t) k * rows);
    for (size_t i = 0; i < source.size(); i++) {
        const int value = (int) ((i * 17 + (size_t) seed * 29) % 113) - 56;
        source[i] = (float) value / 41.0f;
    }
    FixtureTensor result{tensor, std::vector<uint8_t>(ggml_nbytes(tensor))};
    const size_t written = ggml_quantize_chunk(GGML_TYPE_Q8_0, source.data(), result.bytes.data(),
                                                0, rows, k, nullptr);
    require(written == ggml_nbytes(tensor), std::string("could not quantize fixture tensor: ") + name);
    return result;
}

static FixtureTensor make_f32(ggml_context * ctx, const char * name,
                              int ne0, int ne1, const std::vector<float> & values) {
    ggml_tensor * tensor = ne1 == 1
        ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne0)
        : ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
    require(tensor != nullptr, std::string("could not allocate fixture tensor: ") + name);
    require(values.size() == (size_t) ne0 * ne1, std::string("bad fixture element count: ") + name);
    ggml_set_name(tensor, name);
    FixtureTensor result{tensor, std::vector<uint8_t>(values.size() * sizeof(float))};
    std::memcpy(result.bytes.data(), values.data(), result.bytes.size());
    return result;
}

static void write_fixture(const std::filesystem::path & path) {
    constexpr int k = 64;
    constexpr int qkv_rows = 256;
    constexpr int vocab = 8;
    constexpr int projected = 16;
    constexpr int layer_count = 2;

    ggml_init_params params{ggml_tensor_overhead() * 40 + 4096, nullptr, true};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> tensors(ggml_init(params), ggml_free);
    std::unique_ptr<gguf_context, decltype(&gguf_free)> file(gguf_init_empty(), gguf_free);
    require(tensors != nullptr && file != nullptr, "could not allocate tiny model fixture");

    std::vector<FixtureTensor> fixture_tensors;
    fixture_tensors.push_back(make_q8(tensors.get(), "bb.blk.0.attn_q.weight", k, qkv_rows, 1));
    fixture_tensors.push_back(make_q8(tensors.get(), "bb.blk.0.attn_k.weight", k, qkv_rows, 2));
    fixture_tensors.push_back(make_q8(tensors.get(), "bb.blk.0.attn_v.weight", k, qkv_rows, 3));

    std::vector<float> keep(7);
    for (size_t i = 0; i < keep.size(); i++) keep[i] = (float) (i + 1) / 7.0f;
    fixture_tensors.push_back(make_f32(tensors.get(), "misc.keep.weight", (int) keep.size(), 1, keep));

    fixture_tensors.push_back(make_q8(tensors.get(), "te.token_embd.weight", k, vocab, 4));

    std::vector<float> norm(k, 1.0f);
    fixture_tensors.push_back(make_f32(tensors.get(), "te.output_norm.weight", k, 1, norm));

    std::vector<float> projection((size_t) k * projected);
    for (size_t i = 0; i < projection.size(); i++) {
        const int value = (int) ((i * 11 + 5) % 67) - 33;
        projection[i] = (float) value / 37.0f;
    }
    fixture_tensors.push_back(make_f32(tensors.get(), "te.proj.weight", k, projected, projection));

    for (int layer = 0; layer < layer_count; layer++) {
        const std::string prefix = "te.blk." + std::to_string(layer) + ".";
        for (const char * name : {"attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
                                  "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"}) {
            std::vector<float> weights((size_t) k * k);
            const int seed = layer * 7 + (int) fixture_tensors.size();
            for (size_t i = 0; i < weights.size(); i++) {
                const int value = (int) ((i * 13 + (size_t) seed * 19) % 61) - 30;
                weights[i] = (float) value / 128.0f;
                if (i % (k + 1) == 0) weights[i] += 0.25f;
            }
            fixture_tensors.push_back(make_f32(tensors.get(), (prefix + name).c_str(), k, k, weights));
        }
        for (const char * name : {"attn_norm.weight", "attn_q_norm.weight", "attn_k_norm.weight",
                                  "post_attn_norm.weight", "ffn_norm.weight", "post_ffn_norm.weight"}) {
            fixture_tensors.push_back(make_f32(tensors.get(), (prefix + name).c_str(), k, 1, norm));
        }
    }

    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> host_buffer(
        ggml_backend_alloc_ctx_tensors_from_buft(tensors.get(), ggml_backend_cpu_buffer_type()),
        ggml_backend_buffer_free);
    require(host_buffer != nullptr, "could not allocate backend-backed fixture tensors");
    for (const FixtureTensor & fixture_tensor : fixture_tensors) {
        require(fixture_tensor.tensor->buffer == host_buffer.get(), "fixture tensor is missing its host backend buffer");
        ggml_backend_tensor_set(fixture_tensor.tensor, fixture_tensor.bytes.data(), 0, fixture_tensor.bytes.size());
        gguf_add_tensor(file.get(), fixture_tensor.tensor);
    }

    const std::string filename = path.u8string();
    require(gguf_write_to_file(file.get(), filename.c_str(), false), "could not write tiny model fixture");
}

static std::vector<uint8_t> tensor_bytes(ggml_tensor * tensor) {
    require(tensor != nullptr && tensor->buffer != nullptr, "tensor is not backend allocated");
    std::vector<uint8_t> bytes(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size());
    return bytes;
}

static void verify_qkv_packing(const std::filesystem::path & path, Backend & backend) {
    GGUFModel separate;
    GGUFModelCleanup separate_cleanup{separate};
    GGUFModel packed;
    GGUFModelCleanup packed_cleanup{packed};
    require(separate.load(path.u8string(), backend), "could not load separate QKV fixture");
    require(packed.load(path.u8string(), backend, nullptr, nullptr, true), "could not load packed QKV fixture");
    require(separate.buffer != nullptr && packed.buffer != nullptr && packed.buffer_packed != nullptr,
            "QKV storage buffers are missing");
    const size_t separate_bytes = ggml_backend_buffer_get_size(separate.buffer);
    const size_t packed_bytes_total = ggml_backend_buffer_get_size(packed.buffer) +
                                      ggml_backend_buffer_get_size(packed.buffer_packed);
    const size_t alignment_allowance = 2 * ggml_backend_get_alignment(backend.backend);
    require(packed_bytes_total <= separate_bytes + alignment_allowance,
            "QKV packing reserved storage in addition to the original projection buffers");

    constexpr std::array<const char *, 3> suffixes = {"attn_q.weight", "attn_k.weight", "attn_v.weight"};
    const std::string prefix = "bb.blk.0";
    const auto joined_it = packed.packed_qkv.find(prefix);
    require(joined_it != packed.packed_qkv.end() && joined_it->second != nullptr,
            "packed QKV tensor is missing for bb.blk.0");
    ggml_tensor * joined = joined_it->second;
    require(joined->type == GGML_TYPE_Q8_0 && joined->ne[0] == 64 && joined->ne[1] == 3 * 256,
            "packed QKV tensor has unexpected type or dimensions");

    const std::vector<uint8_t> packed_bytes = tensor_bytes(joined);
    size_t offset = 0;
    for (const char * suffix : suffixes) {
        const std::string name = prefix + "." + suffix;
        ggml_tensor * original = separate.get(name);
        ggml_tensor * optimized = packed.get(name);
        require(original->type == GGML_TYPE_Q8_0 && optimized->type == original->type &&
                ggml_are_same_shape(original, optimized), "packing changed an original QKV tensor shape or type");
        require(optimized->buffer == joined->buffer,
                "packed QKV projection does not share the combined tensor buffer");
        require(reinterpret_cast<uintptr_t>(optimized->data) ==
                    reinterpret_cast<uintptr_t>(joined->data) + offset,
                "packed QKV projection does not alias its expected byte range");

        const std::vector<uint8_t> before = tensor_bytes(original);
        const std::vector<uint8_t> after = tensor_bytes(optimized);
        require(before == after, "packing changed raw QKV tensor bytes: " + name);
        require(offset + after.size() <= packed_bytes.size(), "packed QKV tensor is too short");
        require(std::memcmp(packed_bytes.data() + offset, after.data(), after.size()) == 0,
                "combined QKV bytes differ from the original projection: " + name);
        offset += after.size();
    }
    require(offset == packed_bytes.size(), "combined QKV tensor contains unexpected trailing bytes");

    for (const auto & [name, original] : separate.tensors) {
        ggml_tensor * optimized = packed.find(name);
        require(optimized != nullptr && optimized->type == original->type &&
                ggml_are_same_shape(original, optimized) && tensor_bytes(original) == tensor_bytes(optimized),
                "packing changed original tensor data, type, or shape: " + name);
    }

    ggml_tensor * separate_keep = separate.get("misc.keep.weight");
    ggml_tensor * packed_keep = packed.get("misc.keep.weight");
    require(packed_keep->type == GGML_TYPE_F32 && ggml_are_same_shape(separate_keep, packed_keep) &&
            tensor_bytes(separate_keep) == tensor_bytes(packed_keep),
            "packing changed an unrelated tensor");

    packed.free();
    separate.free();
}

static void configure_tiny_text_encoder(BreezeModel & model) {
    model.cfg.te.hidden = 64;
    model.cfg.te.vocab = 8;
    model.cfg.te.n_layer = 2;
    model.cfg.te.n_head = 1;
    model.cfg.te.n_kv_head = 1;
    model.cfg.te.head_dim = 64;
    model.cfg.te.ffn = 64;
    model.cfg.te.sliding_window = 2;
    model.cfg.te.layer_is_full = {0, 1};
    model.cfg.te.rms_eps = 1e-6f;
    model.cfg.te.query_pre_attn_scalar = 64.0f;
    model.cfg.te.embed_scale = 0.75f;
}

static void verify_cpu_embedding_placement_and_output(const std::filesystem::path & path) {
    BreezeModel gpu_model;
    ModelCleanup gpu_cleanup{gpu_model};
    gpu_model.backend.init(true);
    require(gpu_model.backend.is_gpu, "GPU backend is unavailable");
    configure_tiny_text_encoder(gpu_model);
    require(gpu_model.gg.load(path.u8string(), gpu_model.backend), "could not load GPU text encoder fixture");

    BreezeModel cpu_embedding_model;
    ModelCleanup cpu_cleanup{cpu_embedding_model};
    cpu_embedding_model.backend.init(true);
    require(cpu_embedding_model.backend.is_gpu, "second GPU backend is unavailable");
    cpu_embedding_model.cpu.init(false);
    ggml_backend_cpu_set_n_threads(cpu_embedding_model.cpu.backend, 1);
    configure_tiny_text_encoder(cpu_embedding_model);
    require(cpu_embedding_model.gg.load(path.u8string(), cpu_embedding_model.backend,
                                        &cpu_embedding_model.cpu, "te.token_embd.weight"),
            "could not load CPU embedding fixture");
    cpu_embedding_model.te_embedding_cpu = true;

    ggml_tensor * cpu_embedding = cpu_embedding_model.gg.get("te.token_embd.weight");
    require(cpu_embedding->buffer == cpu_embedding_model.gg.buffer_cpu,
            "embedding table was not allocated on the CPU backend");
    require(cpu_embedding_model.gg.get("te.output_norm.weight")->buffer == cpu_embedding_model.gg.buffer &&
            cpu_embedding_model.gg.get("te.proj.weight")->buffer == cpu_embedding_model.gg.buffer,
            "text encoder projection weights did not remain on the GPU backend");
    require(!cpu_embedding_model.te_cpu && cpu_embedding_model.te_embedding_cpu,
            "CPU embedding mode unexpectedly moved the full text encoder to CPU");

    const std::vector<int> tokens = {1, 3, 5};
    const std::vector<float> expected = text_encoder_forward(gpu_model, tokens);
    const std::vector<float> actual = text_encoder_forward(cpu_embedding_model, tokens);
    require(expected.size() == actual.size(), "CPU embedding changed text encoder output dimensions");
    require(std::memcmp(expected.data(), actual.data(), expected.size() * sizeof(float)) == 0,
            "CPU embedding lookup changed text encoder output bits");
}

int main() {
    Backend probe;
    probe.init(true);
    if (!probe.is_gpu) {
        probe.free();
        std::cout << "SKIP: GPU backend is unavailable\n";
        return 77;
    }

    try {
        TempDir temp;
        const std::filesystem::path fixture = temp.path() / "model.gguf";
        write_fixture(fixture);

        verify_qkv_packing(fixture, probe);
        probe.free();
        verify_cpu_embedding_placement_and_output(fixture);
        std::cout << "model storage GPU regressions passed\n";
        return 0;
    } catch (const std::exception & error) {
        probe.free();
        std::cerr << "model storage regression failed: " << error.what() << '\n';
        return 1;
    }
}
