#include "models.h"
#include "json.h"
#include "server.h"

#include "breeze/model.h"
#include "breeze/scheduler.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <future>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

using namespace breeze;

static void require(bool ok, const std::string & message) {
    if (!ok) throw std::runtime_error(message);
}

static const JVal & member(const JVal & object, const char * key) {
    const auto it = object.obj.find(key);
    if (object.kind != JVal::Obj || it == object.obj.end())
        throw std::runtime_error(std::string("missing JSON member: ") + key);
    return it->second;
}

static std::string json_quote(const std::string & value) {
    std::string result = "\"";
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') result += '\\';
        result += (char) c;
    }
    result += '"';
    return result;
}

struct CacheState {
    int capacity = 0;
    int loaded = 0;
    int temporary_loaded = 0;
    int weighted_compositions = 0;
    std::map<std::string, bool> loaded_ids;

    int resident() const { return loaded - temporary_loaded; }
};

static CacheState cache_state(const Models & models) {
    JVal root;
    require(JParser(models.json()).parse(root), "models status returned invalid JSON");
    const JVal & cache = member(root, "lora_cache");
    CacheState result;
    result.capacity = (int) member(cache, "capacity").num;
    result.loaded = (int) member(cache, "loaded").num;
    result.temporary_loaded = (int) member(cache, "temporary_loaded").num;
    result.weighted_compositions = (int) member(cache, "weighted_compositions").num;
    for (const auto & row : member(root, "data").arr)
        result.loaded_ids[member(row, "id").str] = member(row, "loaded").b;
    return result;
}

class TempDir {
public:
    TempDir() {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        m_path = std::filesystem::temp_directory_path() /
                 ("breeze-lora-cache-test-" + std::to_string(stamp));
        std::filesystem::create_directories(m_path);
    }

    ~TempDir() {
        std::error_code ignored;
        std::filesystem::remove_all(m_path, ignored);
    }

    const std::filesystem::path & path() const { return m_path; }

private:
    std::filesystem::path m_path;
};

static void write_adapter_fixture(const std::filesystem::path & path) {
    ggml_init_params params{1024 * 1024, nullptr, false};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> tensors(ggml_init(params), ggml_free);
    std::unique_ptr<gguf_context, decltype(&gguf_free)> file(gguf_init_empty(), gguf_free);
    require(tensors != nullptr && file != nullptr, "could not allocate tiny GGUF fixture");

    constexpr const char * base = "bb.blk.0.attn_q.weight";
    const std::string a_name = std::string(base) + ".lora_a";
    const std::string b_name = std::string(base) + ".lora_b";
    ggml_tensor * a = ggml_new_tensor_2d(tensors.get(), GGML_TYPE_F32, 2, 1);
    ggml_tensor * b = ggml_new_tensor_2d(tensors.get(), GGML_TYPE_F32, 1, 2);
    require(a != nullptr && b != nullptr, "could not allocate tiny LoRA tensors");
    ggml_set_name(a, a_name.c_str());
    ggml_set_name(b, b_name.c_str());
    const float a_data[] = {1.0f, 0.0f};
    const float b_data[] = {0.0f, 1.0f};
    std::memcpy(a->data, a_data, sizeof(a_data));
    std::memcpy(b->data, b_data, sizeof(b_data));

    gguf_set_val_str(file.get(), "breeze.lora.format", "breeze-lora-v1");
    gguf_set_val_f32(file.get(), "breeze.lora.scale", 1.0f);
    gguf_add_tensor(file.get(), a);
    gguf_add_tensor(file.get(), b);
    require(gguf_write_to_file(file.get(), path.u8string().c_str(), false),
            "could not write tiny LoRA GGUF fixture");
}

static void init_test_model(BreezeModel & model) {
    model.backend.init(false);
    ggml_init_params params{ggml_tensor_overhead() + 4096, nullptr, true};
    model.gg.meta = ggml_init(params);
    require(model.gg.meta != nullptr, "could not allocate test base metadata");
    constexpr const char * base = "bb.blk.0.attn_q.weight";
    ggml_tensor * weight = ggml_new_tensor_2d(model.gg.meta, GGML_TYPE_F32, 2, 2);
    require(weight != nullptr, "could not allocate test base tensor");
    ggml_set_name(weight, base);
    model.gg.tensors.emplace(base, weight);
}

struct ModelCleanup {
    BreezeModel & model;
    ~ModelCleanup() { model.free(); }
};

static Models::Selection select_id(Models & models, const std::string & id) {
    Models::Selection selection;
    int status = 0;
    std::string error;
    require(models.select(true, id, false, "", selection, status, error),
            "could not select registered LoRA " + id + ": " + error);
    return selection;
}

static std::shared_ptr<Models::Lease> acquire(Models & models, const Models::Selection & selection) {
    int status = 0;
    std::string error;
    auto lease = models.acquire(selection, status, error);
    require(lease != nullptr, "could not acquire LoRA lease: " + error);
    require(status == 200, "LoRA acquire returned an unexpected status");
    return lease;
}

struct AcquireOutcome {
    std::shared_ptr<Models::Lease> lease;
    int status = 0;
    std::string error;
};

static std::shared_ptr<Models::Lease> acquire_promptly(Models & models, const Models::Selection & selection) {
    std::promise<AcquireOutcome> promise;
    std::future<AcquireOutcome> future = promise.get_future();
    std::atomic<bool> cancelled{false};
    std::thread worker([&] {
        AcquireOutcome outcome;
        try {
            outcome.lease = models.acquire(selection, outcome.status, outcome.error,
                                           [&] { return cancelled.load(); });
        } catch (const std::exception & e) {
            outcome.status = 500;
            outcome.error = e.what();
        }
        promise.set_value(std::move(outcome));
    });

    const bool prompt = future.wait_for(std::chrono::seconds(2)) == std::future_status::ready;
    if (!prompt) {
        cancelled.store(true);
        (void) future.wait_for(std::chrono::seconds(1));
        worker.join();
        throw std::runtime_error("adapter acquisition waited for a cache slot");
    }
    AcquireOutcome outcome = future.get();
    worker.join();
    require(outcome.lease != nullptr, "prompt LoRA acquisition failed: " + outcome.error);
    require(outcome.status == 200, "prompt LoRA acquire returned an unexpected status");
    return outcome.lease;
}

static void test_options() {
    BreezeModel model;
    GenScheduler scheduler;
    Models models(model, scheduler, "base", {}, 3, 900);
    Models::Selection selected;
    int status;
    std::string error;
#ifdef _WIN32
    const std::string role = "E:/models/role.gguf";
    const std::string effect = "E:/models/asmr.gguf";
    const std::string third = "E:/models/third.gguf";
    const std::string fourth = "E:/models/fourth.gguf";
#else
    const std::string role = "/models/role.gguf";
    const std::string effect = "/models/asmr.gguf";
    const std::string third = "/models/third.gguf";
    const std::string fourth = "/models/fourth.gguf";
#endif
    require(models.select(true, "role", true, role, selected, status, error), "role selection failed");
    const auto parse = [&](const std::string & text) {
        JVal body;
        require(JParser(text).parse(body), "invalid test JSON");
        return models.configure(body, selected, status, error);
    };
    require(parse("{}") && selected.weight == 1.0f && selected.extra_loras.empty(), "default weight changed");
    require(parse("{\"lora_weight\":0,\"extra_loras\":[{\"lora_path\":\"" + effect + "\"}]}") &&
            selected.weight == 0 && selected.extra_loras[0].weight == 1.0f, "zero or extra default lost");
    require(parse("{\"lora_weight\":0.9,\"extra_loras\":[{\"lora_path\":\"" + effect +
                  "\",\"weight\":1.8}]}") && selected.extra_loras[0].weight == 1.8f,
            "weight above one rejected");
    for (const auto * bad : {"-1", "null", "true", "\"1.8\"", "1e999"})
        require(!parse(std::string("{\"lora_weight\":") + bad + "}"), "invalid weight accepted");
    require(!parse("{\"extra_loras\":{}}"), "non-array accepted");
    require(!parse("{\"extra_loras\":[{\"lora_path\":\"relative.gguf\"}]}"),
            "relative extra path accepted");
    require(!parse("{\"extra_loras\":[{\"lora_path\":\"" + role + "\"}]}"),
            "duplicate role accepted");
    require(parse("{\"extra_loras\":[{\"lora_path\":\"" + effect + "\"},{\"lora_path\":\"" +
                  third + "\"},{\"lora_path\":\"" + fourth + "\"}]}") &&
            selected.extra_loras.size() == 3, "stack larger than the cache capacity was rejected");
    require(parse("{\"extra_loras\":[{\"lora_path\":\"" + effect + "\"},{\"lora_path\":\"" +
                  third + "\",\"weight\":0}]}"), "zero-weight extra path was rejected");
}

static void test_cache_lifecycle() {
    require(ServerOptions{}.lora_cache_size == 3, "server default LoRA cache capacity is not three");
    require(!ServerOptions{}.te_cpu, "CPU text encoding must remain opt-in");

    TempDir temp;
    std::map<std::string, std::string> adapters;
    for (const char id : {'a', 'b', 'c', 'd'}) {
        const auto path = temp.path() / (std::string(1, id) + ".gguf");
        write_adapter_fixture(path);
        adapters.emplace(std::string("voice-") + id, path.u8string());
    }

    BreezeModel model;
    ModelCleanup cleanup{model};
    init_test_model(model);
    GenScheduler scheduler;
    scheduler.start(model, 4);
    {
        Models models(model, scheduler, "base", adapters, 3, 900);
        const CacheState empty = cache_state(models);
        require(empty.capacity == 3 && empty.loaded == 0 && empty.temporary_loaded == 0,
                "test server did not start with an empty three-slot cache");

        for (const char id : {'a', 'b', 'c'}) {
            auto lease = acquire(models, select_id(models, std::string("voice-") + id));
            lease.reset();
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        CacheState state = cache_state(models);
        require(state.loaded == 3 && state.temporary_loaded == 0, "first three adapters did not fill the cache");
        for (const char id : {'a', 'b', 'c'})
            require(state.loaded_ids.at(std::string("voice-") + id), "an adapter was missing before the fourth load");

        auto fourth = acquire(models, select_id(models, "voice-d"));
        state = cache_state(models);
        require(state.loaded == 3 && state.temporary_loaded == 0,
                "an idle cache entry was not evicted before loading the fourth adapter");
        require(!state.loaded_ids.at("voice-a") && state.loaded_ids.at("voice-b") &&
                state.loaded_ids.at("voice-c") && state.loaded_ids.at("voice-d"),
                "fourth adapter did not replace the least-recently-used idle adapter");
        fourth.reset();

        auto busy_b = acquire(models, select_id(models, "voice-b"));
        auto busy_c = acquire(models, select_id(models, "voice-c"));
        auto busy_d = acquire(models, select_id(models, "voice-d"));
        require(!cache_state(models).loaded_ids.at("voice-a"), "the fourth-load phase did not evict voice-a");

        auto temporary_a1 = acquire_promptly(models, select_id(models, "voice-a"));
        auto temporary_a2 = acquire(models, select_id(models, "voice-a"));
        require(temporary_a1->slot() == temporary_a2->slot(), "concurrent leases did not share one temporary adapter");
        state = cache_state(models);
        require(state.loaded == 4 && state.temporary_loaded == 1 && state.resident() == 3 &&
                state.loaded_ids.at("voice-a"),
                "a fourth adapter did not load temporarily while all resident entries were busy");
        temporary_a1.reset();
        state = cache_state(models);
        require(state.loaded == 4 && state.temporary_loaded == 1 && state.loaded_ids.at("voice-a"),
                "temporary adapter unloaded before its last lease released");
        temporary_a2.reset();
        state = cache_state(models);
        require(state.loaded == 3 && state.temporary_loaded == 0 && !state.loaded_ids.at("voice-a"),
                "last temporary lease release did not unload the adapter while the original three were busy");

        auto temporary_a3 = acquire(models, select_id(models, "voice-a"));
        state = cache_state(models);
        require(state.loaded == 4 && state.temporary_loaded == 1, "temporary adapter was not reloaded on demand");
        busy_b.reset();
        busy_c.reset();
        busy_d.reset();
        state = cache_state(models);
        require(state.loaded == 4 && state.temporary_loaded == 1 && state.resident() == 3 &&
                state.loaded_ids.at("voice-a"),
                "temporary adapter was promoted after resident adapters released");
        temporary_a3.reset();
        state = cache_state(models);
        require(state.loaded == 3 && state.temporary_loaded == 0 && !state.loaded_ids.at("voice-a"),
                "temporary adapter remained resident after the original cache slots were released");

        auto busy_b_again = acquire(models, select_id(models, "voice-b"));
        auto busy_c_again = acquire(models, select_id(models, "voice-c"));
        auto busy_d_again = acquire(models, select_id(models, "voice-d"));
        Models::Selection failed_load = select_id(models, "voice-a");
        failed_load.extra_loras.push_back({(temp.path() / "z-missing.gguf").u8string(), 1.0f});
        int status = 0;
        std::string error;
        auto failed_lease = models.acquire(failed_load, status, error);
        require(!failed_lease && status == 422, "request with a missing later adapter did not fail");
        state = cache_state(models);
        require(state.loaded == 3 && state.temporary_loaded == 0 && !state.loaded_ids.at("voice-a") &&
                model.loras.size() == 3,
                "partial adapter-load failure left an unused temporary adapter resident");
        busy_b_again.reset();
        busy_c_again.reset();
        busy_d_again.reset();

        Models::Selection oversized = select_id(models, "voice-a");
        std::string body_text = "{\"extra_loras\":[";
        for (const char id : {'b', 'c', 'd'}) {
            if (id != 'b') body_text += ",";
            body_text += "{\"lora_path\":" + json_quote(adapters.at(std::string("voice-") + id)) + "}";
        }
        body_text += "]}";
        JVal body;
        require(JParser(body_text).parse(body), "invalid oversized-stack test JSON");
        require(models.configure(body, oversized, status, error), "a single stack larger than three was rejected");
        require(oversized.extra_loras.size() == 3, "oversized stack members were not parsed");
        auto oversized_lease = acquire(models, oversized);
        state = cache_state(models);
        require(state.loaded == 4 && state.temporary_loaded == 1 && state.resident() == 3 &&
                model.lora_stacks.size() == 1,
                "single stack larger than capacity did not use a temporary over-capacity adapter");
        oversized_lease.reset();
        state = cache_state(models);
        require(state.loaded == 3 && state.temporary_loaded == 0 && !state.loaded_ids.at("voice-a") &&
                model.lora_stacks.empty(),
                "oversized stack release retained its over-capacity adapter or graph");

        Models::Selection composition = select_id(models, "voice-a");
        composition.weight = 0.9f;
        composition.extra_loras.push_back({adapters.at("voice-c"), 1.3f});
        auto first = acquire(models, composition);
        auto second = acquire(models, composition);
        require(first->slot() == second->slot(), "same weighted composition did not share one slot");
        const std::string composition_slot = first->slot();
        require(model.lora_stacks.size() == 1 && cache_state(models).weighted_compositions == 1,
                "weighted composition was not registered");

        scheduler.pool(composition_slot).step_ms[1] = 1.0;
        first.reset();
        require(model.lora_stacks.size() == 1 && cache_state(models).weighted_compositions == 1,
                "composition was removed while another lease was still using it");
        require(scheduler.pool(composition_slot).step_ms.count(1) == 1,
                "composition generation pool was removed before the final lease");

        second.reset();
        require(model.lora_stacks.empty() && cache_state(models).weighted_compositions == 0,
                "last release did not immediately remove the weighted composition");
        require(scheduler.pool(composition_slot).step_ms.empty(),
                "last release retained the weighted composition generation pool");
        scheduler.exec([&] { scheduler.unload_lora(composition_slot); });

        for (int i = 1; i <= 8; ++i) {
            Models::Selection varying = select_id(models, "voice-a");
            varying.weight = 0.1f * (float) i;
            auto lease = acquire(models, varying);
            const std::string slot = lease->slot();
            scheduler.pool(slot).step_ms[i] = (double) i;
            require(model.lora_stacks.size() == 1, "varying weight did not create exactly one composition");
            lease.reset();
            require(model.lora_stacks.empty() && cache_state(models).weighted_compositions == 0,
                    "different weights accumulated idle composition entries");
            require(scheduler.pool(slot).step_ms.empty(), "different weights retained an idle generation pool");
            scheduler.exec([&] { scheduler.unload_lora(slot); });
        }

        state = cache_state(models);
        require(state.loaded <= 3 && state.temporary_loaded == 0 && state.weighted_compositions == 0,
                "repeated compositions grew the cache after all leases released");
    }
    scheduler.stop();
}

int main() {
    try {
        test_options();
        test_cache_lifecycle();
        puts("LoRA options, temporary cache entries, and composition cleanup: OK");
        return 0;
    } catch (const std::exception & e) {
        fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
