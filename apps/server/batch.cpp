#include "batch.h"
#include "encode.h"
#include "execution.h"
#include "models.h"
#include "openai.h"
#include "request_log.h"
#include "server.h"

#include "breeze/scheduler.h"
#include "httplib.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <set>
#include <thread>
#include <vector>

namespace breeze {

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

bool loopback(const std::string & address) {
    return address == "::1" || address.rfind("127.", 0) == 0 || address.rfind("::ffff:127.", 0) == 0;
}

struct Item {
    std::string id;
    OpenAiSpeech speech;
    Models::Selection selection;
    int status = 200;
    std::string error;
    std::string audio;
    double first_audio_ms = 0;
    double wall_ms = 0;
    size_t samples = 0;
};

void finish_wav(std::string & bytes) {
    const uint32_t riff = (uint32_t) bytes.size() - 8;
    const uint32_t data = (uint32_t) bytes.size() - 44;
    for (int i = 0; i < 4; i++) {
        bytes[4 + i] = (char) (riff >> (8 * i));
        bytes[40 + i] = (char) (data >> (8 * i));
    }
}

}

void add_batch_route(httplib::Server & server, BreezeModel & model, MimiCodec & codec,
                     GenScheduler & scheduler, VoiceStore & voices, Models & models,
                     RequestQueue & execution_queue, const ServerOptions & options) {
    server.Post("/v1/audio/speech/batch", [&](const httplib::Request & request, httplib::Response & response) {
        const auto fail = [&](int status, const std::string & error) {
            response.status = status;
            response.set_content(openai_error(error), "application/json");
        };
        if (request.get_header_value("Content-Type").find("json") == std::string::npos)
            return fail(415, "application/json required");
        JVal root;
        if (!JParser(request.body).parse(root) || root.kind != JVal::Obj)
            return fail(400, "request body must be a JSON object");
        bool parallel = false;
        const auto execution = root.obj.find("execution");
        if (execution != root.obj.end()) {
            if (execution->second.kind != JVal::Str ||
                (execution->second.str != "serial" && execution->second.str != "parallel"))
                return fail(400, "execution must be serial or parallel");
            parallel = execution->second.str == "parallel";
        }
        const auto source = root.obj.find("items");
        if (source == root.obj.end() || source->second.kind != JVal::Arr ||
            source->second.arr.empty() || source->second.arr.size() > 16)
            return fail(400, "items must contain between 1 and 16 speech requests");

        std::vector<Item> items;
        std::set<std::string> ids;
        const std::string default_id = models.default_id();
        for (const auto & value : source->second.arr) {
            if (value.kind != JVal::Obj) return fail(400, "each item must be a JSON object");
            if (value.obj.count("execution")) return fail(400, "set execution on the batch, not individual items");
            Item item;
            const auto id = value.obj.find("id");
            if (id != value.obj.end()) {
                if (id->second.kind != JVal::Str || id->second.str.empty())
                    return fail(400, "item id must be a nonempty string");
                item.id = id->second.str;
            } else item.id = std::to_string(items.size());
            if (!ids.insert(item.id).second) return fail(400, "item IDs must be unique within the batch");
            item.error = parse_openai_speech(value, voices, options.split_chars, item.speech, item.status);
            if (!item.error.empty()) return fail(item.status, "item " + item.id + ": " + item.error);
            const auto format = value.obj.find("response_format");
            if ((format != value.obj.end() && item.speech.format != "wav") || item.speech.sse)
                return fail(400, "batch output is WAV; response_format must be wav and SSE is unavailable");
            item.speech.format = "wav";
            item.speech.g.chunk_first = options.chunk_first;
            item.speech.g.chunk_max = options.chunk_max;
            // There is no player consuming this aggregate response before the entire batch completes.
            item.speech.g.pace = false;
            if (!item.speech.has_model && !item.speech.has_lora_path) {
                item.speech.has_model = true;
                item.speech.model = default_id;
            }
            if ((item.speech.has_lora_path || item.speech.lora_options.obj.count("extra_loras")) && !loopback(request.remote_addr))
                return fail(403, "lora_path is restricted to localhost");
            if (models.select(item.speech.has_model, item.speech.model, item.speech.has_lora_path,
                              item.speech.lora_path, item.selection, item.status, item.error))
                models.configure(item.speech.lora_options, item.selection, item.status, item.error);
            items.push_back(std::move(item));
        }

        const auto started = Clock::now();
        std::atomic<size_t> next{0};
        const auto worker = [&] {
            for (;;) {
                const size_t index = next.fetch_add(1);
                if (index >= items.size()) return;
                Item & item = items[index];
                if (!item.error.empty()) continue;
                const auto item_started = Clock::now();
                const auto request_log = log_speech_request("HTTP batch/" + item.id, item.selection, item.speech.g, parallel);
                try {
                    RequestExecution permit(execution_queue, parallel, scheduler.cost(item.speech.g.cfg_scale),
                                             request.is_connection_closed,
                                             [&] { log_speech_event(request_log, "排队等待"); });
                    if (!permit.acquired()) {
                        log_speech_event(request_log, "已取消");
                        item.status = 499;
                        item.error = "request cancelled";
                        continue;
                    }
                    item.speech.g.cancelled = permit.cancellation();
                    auto lease = models.acquire(item.selection, item.status, item.error, item.speech.g.cancelled);
                    if (!lease) {
                        log_speech_event(request_log, "加载失败：" + item.error);
                        continue;
                    }
                    item.speech.g.lora_id = lease->slot();
                    std::string sent, content_type;
                    auto encoder = make_encoder("wav", model.cfg.sample_rate, sent, content_type);
                    encoder->begin(item.audio);
                    GenTimings timings;
                    SpeechProgress progress(request_log, item.speech.g.text, model.cfg.sample_rate);
                    generate(model, codec, item.speech.g, [&](const float * samples, int count) {
                        if (permit.cancelled()) return false;
                        if (!item.samples) item.first_audio_ms = elapsed_ms(item_started);
                        encoder->write(samples, count, item.audio);
                        item.samples += (size_t) count;
                        progress.update((size_t) count);
                        return true;
                    }, &timings, &scheduler);
                    encoder->finish(item.audio);
                    if (permit.cancelled()) {
                        item.status = 499;
                        item.error = "request cancelled";
                        item.audio.clear();
                    } else if (!item.samples) {
                        item.status = 500;
                        item.error = "generation returned no audio";
                        item.audio.clear();
                    } else finish_wav(item.audio);
                    progress.finish(item.error.empty());
                    if (!item.error.empty()) log_speech_event(request_log, item.error);
                } catch (const std::exception & error) {
                    log_speech_event(request_log, std::string("生成失败：") + error.what());
                    item.status = 500;
                    item.error = error.what();
                    item.audio.clear();
                }
                item.wall_ms = elapsed_ms(item_started);
            }
        };
        const size_t count = parallel ? std::min(items.size(), (size_t) scheduler.parallel()) : 1;
        std::vector<std::thread> workers;
        for (size_t i = 1; i < count; i++) workers.emplace_back(worker);
        worker();
        for (auto & thread : workers) thread.join();

        std::string result = "{\"execution\":" + json_quote(parallel ? "parallel" : "serial") +
                             ",\"elapsed_ms\":" + std::to_string(elapsed_ms(started)) + ",\"results\":[";
        for (size_t i = 0; i < items.size(); i++) {
            const Item & item = items[i];
            if (i) result += ",";
            result += "{\"id\":" + json_quote(item.id) + ",\"model\":" + json_quote(item.selection.id);
            if (!item.error.empty()) {
                result += ",\"status\":\"error\",\"status_code\":" + std::to_string(item.status) +
                          ",\"error\":" + json_quote(item.error) + "}";
            } else {
                result += ",\"status\":\"ok\",\"format\":\"wav\",\"sample_rate\":" +
                          std::to_string(model.cfg.sample_rate) + ",\"first_audio_ms\":" +
                          std::to_string(item.first_audio_ms) + ",\"wall_ms\":" + std::to_string(item.wall_ms) +
                          ",\"duration_seconds\":" + std::to_string((double) item.samples / model.cfg.sample_rate) +
                          ",\"audio_base64\":" + json_quote(audio_base64(item.audio)) + "}";
            }
        }
        response.set_content(result + "]}", "application/json");
    });
}

}
