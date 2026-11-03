#include "ws_api.h"
#include "json.h"
#include "request_log.h"

#include "breeze/audio.h"
#include "breeze/generation.h"
#include "breeze/scheduler.h"

#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <exception>
#include <limits>
#include <thread>

namespace breeze {

static std::string json_str(const JVal & msg, const char * key) {
    const auto it = msg.obj.find(key);
    return it != msg.obj.end() && it->second.kind == JVal::Str ? it->second.str : "";
}

static double json_num(const JVal & msg, const char * key, double def) {
    const auto it = msg.obj.find(key);
    return it != msg.obj.end() && it->second.kind == JVal::Num ? it->second.num : def;
}

static std::string esc(const std::string & s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else if (c == '\n') o += "\\n";
        else if ((unsigned char) c < 0x20) continue;
        else o += c;
    }
    return o;
}

// Checkpoints must round-trip reference text, including tabs and carriage returns.
static std::string checkpoint_esc(const std::string & s) {
    std::string out;
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') { out += '\\'; out += (char) c; }
        else if (c < 0x20) {
            char buf[7];
            snprintf(buf, sizeof buf, "\\u%04x", (unsigned) c);
            out += buf;
        } else out += (char) c;
    }
    return out;
}

static bool loopback_peer(const std::string & address) {
    return address == "127.0.0.1" || address.rfind("127.", 0) == 0 ||
           address == "::1" || address == "localhost";
}

static bool sentence_end(const std::string & s, size_t i) {
    const char c = s[i];
    if (c == '.' || c == '!' || c == '?' || c == ';') {
        // a bare dot inside a number or an abbreviation is not the end of anything
        return i + 1 >= s.size() || s[i + 1] == ' ' || s[i + 1] == '\n';
    }
    // the cjk stops carry their own spacing
    static const char * stops[] = { "\xe3\x80\x82", "\xef\xbc\x81", "\xef\xbc\x9f", "\xef\xbc\x9b" };
    for (const char * st : stops)
        if (s.compare(i, 3, st) == 0) return true;
    return false;
}

// moves whole sentences out of buf, leaving a trailing partial behind unless force is set
static std::vector<std::string> drain(std::string & buf, int budget, bool force) {
    std::vector<std::string> out;
    size_t cut = 0;
    for (size_t i = 0; i < buf.size(); i++)
        if (sentence_end(buf, i)) cut = i + ((unsigned char) buf[i] < 0x80 ? 1 : 3);

    // nothing finished but the buffer is already long enough to speak, so break it on a space
    if (!cut && !force && (int) buf.size() > budget) {
        const size_t sp = buf.rfind(' ');
        if (sp != std::string::npos) cut = sp + 1;
    }
    // flush and end mean all of it, otherwise a tail after the last full stop (a closing quote,
    // an unfinished clause) sits in the buffer and gets glued onto the front of the next message
    if (force) cut = buf.size();
    if (!cut) return out;

    std::string ready = buf.substr(0, cut);
    buf.erase(0, cut);
    for (std::string & p : split_text(ready, budget)) {
        while (!p.empty() && p.front() == ' ') p.erase(p.begin());
        if (!p.empty()) out.push_back(p);
    }
    return out;
}

namespace {

// each piece remembers the instruction that was live when it was queued, otherwise a
// direction sent while the queue is deep would land on whatever piece happens to start next
struct Piece {
    std::string text;
    std::string instruction;
};

struct Session {
    GenSession gen;
    std::mutex mu;
    std::condition_variable cv;
    std::deque<Piece> queue;
    std::string buffer;
    std::string instruction = "Speak clearly and naturally.";
    int budget = 600;
    int cost = 1;
    bool parallel_execution = false;
    Models::Selection selection;
    GenRequest log_parameters;
    std::string voice_id;
    bool selection_registration_pending = false;
    bool started = false;
    bool ending = false;
    bool quit = false;
    bool speaking = false;
    std::atomic<bool> cancel{false};
};

} // namespace

static void event(WsConn & c, const std::string & type, const std::string & extra = "") {
    c.send_text("{\"type\":\"" + type + "\"" + (extra.empty() ? "" : "," + extra) + "}");
}

// drains the queue one piece at a time, taking a slot for each so other connections interleave
static void speaker(WsConn & conn, Session & s, GenScheduler & sched, Models & models,
                    RequestQueue & execution_queue, const std::string & base_model) {
    for (;;) {
        Piece piece;
        Models::Selection selected;
        bool register_id = false;
        bool parallel_execution = false;
        GenRequest log_parameters;
        std::string voice_id;
        {
            std::unique_lock<std::mutex> lock(s.mu);
            s.speaking = false;
            s.cv.wait(lock, [&] { return s.quit || !s.queue.empty() || s.ending; });
            if (s.quit) return;
            if (s.queue.empty()) {
                // `end` arrived with nothing left to speak, so the speaker owns the terminal event
                s.ending = false;
                lock.unlock();
                event(conn, "done");
                continue;
            }
            piece = std::move(s.queue.front());
            s.queue.pop_front();
            s.speaking = true;
            selected = s.selection;
            register_id = s.selection_registration_pending;
            parallel_execution = s.parallel_execution;
            log_parameters = s.log_parameters;
            voice_id = s.voice_id;
        }
        log_parameters.text = piece.text;
        log_parameters.instruction = piece.instruction;
        const auto request_log = log_speech_request("WebSocket", base_model, selected, voice_id,
                                                   log_parameters, parallel_execution);
        if (s.cancel) {
            log_speech_event(request_log, "已取消");
            s.cancel = false;
            event(conn, "cancelled");
            continue;
        }

        const auto cancelled = [&] { return s.cancel || !conn.alive(); };
        RequestExecution permit(execution_queue, parallel_execution, s.cost, cancelled,
                                [&] { log_speech_event(request_log, "排队等待"); event(conn, "queued"); });
        if (!permit.acquired()) {
            log_speech_event(request_log, "已取消");
            if (!conn.alive()) return;
            s.cancel = false;
            event(conn, "cancelled");
            continue;
        }
        if (s.cancel) {
            log_speech_event(request_log, "已取消");
            s.cancel = false;
            event(conn, "cancelled");
            continue;
        }
        if (!conn.alive()) { log_speech_event(request_log, "连接断开"); return; }

        selected.register_id = register_id;
        int status = 0;
        std::string error;
        std::shared_ptr<Models::Lease> lease;
        try {
            lease = models.acquire(selected, status, error, permit.cancellation());
        } catch (const std::exception & e) {
            error = e.what();
        } catch (...) {
            error = "failed to acquire LoRA adapter";
        }
        if (!lease) {
            log_speech_event(request_log, "加载失败或已取消：" + error);
            if (!conn.alive()) return;
            if (s.cancel) {
                s.cancel = false;
                event(conn, "cancelled");
                continue;
            }
            (void) status;
            event(conn, "error", "\"message\":\"" + esc(error) + "\"");
            continue;
        }
        if (s.cancel) {
            log_speech_event(request_log, "已取消");
            lease.reset();
            event(conn, "cancelled");
            s.cancel = false;
            continue;
        }
        if (!conn.alive()) {
            log_speech_event(request_log, "连接断开");
            lease.reset();
            return;
        }
        if (register_id) {
            std::lock_guard<std::mutex> lock(s.mu);
            s.selection_registration_pending = false;
        }

        {
            std::lock_guard<std::mutex> lock(s.mu);
            s.gen.set_instruction(piece.instruction);
            s.gen.set_lora(lease->slot());
            s.gen.set_cancelled(permit.cancellation());
        }
        event(conn, "speaking", "\"text\":\"" + esc(piece.text) + "\"");
        log_speech_event(request_log, "开始推理");

        bool ok = false;
        try {
            ok = s.gen.speak(piece.text, [&](const float * a, int n) {
                if (s.cancel || !conn.alive()) return false;
                std::vector<uint8_t> pcm = to_pcm16(a, n);
                return conn.send_binary(pcm.data(), pcm.size());
            });
        } catch (const std::exception & e) {
            log_speech_event(request_log, std::string("生成失败：") + e.what());
            fprintf(stderr, "websocket generation error: %s\n", e.what());
            event(conn, "error", "\"message\":\"generation failed\"");
            lease.reset();
            continue;
        } catch (...) {
            log_speech_event(request_log, "生成失败：未知错误");
            fprintf(stderr, "websocket generation error: unknown exception\n");
            event(conn, "error", "\"message\":\"generation failed\"");
            lease.reset();
            continue;
        }
        lease.reset();
        permit.release();
        log_speech_event(request_log, s.cancel || !conn.alive() ? "已取消或连接断开" :
                         (ok ? "本段完成" : "本段提前结束"));

        if (s.cancel) { event(conn, "cancelled"); s.cancel = false; continue; }
        if (!ok && !conn.alive()) return;

        std::lock_guard<std::mutex> lock(s.mu);
        if (s.queue.empty() && s.ending) {
            s.ending = false;
            s.speaking = false;
            event(conn, "done");
        }
    }
}

static void handle_start(WsConn & conn, Session & s, const JVal & msg, BreezeModel & model,
                         MimiCodec & codec, VoiceStore & store, GenScheduler & sched, Models & models,
                         int chunk_first,
                         int chunk_max, int split_chars) {
    if (s.started) {
        event(conn, "error", "\"message\":\"start was already received for this connection\"");
        return;
    }
    const auto optional_string = [&](const char * key, bool & present, std::string & value) {
        const auto it = msg.obj.find(key);
        present = it != msg.obj.end();
        if (!present) return true;
        if (it->second.kind != JVal::Str || it->second.str.empty()) return false;
        value = it->second.str;
        return true;
    };
    bool has_id = false, has_path = false;
    std::string model_id, lora_path;
    if (!optional_string("model_id", has_id, model_id) || !optional_string("lora_path", has_path, lora_path)) {
        event(conn, "error", "\"message\":\"model_id and lora_path must be nonempty strings\"");
        return;
    }
    bool parallel_execution = false;
    const auto execution = msg.obj.find("execution");
    if (execution != msg.obj.end()) {
        if (execution->second.kind != JVal::Str ||
            (execution->second.str != "serial" && execution->second.str != "parallel")) {
            event(conn, "error", "\"message\":\"execution must be serial or parallel\"");
            return;
        }
        parallel_execution = execution->second.str == "parallel";
    }
    if ((has_path || msg.obj.count("extra_loras")) && !loopback_peer(conn.remote_addr())) {
        event(conn, "error", "\"message\":\"lora_path is restricted to localhost requests\"");
        return;
    }
    Models::Selection selection;
    int status = 0;
    std::string error;
    if (!models.select(has_id, model_id, has_path, lora_path, selection, status, error) ||
        !models.configure(msg, selection, status, error)) {
        event(conn, "error", "\"message\":\"" + esc(error) + "\"");
        return;
    }

    GenRequest g;
    g.instruction = json_str(msg, "instruction");
    if (g.instruction.empty()) g.instruction = "Speak clearly and naturally.";
    g.ref_text = json_str(msg, "ref_text");
    g.cfg_scale = (float) json_num(msg, "cfg_scale", 1.0);
    g.seed = (int) json_num(msg, "seed", 42);
    g.temperature = (float) json_num(msg, "temperature", 0);
    g.top_k = (int) json_num(msg, "top_k", 0);
    g.max_new_tokens = (int) json_num(msg, "max_new_tokens", 0);
    g.chunk_first = chunk_first;
    g.chunk_max = chunk_max;

    const std::string vid = json_str(msg, "voice_id");
    if (!vid.empty() && !store.take(vid, g.ref_codes, g.ref_frames, g.ref_text)) {
        event(conn, "error", "\"message\":\"unknown voice_id\"");
        return;
    }

    std::lock_guard<std::mutex> lock(s.mu);
    s.instruction = g.instruction;
    s.cost = sched.cost(g.cfg_scale);
    s.parallel_execution = parallel_execution;
    s.selection = std::move(selection);
    s.log_parameters.cfg_scale = g.cfg_scale;
    s.log_parameters.seed = g.seed;
    s.voice_id = vid;
    s.selection_registration_pending = s.selection.register_id;
    s.budget = (int) json_num(msg, "split_chars", split_chars);
    // streaming drains sentence by sentence, so it always needs a real budget to aim at
    if (s.budget <= 0) s.budget = 600;
    s.gen.begin(model, codec, g, nullptr, &sched);
    s.started = true;
    s.queue.clear();
    s.buffer.clear();
    s.cancel = false;
    event(conn, "started", "\"voice_id\":\"" + esc(vid) + "\"");
}

void ws_connection(WsConn & conn, BreezeModel & model, MimiCodec & codec, VoiceStore & store,
                   GenScheduler & sched, Models & models,
                   RequestQueue & execution_queue,
                   int chunk_first, int chunk_max, int split_chars, const std::string & base_model) {
    Session s;
    std::thread worker([&] { speaker(conn, s, sched, models, execution_queue, base_model); });
    event(conn, "ready", "\"sample_rate\":24000,\"format\":\"s16le\"");

    std::string raw;
    bool binary = false;
    while (conn.recv(raw, binary)) {
        if (binary) continue;
        JVal msg;
        if (!JParser(raw).parse(msg) || msg.kind != JVal::Obj) {
            event(conn, "error", "\"message\":\"messages have to be a json object\"");
            continue;
        }
        const std::string type = json_str(msg, "type");

        if (type == "start") {
            handle_start(conn, s, msg, model, codec, store, sched, models,
                         chunk_first, chunk_max, split_chars);
            continue;
        }
        if (!s.started) {
            event(conn, "error", "\"message\":\"send start first\"");
            continue;
        }
        if (type == "__gpu_audio_checkpoint") {
            if (!loopback_peer(conn.remote_addr())) {
                event(conn, "__gpu_audio_error", "\"message\":\"localhost only\"");
                continue;
            }
            std::lock_guard<std::mutex> lock(s.mu);
            if (s.speaking || s.ending || !s.queue.empty() || !s.buffer.empty()) {
                event(conn, "__gpu_audio_error", "\"message\":\"session is not idle\"");
                continue;
            }
            const auto state = s.gen.checkpoint();
            std::string fields = "\"state\":{\"frames\":" + std::to_string(state.frames) +
                ",\"piece\":" + std::to_string(state.piece) + ",\"text\":\"" + checkpoint_esc(state.text) + "\",\"codes\":[";
            for (size_t i = 0; i < state.codes.size(); ++i) {
                if (i) fields += ",";
                fields += std::to_string(state.codes[i]);
            }
            fields += "]}";
            event(conn, "__gpu_audio_checkpoint", fields);
        } else if (type == "__gpu_audio_restore") {
            if (!loopback_peer(conn.remote_addr())) {
                event(conn, "__gpu_audio_error", "\"message\":\"localhost only\"");
                continue;
            }
            const auto it = msg.obj.find("state");
            if (it == msg.obj.end() || it->second.kind != JVal::Obj) {
                event(conn, "__gpu_audio_error", "\"message\":\"invalid checkpoint\"");
                continue;
            }
            const auto & obj = it->second.obj;
            const auto frames = obj.find("frames"), piece = obj.find("piece");
            const auto text = obj.find("text"), codes = obj.find("codes");
            const auto valid_uint = [](const JVal & v, double limit) {
                return v.kind == JVal::Num && std::isfinite(v.num) && v.num >= 0 &&
                    v.num <= limit && std::floor(v.num) == v.num;
            };
            if (frames == obj.end() || piece == obj.end() || text == obj.end() || codes == obj.end() ||
                !valid_uint(frames->second, std::numeric_limits<int>::max()) ||
                !valid_uint(piece->second, std::numeric_limits<uint32_t>::max()) ||
                text->second.kind != JVal::Str || codes->second.kind != JVal::Arr ||
                codes->second.arr.size() > 2000000 ||
                codes->second.arr.size() != (size_t) frames->second.num * (size_t) model.cfg.num_codebooks) {
                event(conn, "__gpu_audio_error", "\"message\":\"invalid checkpoint\"");
                continue;
            }
            GenSessionCheckpoint state;
            state.frames = (int) frames->second.num;
            state.piece = (uint32_t) piece->second.num;
            state.text = text->second.str;
            bool valid = true;
            for (const auto & code : codes->second.arr) {
                if (!valid_uint(code, std::numeric_limits<int>::max())) { valid = false; break; }
                state.codes.push_back((int) code.num);
            }
            std::lock_guard<std::mutex> lock(s.mu);
            if (!valid || s.speaking || s.ending || !s.queue.empty() || !s.buffer.empty()) {
                event(conn, "__gpu_audio_error", "\"message\":\"invalid or busy session\"");
                continue;
            }
            s.gen.restore_checkpoint(state);
            event(conn, "__gpu_audio_restored");
        } else if (type == "instruction") {
            std::lock_guard<std::mutex> lock(s.mu);
            // applies to everything queued from here on, pieces already waiting keep theirs
            s.instruction = json_str(msg, "instruction");
            event(conn, "instruction_set");
        } else if (type == "text" || type == "flush" || type == "end") {
            std::unique_lock<std::mutex> lock(s.mu);
            s.buffer += json_str(msg, "text");
            const bool force = type != "text";
            // while there is no clip to clone the opening piece doubles as the reference, and a
            // long one makes the model skip sentences later, so it stays near a normal clip length
            const int budget = s.gen.needs_anchor() && s.queue.empty() ? 200 : s.budget;
            for (std::string & p : drain(s.buffer, budget, force))
                s.queue.push_back({std::move(p), s.instruction});
            if (type == "end") s.ending = true;
            lock.unlock();
            s.cv.notify_one();
        } else if (type == "cancel") {
            std::lock_guard<std::mutex> lock(s.mu);
            s.queue.clear();
            s.buffer.clear();
            s.ending = false;
            // only latch it while something is actually being spoken, otherwise the flag
            // sits true and swallows whatever gets sent next
            s.cancel = s.speaking;
        } else {
            event(conn, "error", "\"message\":\"unknown type\"");
        }
    }

    {
        std::lock_guard<std::mutex> lock(s.mu);
        s.quit = true;
        s.cancel = true;
    }
    s.cv.notify_all();
    worker.join();
}

} // namespace breeze
