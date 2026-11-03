#include "server.h"
#include "encode.h"
#include "execution.h"
#include "json.h"
#include "models.h"
#include "openai.h"
#include "request_log.h"
#include "voices.h"
#include "batch.h"
#include "ws.h"
#include "ws_api.h"

#include "breeze/audio.h"
#include "breeze/generation.h"
#include "breeze/model.h"
#include "breeze/scheduler.h"

#include "httplib.h"
#include "breeze_webui_assets.h"

#ifdef _WIN32
#include <windows.h> // after httplib, it pulls in winsock2 first
#endif

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

namespace breeze {

static std::string mmss(double s) {
    if (!(s >= 0) || s > 86399) s = 0;
    char buf[16];
    snprintf(buf, sizeof buf, "%02d:%02d", (int) s / 60, (int) s % 60);
    return buf;
}

static std::string bar(double frac, int width) {
    static const char * part[] = { " ", "\u258f", "\u258e", "\u258d", "\u258c", "\u258b", "\u258a", "\u2589" };
    if (!(frac > 0)) frac = 0;
    if (frac > 1) frac = 1;
    const double filled = frac * width;
    const int full = (int) filled;
    std::string s;
    for (int i = 0; i < full; i++) s += "\u2588";
    if (full < width) {
        s += part[(int) ((filled - full) * 8)];
        for (int i = full + 1; i < width; i++) s += " ";
    }
    return s;
}

// an origin thats not on the allowlist gets an empty string, so no header goes out
static std::string cors_origin(const std::string & allowed, const httplib::Request & req) {
    if (allowed == "*") return "*";
    const std::string origin = req.get_header_value("Origin");
    if (origin.empty()) return "";
    size_t start = 0;
    while (start <= allowed.size()) {
        size_t end = allowed.find(',', start);
        if (end == std::string::npos) end = allowed.size();
        if (allowed.compare(start, end - start, origin) == 0) return origin;
        start = end + 1;
    }
    return "";
}

static bool loopback_peer(const std::string & address) {
    return address == "::1" || address == "localhost" ||
           address.rfind("127.", 0) == 0 || address.rfind("::ffff:127.", 0) == 0;
}

static void fail(httplib::Response & res, int status, const std::string & message) {
    res.status = status;
    res.set_content(std::string("{\"error\":\"") + message + "\"}", "application/json");
}

// one generation with the console progress bar, emit gets each decoded chunk and false stops it
static bool run_generation(BreezeModel & model, MimiCodec & codec, GenScheduler & sched, const GenRequest & g, int sr,
                           bool verbose, const AudioCallback & emit) {
    GenTimings tm;
    const auto t0 = std::chrono::steady_clock::now();
    const auto elapsed = [&] {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    };
    // the model decides when to stop, so the total is only ever an estimate
    const double est = estimate_seconds(g.text);
    size_t total = 0;
    bool success = true;
    try {
        generate(model, codec, g, [&](const float * s, int n) {
            if (!emit(s, n)) return false;
            total += (size_t) n;
            const double secs = (double) total / sr, wall = elapsed();
            const double rate = wall > 0 ? secs / wall : 0;
            const double frac = est > 0 ? secs / est : 0;
            const double eta = rate > 0 ? (est > secs ? (est - secs) / rate : 0) : 0;
            // bars from several streams would just write over each other
            if (sched.parallel() > 1) return true;
            printf("\r%3.0f%%|%s| %.1f/%.1fs [%s<%s, %.1f fps, %.2fx]  ",
                   (frac < 1 ? frac : 1) * 100, bar(frac, 24).c_str(), secs, est,
                   mmss(wall).c_str(), mmss(eta).c_str(), secs * 12.5 / wall, rate);
            fflush(stdout);
            return true;
        }, &tm, &sched);
    } catch (const std::exception & e) {
        success = false;
        fprintf(stderr, "\ngeneration error: %s\n", e.what());
    }
    const double secs = (double) total / sr, wall = elapsed();
    printf("\r%3.0f%%|%s| %.1f/%.1fs [%s, %.1f fps, %.2fx]        \n",
           100.0, bar(1, 24).c_str(), secs, secs, mmss(wall).c_str(),
           wall > 0 ? secs * 12.5 / wall : 0.0, wall > 0 ? secs / wall : 0.0);
    printf("     %d frames in %d flushes, first audio %.0f ms\n",
           tm.frames, tm.flushes, tm.first_audio);
    if (verbose && tm.frames > 0) {
        printf("     ref %.0f  prompt %.0f  prefill %.0f ms | per frame: backbone %.2f"
               "  depth %.2f  vocoder %.2f ms\n",
               tm.encode_ref, tm.prompt, tm.prefill, tm.backbone / tm.frames,
               tm.depth / tm.frames, tm.vocoder / tm.frames);
    }
    fflush(stdout);
    return success && total > 0;
}

int run_server(const ServerOptions & opts) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8); // otherwise the bar glyphs and any chinese text come out as mojibake
#endif
    BreezeModel model;
    printf("loading %s ...\n", opts.model.c_str());
    if (!model.load(opts.model, opts.use_gpu)) {
        fprintf(stderr, "failed to load model\n");
        return 1;
    }
    printf("backend: %s, sample rate: %d\n", model.backend.name(), model.cfg.sample_rate);
    MimiCodec codec;
    codec.init(model);
    GenScheduler sched;
    sched.start(model, opts.parallel);
    if (opts.warmup) {
        GenRequest base;
        base.chunk_first = opts.chunk_first;
        base.chunk_max = opts.chunk_max;
        try {
            warm_up(model, codec, base);
            sched.warm_up(base);
        } catch (const std::exception & e) {
            fprintf(stderr, "warm up failed: %s\n", e.what());
        }
    }
    if (opts.parallel > 1) printf("up to %d streams at once\n", opts.parallel);

    Models models(model, sched, opts.model_id, opts.switchable_loras,
                  opts.lora_cache_size, opts.lora_idle_seconds);
    RequestQueue execution_queue(sched);
    const int sr = model.cfg.sample_rate;

    VoiceStore store;
    store.load_dir(opts.voices_dir, model.cfg.num_codebooks);

    // Declare the server after Models so its workers are destroyed before their cache manager.
    httplib::Server svr;
    std::mutex admission_mutex;
    bool draining = false;
    std::unordered_set<WsConn *> websocket_sessions;
    std::unordered_set<const httplib::Request *> admitted_http;
    svr.set_pre_routing_handler([&](const httplib::Request & req, httplib::Response & res) {
        if (req.path.rfind("/__gpu_audio/", 0) == 0) return httplib::Server::HandlerResponse::Unhandled;
        std::lock_guard<std::mutex> lock(admission_mutex);
        const bool safe_read = req.method == "GET" &&
            (req.path == "/health" || req.path == "/v1/models" || req.path == "/v1/voices");
        if (draining && !safe_read) {
            res.status = 503;
            res.set_content("{\"error\":\"service draining\"}", "application/json");
            return httplib::Server::HandlerResponse::Handled;
        }
        admitted_http.insert(&req);
        return httplib::Server::HandlerResponse::Unhandled;
    });
    svr.set_post_routing_handler([&](const httplib::Request & req, httplib::Response & res) {
        {
            std::lock_guard<std::mutex> lock(admission_mutex);
            admitted_http.erase(&req);
        }
        if (opts.cors.empty()) return;
        const std::string origin = cors_origin(opts.cors, req);
        if (origin.empty()) return;
        res.set_header("Access-Control-Allow-Origin", origin);
        res.set_header("Access-Control-Expose-Headers", "X-Sample-Rate, X-Sample-Format");
        if (origin != "*") res.set_header("Vary", "Origin");
    });
    svr.Post("/__gpu_audio/drain", [&](const httplib::Request & req, httplib::Response & res) {
        if (!loopback_peer(req.remote_addr)) return fail(res, 403, "localhost only");
        std::lock_guard<std::mutex> lock(admission_mutex);
        draining = true;
        res.set_content("{\"ok\":true}", "application/json");
    });
    svr.Post("/__gpu_audio/resume", [&](const httplib::Request & req, httplib::Response & res) {
        if (!loopback_peer(req.remote_addr)) return fail(res, 403, "localhost only");
        std::lock_guard<std::mutex> lock(admission_mutex);
        draining = false;
        execution_queue.resume();
        res.set_content("{\"ok\":true}", "application/json");
    });
    svr.Post("/__gpu_audio/cancel", [&](const httplib::Request & req, httplib::Response & res) {
        if (!loopback_peer(req.remote_addr)) return fail(res, 403, "localhost only");
        std::lock_guard<std::mutex> lock(admission_mutex);
        draining = true;
        execution_queue.cancel_all();
        for (WsConn * connection : websocket_sessions) connection->close();
        res.set_content("{\"ok\":true}", "application/json");
    });
    svr.Get("/__gpu_audio/idle", [&](const httplib::Request & req, httplib::Response & res) {
        if (!loopback_peer(req.remote_addr)) return fail(res, 403, "localhost only");
        std::lock_guard<std::mutex> lock(admission_mutex);
        const auto activity = execution_queue.activity();
        res.set_content("{\"draining\":" + std::string(draining ? "true" : "false") +
                        ",\"http\":" + std::to_string(admitted_http.size()) +
                        ",\"websockets\":" + std::to_string(websocket_sessions.size()) +
                        ",\"active\":" + std::to_string(activity.first) +
                        ",\"waiting\":" + std::to_string(activity.second) + "}", "application/json");
    });
    store.add_routes(svr, model, codec, sched, execution_queue, opts.voices_dir);

    WsServer ws;
    const int ws_port = opts.ws_port == 0 ? opts.port + 1 : opts.ws_port;
    if (ws_port > 0) {
        const bool up = ws.start(opts.host, ws_port, [&](WsConn & c) {
            {
                std::lock_guard<std::mutex> lock(admission_mutex);
                if (draining) {
                    c.send_text("{\"type\":\"error\",\"message\":\"service draining\"}");
                    return;
                }
                websocket_sessions.insert(&c);
            }
            try {
                ws_connection(c, model, codec, store, sched, models, execution_queue,
                              opts.chunk_first, opts.chunk_max, opts.split_chars, opts.model);
            } catch (...) {
                std::lock_guard<std::mutex> lock(admission_mutex);
                websocket_sessions.erase(&c);
                throw;
            }
            std::lock_guard<std::mutex> lock(admission_mutex);
            websocket_sessions.erase(&c);
        });
        if (up) printf("websocket on ws://%s:%d\n", opts.host.c_str(), ws_port);
        else fprintf(stderr, "could not open the websocket port %d\n", ws_port);
    }

    svr.Get("/health", [&](const httplib::Request &, httplib::Response & res) {
        res.set_content("{\"status\":\"ok\",\"model_id\":\"" + models.default_id() + "\",\"sample_rate\":" + std::to_string(sr) +
                        ",\"ws_port\":" + std::to_string(ws_port > 0 ? ws_port : 0) + "}",
                        "application/json");
    });

    svr.Get("/v1/models", [&](const httplib::Request &, httplib::Response & res) {
        res.set_content(models.json(), "application/json");
    });

    svr.Post("/v1/models/active", [&](const httplib::Request & req, httplib::Response & res) {
        if (!loopback_peer(req.remote_addr))
            return fail(res, 403, "default model changes are restricted to localhost");
        if (req.get_header_value("Content-Type").find("json") == std::string::npos)
            return fail(res, 415, "application/json required");
        JVal body;
        if (!JParser(req.body).parse(body) || body.kind != JVal::Obj)
            return fail(res, 400, "request body must be a JSON object");
        const auto id_field = body.obj.find("model_id");
        if (id_field == body.obj.end() || id_field->second.kind != JVal::Str || id_field->second.str.empty())
            return fail(res, 400, "model_id is required");
        const auto path_field = body.obj.find("lora_path");
        const bool has_path = path_field != body.obj.end();
        if (has_path && (path_field->second.kind != JVal::Str || path_field->second.str.empty()))
            return fail(res, 400, "lora_path must be a nonempty absolute path");

        const std::string target_id = id_field->second.str;
        Models::Selection selection;
        int status = 0;
        std::string error;
        if (!models.select(true, target_id, has_path,
                           has_path ? path_field->second.str : std::string(), selection, status, error))
            return fail(res, status, error);

        // A path supplied with a new or existing ID becomes registered only after its load succeeds.
        if (has_path) {
            std::shared_ptr<Models::Lease> lease = models.acquire(selection, status, error);
            if (!lease) return fail(res, status, error);
        }

        const std::string old_default = models.default_id();
        if (!models.set_default(target_id, status, error)) return fail(res, status, error);
        const std::string active = models.default_id();
        res.set_content("{\"active_model\":\"" + active + "\",\"changed\":" +
                        (old_default == active ? "false}" : "true}"), "application/json");
    });

    add_batch_route(svr, model, codec, sched, store, models, execution_queue, opts);

    const auto select_form_model = [&](const httplib::Request & req, Models::Selection & selection,
                                       int & status, std::string & error) {
        const bool has_id = req.has_file("model_id") || req.has_param("model_id");
        const bool has_path = req.has_file("lora_path") || req.has_param("lora_path");
        const bool has_extras = req.has_file("extra_loras") || req.has_param("extra_loras");
        if ((has_path || has_extras) && !loopback_peer(req.remote_addr)) {
            status = 403;
            error = "lora_path is restricted to localhost requests";
            return false;
        }
        if (!models.select(has_id, field(req, "model_id", ""), has_path,
                           field(req, "lora_path", ""), selection, status, error)) return false;
        JVal options;
        for (const char * key : {"lora_weight", "extra_loras"}) {
            if (!req.has_file(key) && !req.has_param(key)) continue;
            JVal value;
            if (!JParser(field(req, key, "")).parse(value)) {
                status = 400;
                error = std::string(key) + " must contain valid JSON";
                return false;
            }
            options.obj[key] = std::move(value);
        }
        return models.configure(options, selection, status, error);
    };

    // OpenAI JSON uses its standard `model` field; native form requests use `model_id`.
    const auto speak_openai = [&](const httplib::Request & req, httplib::Response & res) {
        OpenAiSpeech o;
        int status = 200;
        const std::string err = parse_openai_speech(req.body, store, opts.split_chars, o, status);
        if (!err.empty()) {
            res.status = status;
            res.set_content(openai_error(err), "application/json");
            return;
        }
        Models::Selection selection;
        std::string error;
        if ((o.has_lora_path || o.lora_options.obj.count("extra_loras")) && !loopback_peer(req.remote_addr)) {
            res.status = 403;
            res.set_content(openai_error("lora_path is restricted to localhost requests"), "application/json");
            return;
        }
        if (!models.select(o.has_model, o.model, o.has_lora_path, o.lora_path,
                           selection, status, error) || !models.configure(o.lora_options, selection, status, error)) {
            res.status = status;
            res.set_content(openai_error(error, status == 409 || status == 422 ? "server_error" : "invalid_request_error"),
                            "application/json");
            return;
        }
        const int cost = sched.cost(o.g.cfg_scale);
        const auto request_log = log_speech_request("HTTP JSON", opts.model, selection, o.voice, o.g, o.parallel);
        auto permit = std::make_shared<RequestExecution>(execution_queue, o.parallel, cost, req.is_connection_closed,
            [&] { log_speech_event(request_log, "排队等待"); });
        if (!permit->acquired()) {
            log_speech_event(request_log, "已取消");
            res.status = 499;
            return;
        }
        o.g.cancelled = permit->cancellation();
        std::shared_ptr<Models::Lease> lease = models.acquire(selection, status, error, o.g.cancelled);
        if (!lease) {
            log_speech_event(request_log, "加载失败：" + error);
            res.status = status;
            res.set_content(openai_error(error, status == 409 || status == 422 ? "server_error" : "invalid_request_error"),
                            "application/json");
            return;
        }
        o.g.lora_id = lease->slot();
        log_speech_event(request_log, "开始推理");
        o.g.chunk_first = opts.chunk_first;
        o.g.chunk_max = opts.chunk_max;
        std::string sent, ctype;
        std::shared_ptr<AudioEncoder> enc = make_encoder(o.format, sr, sent, ctype);

        res.set_header("X-Sample-Rate", std::to_string(enc->sample_rate(sr)));
        res.set_header("Cache-Control", "no-store");
        if (o.sse) ctype = "text/event-stream";

        const char * mode = o.g.ref_frames > 0 ? (o.g.instruction != GenRequest().instruction ? "direction" : "clone")
                                               : "design";
        printf("gen  %s, %d chars, cfg %.1f, seed %d, openai %s%s%s\n", mode, (int) o.g.text.size(),
               o.g.cfg_scale, o.g.seed, sent.c_str(), sent != o.format ? " for " : "",
               sent != o.format ? o.format.c_str() : "");
        fflush(stdout);

        res.set_chunked_content_provider(
            ctype, [&model, &codec, &sched, g = o.g, sse = o.sse, enc, lease, permit, request_log, sr, verbose = opts.verbose](
                       size_t, httplib::DataSink & sink) {
                std::string buf;
                const auto flush = [&] {
                    if (buf.empty()) return true;
                    const std::string out = sse ? sse_audio_delta(buf) : buf;
                    buf.clear();
                    return !permit->cancelled() && sink.write(out.data(), out.size());
                };
                enc->begin(buf);
                bool alive = flush();
                bool generated = false;
                if (alive) {
                    generated = run_generation(model, codec, sched, g, sr, verbose, [&](const float * s, int n) {
                        enc->write(s, n, buf);
                        return alive = flush();
                    });
                    if (alive && !permit->cancelled()) {
                        enc->finish(buf);
                        if (flush() && sse) {
                            const std::string done = sse_audio_done();
                            sink.write(done.data(), done.size());
                        }
                    }
                }
                log_speech_event(request_log, permit->cancelled() || !alive ? "已取消或连接断开" :
                                 (generated ? "生成完成" : "生成失败"));
                sink.done();
                return true;
            });
    };

    svr.Post("/v1/audio/speech", [&](const httplib::Request & req, httplib::Response & res) {
        if (req.get_header_value("Content-Type").find("json") != std::string::npos) {
            speak_openai(req, res);
            return;
        }
        GenRequest g;
        g.text = field(req, "text", "");
        g.instruction = field(req, "instruction", "Speak clearly and naturally.");
        g.ref_text = field(req, "ref_text", "");
        g.cfg_scale = (float) atof(field(req, "cfg_scale", "1.0").c_str());
        g.seed = atoi(field(req, "seed", "42").c_str());
        g.temperature = (float) atof(field(req, "temperature", "0").c_str());
        g.top_k = atoi(field(req, "top_k", "0").c_str());
        g.top_p = (float) atof(field(req, "top_p", "0").c_str());
        g.repetition_penalty = (float) atof(field(req, "repetition_penalty", "0").c_str());
        g.max_new_tokens = atoi(field(req, "max_new_tokens", "0").c_str());
        g.split_chars = atoi(field(req, "split_chars", std::to_string(opts.split_chars)).c_str());
        g.chunk_first = opts.chunk_first;
        g.chunk_max = opts.chunk_max;
        if (g.text.empty()) return fail(res, 400, "text is required");

        Models::Selection selection;
        int status = 0;
        std::string error;
        if (!select_form_model(req, selection, status, error)) return fail(res, status, error);
        const std::string execution = field(req, "execution", "serial");
        if (execution != "serial" && execution != "parallel")
            return fail(res, 400, "execution must be serial or parallel");
        const int cost = sched.cost(g.cfg_scale);
        const auto request_log = log_speech_request("HTTP 表单", opts.model, selection,
                                                   field(req, "voice_id", ""), g, execution == "parallel");
        auto permit = std::make_shared<RequestExecution>(execution_queue, execution == "parallel", cost, req.is_connection_closed,
            [&] { log_speech_event(request_log, "排队等待"); });
        if (!permit->acquired()) {
            log_speech_event(request_log, "已取消");
            res.status = 499;
            return;
        }
        g.cancelled = permit->cancellation();
        std::shared_ptr<Models::Lease> lease = models.acquire(selection, status, error, g.cancelled);
        if (!lease) {
            log_speech_event(request_log, "加载失败：" + error);
            return fail(res, status, error);
        }
        g.lora_id = lease->slot();
        if (req.has_file("ref_audio")) {
            const auto & f = req.get_file_value("ref_audio");
            if (!f.content.empty() && !g.ref_text.empty()) {
                // a raw upload joins the same hash cache an unnamed POST /v1/voices entry
                // does: identical clip and transcript reuse the codes instead of re-running
                // the codec (which is most of the pre-audio wait), and identical seeds give
                // byte identical audio either way
                const std::string key = voice_key(f.content, g.ref_text);
                if (!store.take(key, g.ref_codes, g.ref_frames, g.ref_text)) {
                    std::vector<float> pcm;
                    if (read_wav_buffer((const uint8_t *) f.content.data(), f.content.size(), sr, pcm) && !pcm.empty()) {
                        CachedVoice v;
                        v.text = g.ref_text;
                        const auto t0 = std::chrono::steady_clock::now();
                        sched.exec([&] { v.codes = codec.encode(pcm, v.frames); });
                        v.encode_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                        printf("ref %s: encoded %d frames in %.0f ms\n", key.c_str(), v.frames, v.encode_ms);
                        fflush(stdout);
                        g.ref_codes = v.codes;
                        g.ref_frames = v.frames;
                        store.put(key, std::move(v));
                    }
                } else {
                    printf("ref %s: cache hit, %d frames, encode skipped\n", key.c_str(), g.ref_frames);
                    fflush(stdout);
                }
            } else if (!f.content.empty()) {
                read_wav_buffer((const uint8_t *) f.content.data(), f.content.size(), sr, g.ref_audio);
            }
            // without this a bad clip quietly turns a clone into voice design
            if (!f.content.empty() && g.ref_codes.empty() && g.ref_audio.empty()) {
                log_speech_event(request_log, "参考音频读取失败");
                res.status = 400;
                res.set_content("{\"error\":\"could not read ref_audio\"}", "application/json");
                return;
            }
        }
        const std::string vid = field(req, "voice_id", "");
        if (!vid.empty() && !store.take(vid, g.ref_codes, g.ref_frames, g.ref_text)) {
            log_speech_event(request_log, "音色不存在：" + vid);
            res.status = 404;
            res.set_content("{\"error\":\"unknown voice_id\"}", "application/json");
            return;
        }
        if (g.text.empty()) {
            res.status = 400;
            res.set_content("{\"error\":\"text is required\"}", "application/json");
            return;
        }

        res.set_header("X-Sample-Rate", std::to_string(sr));
        res.set_header("X-Sample-Format", "s16le");
        res.set_header("Cache-Control", "no-store");

        const bool sent_ins = req.has_file("instruction") || req.has_param("instruction");
        const bool has_ref = !g.ref_audio.empty() || !g.ref_codes.empty();
        const char * mode = has_ref ? (sent_ins ? "direction" : "clone") : "design";
        printf("gen  %s, %d chars, cfg %.1f, seed %d\n", mode, (int) g.text.size(), g.cfg_scale, g.seed);
        fflush(stdout);
        log_speech_event(request_log, "开始推理");

        res.set_chunked_content_provider(
            "audio/pcm",
            [&model, &codec, &sched, g, lease, permit, request_log, sr, verbose = opts.verbose](size_t, httplib::DataSink & sink) {
                bool alive = true;
                const bool generated = run_generation(model, codec, sched, g, sr, verbose, [&](const float * s, int n) {
                    const std::vector<uint8_t> pcm = to_pcm16(s, n);
                    return alive = !permit->cancelled() && sink.write((const char *) pcm.data(), pcm.size());
                });
                log_speech_event(request_log, permit->cancelled() || !alive ? "已取消或连接断开" :
                                 (generated ? "生成完成" : "生成失败"));
                sink.done();
                return true;
            });
    });

    svr.Post("/v1/audio/convert", [&](const httplib::Request & req, httplib::Response & res) {
        std::vector<float> src, ref;
        const auto load = [&](const char * name, std::vector<float> & out) {
            if (!req.has_file(name)) return false;
            const auto & f = req.get_file_value(name);
            return !f.content.empty() &&
                   read_wav_buffer((const uint8_t *) f.content.data(), f.content.size(), sr, out);
        };
        if (!load("source", src)) {
            res.status = 400;
            res.set_content("{\"error\":\"a source wav file is required\"}", "application/json");
            return;
        }
        std::string ref_text = field(req, "ref_text", "");
        std::vector<int> vcodes;
        int vframes = 0;
        const std::string vid = field(req, "voice_id", "");
        if (!vid.empty()) {
            if (!store.take(vid, vcodes, vframes, ref_text)) {
                res.status = 404;
                res.set_content("{\"error\":\"unknown voice_id\"}", "application/json");
                return;
            }
        } else if (!load("ref_audio", ref)) {
            res.status = 400;
            res.set_content("{\"error\":\"ref_audio or voice_id is required\"}", "application/json");
            return;
        }
        if (ref_text.empty()) {
            res.status = 400;
            res.set_content("{\"error\":\"ref_text is required\"}", "application/json");
            return;
        }

        Models::Selection selection;
        int status = 0;
        std::string error;
        if (!select_form_model(req, selection, status, error)) return fail(res, status, error);
        GenRequest log_parameters;
        log_parameters.text = field(req, "text", "");
        log_parameters.instruction.clear();
        log_parameters.cfg_scale = (float) atof(field(req, "cfg_scale", "1.0").c_str());
        log_parameters.seed = atoi(field(req, "seed", "42").c_str());
        const auto request_log = log_speech_request("HTTP 音色转换", opts.model, selection, vid,
                                                   log_parameters, false);
        RequestExecution permit(execution_queue, false, sched.parallel(), req.is_connection_closed,
            [&] { log_speech_event(request_log, "排队等待"); });
        if (!permit.acquired()) {
            log_speech_event(request_log, "已取消");
            res.status = 499;
            return;
        }
        std::shared_ptr<Models::Lease> lease = models.acquire(selection, status, error, permit.cancellation());
        if (!lease) {
            log_speech_event(request_log, "加载失败：" + error);
            return fail(res, status, error);
        }
        log_speech_event(request_log, "开始转换");

        printf("conv %.2f s source, %s\n", (double) src.size() / sr,
               vframes > 0 ? "cached reference" : "uploaded reference");
        fflush(stdout);
        try {
            ConvertOptions copt;
            copt.cancelled = permit.cancellation();
            copt.src_text = field(req, "text", "");
            copt.temperature = (float) atof(field(req, "temperature", "0.3").c_str());
            copt.top_k = atoi(field(req, "top_k", "1").c_str());
            copt.cfg_scale = (float) atof(field(req, "cfg_scale", "1.0").c_str());
            copt.keep_acoustic = atoi(field(req, "keep_acoustic", "0").c_str());
            copt.seed = atoi(field(req, "seed", "42").c_str());
            copt.ref_codes = vcodes;
            copt.ref_frames = vframes;
            const auto t0 = std::chrono::steady_clock::now();
            std::vector<float> audio;
            sched.exec([&] {
                int T = 0;
                std::vector<int> codes;
                {
                    ScopedLora base(model, "");
                    codes = codec.encode(src, T);
                }
                {
                    ScopedLora active(model, lease->slot());
                    audio = convert_voice(model, codec, codes, T, ref, ref_text, copt);
                }
            });
            if (permit.cancelled()) {
                log_speech_event(request_log, "已取消");
                return fail(res, 499, "request cancelled");
            }
            const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            const double secs = (double) audio.size() / sr;
            printf("     %.2f s in %.2f s, %.2fx rt\n", secs, wall, wall > 0 ? secs / wall : 0.0);
            fflush(stdout);
            std::vector<uint8_t> pcm = to_pcm16(audio.data(), (int) audio.size());
            res.set_header("X-Sample-Rate", std::to_string(sr));
            res.set_header("X-Sample-Format", "s16le");
            res.set_content((const char *) pcm.data(), pcm.size(), "audio/pcm");
            log_speech_event(request_log, "转换完成");
        } catch (const std::exception & e) {
            log_speech_event(request_log, std::string("转换失败：") + e.what());
            fprintf(stderr, "conversion error: %s\n", e.what());
            res.status = 500;
            res.set_content("{\"error\":\"conversion failed\"}", "application/json");
        }
    });

    if (opts.webui) {
        svr.Get("/", [](const httplib::Request &, httplib::Response & res) {
            res.set_content(breeze_webui::index_html, "text/html");
        });
        svr.Get("/style.css", [](const httplib::Request &, httplib::Response & res) {
            res.set_content(breeze_webui::style_css, "text/css");
        });
        svr.Get("/app.js", [](const httplib::Request &, httplib::Response & res) {
            res.set_content(breeze_webui::app_js, "application/javascript");
        });
        printf("web ui: http://%s:%d/\n", opts.host.c_str(), opts.port);
    }

    if (!opts.cors.empty()) {
        // an origin off the list still gets a 204, just without the header, so the browser refuses it
        svr.Options(".*", [](const httplib::Request & req, httplib::Response & res) {
            res.set_header("Access-Control-Allow-Methods", "GET, POST, PATCH, DELETE, OPTIONS");
            res.set_header("Access-Control-Max-Age", "86400");
            const std::string want = req.get_header_value("Access-Control-Request-Headers");
            if (!want.empty()) res.set_header("Access-Control-Allow-Headers", want);
            res.status = 204;
        });
        printf("cors: %s\n", opts.cors == "*" ? "any origin" : opts.cors.c_str());
    }

    printf("listening on http://%s:%d\n", opts.host.c_str(), opts.port);
    const bool listening = svr.listen(opts.host, opts.port);
    if (!listening) {
        fprintf(stderr, "failed to bind %s:%d\n", opts.host.c_str(), opts.port);
    }
    ws.stop();
    models.stop();
    sched.stop();
    model.free();
    return listening ? 0 : 1;
}

}
