#pragma once

#include "breeze/generation.h"
#include "models.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>

namespace breeze {

struct RequestLog {
    uint64_t id;
    std::chrono::steady_clock::time_point started;
};

namespace request_log_detail {

inline std::mutex & output_mutex() {
    static std::mutex mutex;
    return mutex;
}

inline std::atomic<uint64_t> & next_id() {
    static std::atomic<uint64_t> id{1};
    return id;
}

inline std::string escape(const std::string & value) {
    static const char hex[] = "0123456789ABCDEF";
    std::string out;
    out.reserve(value.size());
    for (unsigned char c : value) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"': out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            default:
                if (c < 0x20 || c == 0x7f) {
                    out += "\\x";
                    out.push_back(hex[c >> 4]);
                    out.push_back(hex[c & 0x0f]);
                } else {
                    out.push_back(static_cast<char>(c));
                }
        }
    }
    return out;
}

inline std::string quoted(const std::string & value) {
    return "\"" + escape(value) + "\"";
}

inline void write_block(const std::string & block) {
    std::lock_guard<std::mutex> lock(output_mutex());
    std::fwrite(block.data(), sizeof(char), block.size(), stdout);
    std::fflush(stdout);
}

} // namespace request_log_detail

// Pure formatter exposed so callers and the CPU-only test can inspect the complete block.
inline std::string format_speech_request(uint64_t id, const std::string & source,
                                         const std::string & base_model,
                                         const Models::Selection & selected,
                                         const std::string & voice_id,
                                         const GenRequest & request, bool parallel) {
    using request_log_detail::quoted;
    std::ostringstream out;
    out << "【合成请求 #" << id << "】\n"
        << "来源：" << quoted(source) << '\n'
        << "底模路径：" << quoted(base_model) << '\n'
        << "模型ID：" << quoted(selected.id) << '\n'
        << "请求中的 LoRA（非零权重）：\n";

    bool has_lora = false;
    if (!selected.path.empty() && selected.weight != 0) {
        out << "  主 LoRA：" << quoted(selected.path) << "，weight=" << selected.weight << '\n';
        has_lora = true;
    }
    for (size_t i = 0; i < selected.extra_loras.size(); ++i) {
        const auto & adapter = selected.extra_loras[i];
        if (adapter.weight == 0) continue;
        out << "  附加 LoRA[" << (i + 1) << "]：" << quoted(adapter.path)
            << "，weight=" << adapter.weight << '\n';
        has_lora = true;
    }
    if (!has_lora) out << "  （无）\n";

    out << "voice_id：" << quoted(voice_id) << '\n'
        << "执行方式：" << (parallel ? "并行" : "串行") << '\n'
        << "cfg：" << request.cfg_scale << "，seed：" << request.seed << '\n'
        << "指令：" << quoted(request.instruction) << '\n'
        << "文本：" << quoted(request.text) << '\n';
    return out.str();
}

inline std::string format_speech_event(uint64_t id, const std::string & event,
                                       std::chrono::steady_clock::duration elapsed) {
    using request_log_detail::quoted;
    const double elapsed_ms = std::chrono::duration<double, std::milli>(elapsed).count();
    std::ostringstream out;
    out << "【合成请求 #" << id << " 事件】\n"
        << "事件：" << quoted(event) << "，已用时：" << std::fixed << std::setprecision(3)
        << elapsed_ms << " ms\n";
    return out.str();
}

inline RequestLog log_speech_request(const std::string & source, const std::string & base_model,
                                     const Models::Selection & selected, const std::string & voice_id,
                                     const GenRequest & request, bool parallel) {
    const RequestLog log{request_log_detail::next_id().fetch_add(1, std::memory_order_relaxed),
                         std::chrono::steady_clock::now()};
    request_log_detail::write_block(format_speech_request(log.id, source, base_model, selected,
                                                          voice_id, request, parallel));
    return log;
}

inline void log_speech_event(const RequestLog & log, const std::string & event) {
    request_log_detail::write_block(format_speech_event(
        log.id, event, std::chrono::steady_clock::now() - log.started));
}

} // namespace breeze
