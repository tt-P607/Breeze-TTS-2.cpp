#pragma once

#include "breeze/generation.h"
#include "models.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
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

struct ProgressLine {
    bool open = false;
    uint64_t id = 0;
    size_t columns = 0;
};

inline std::mutex & output_mutex() {
    static std::mutex mutex;
    return mutex;
}

inline std::atomic<uint64_t> & next_id() {
    static std::atomic<uint64_t> id{1};
    return id;
}

inline ProgressLine & progress_line() {
    static ProgressLine line;
    return line;
}

inline void write_bytes(const std::string & text) {
    if (!text.empty()) std::fwrite(text.data(), sizeof(char), text.size(), stdout);
    std::fflush(stdout);
}

inline void close_progress_line_locked() {
    ProgressLine & line = progress_line();
    if (!line.open) return;
    write_bytes("\n");
    line = {};
}

inline std::string escape_inline(const std::string & value) {
    static const char hex[] = "0123456789ABCDEF";
    std::string out;
    out.reserve(value.size());
    for (unsigned char c : value) {
        switch (c) {
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

inline bool decode_utf8(const std::string & text, size_t offset, uint32_t & codepoint, size_t & length) {
    const unsigned char first = static_cast<unsigned char>(text[offset]);
    if (first < 0x80) {
        codepoint = first;
        length = 1;
        return true;
    }

    if (first >= 0xc2 && first <= 0xdf) length = 2;
    else if (first >= 0xe0 && first <= 0xef) length = 3;
    else if (first >= 0xf0 && first <= 0xf4) length = 4;
    else return false;
    if (offset + length > text.size()) return false;

    codepoint = first & (length == 2 ? 0x1f : (length == 3 ? 0x0f : 0x07));
    for (size_t i = 1; i < length; ++i) {
        const unsigned char next = static_cast<unsigned char>(text[offset + i]);
        if ((next & 0xc0) != 0x80) return false;
        if (i == 1 && ((first == 0xe0 && next < 0xa0) || (first == 0xed && next >= 0xa0) ||
                       (first == 0xf0 && next < 0x90) || (first == 0xf4 && next >= 0x90))) return false;
        codepoint = (codepoint << 6) | (next & 0x3f);
    }
    return true;
}

inline int unicode_columns(uint32_t cp) {
    if (cp == 0x200d || (cp >= 0x0300 && cp <= 0x036f) || (cp >= 0x1ab0 && cp <= 0x1aff) ||
        (cp >= 0x1dc0 && cp <= 0x1dff) || (cp >= 0x20d0 && cp <= 0x20ff) ||
        (cp >= 0xfe00 && cp <= 0xfe0f) || (cp >= 0xfe20 && cp <= 0xfe2f) ||
        (cp >= 0xe0100 && cp <= 0xe01ef)) return 0;
    if (cp >= 0x1100 && (cp <= 0x115f || cp == 0x2329 || cp == 0x232a ||
        (cp >= 0x2e80 && cp <= 0xa4cf) || (cp >= 0xac00 && cp <= 0xd7a3) ||
        (cp >= 0xf900 && cp <= 0xfaff) || (cp >= 0xfe10 && cp <= 0xfe19) ||
        (cp >= 0xfe30 && cp <= 0xfe6f) || (cp >= 0xff00 && cp <= 0xff60) ||
        (cp >= 0xffe0 && cp <= 0xffe6) || (cp >= 0x1f300 && cp <= 0x1faff) ||
        (cp >= 0x20000 && cp <= 0x3fffd))) return 2;
    return 1;
}

inline size_t display_columns(const std::string & text) {
    size_t columns = 0;
    for (size_t i = 0; i < text.size();) {
        uint32_t cp = 0;
        size_t length = 1;
        if (decode_utf8(text, i, cp, length)) {
            columns += static_cast<size_t>(unicode_columns(cp));
            i += length;
        } else {
            ++columns;
            ++i;
        }
    }
    return columns;
}

inline std::string escaped_byte(unsigned char c) {
    static const char hex[] = "0123456789ABCDEF";
    std::string out = "\\x00";
    out[2] = hex[c >> 4];
    out[3] = hex[c & 0x0f];
    return out;
}

inline std::string wrap_display_text(const std::string & text) {
    constexpr size_t line_columns = 80;
    constexpr size_t indent_columns = 2;
    std::string out = "  ";
    size_t columns = indent_columns;
    for (size_t i = 0; i < text.size();) {
        const unsigned char byte = static_cast<unsigned char>(text[i]);
        if (byte == '\n' || byte == '\r') {
            if (byte == '\r' && i + 1 < text.size() && text[i + 1] == '\n') ++i;
            out += "\n  ";
            columns = indent_columns;
            ++i;
            continue;
        }
        if (byte < 0x20 || byte == 0x7f) {
            std::string escaped;
            if (byte == '\t') escaped = "\\t";
            else if (byte == '\b') escaped = "\\b";
            else if (byte == '\f') escaped = "\\f";
            else escaped = escaped_byte(byte);
            if (columns + escaped.size() > line_columns) {
                out += "\n  ";
                columns = indent_columns;
            }
            out += escaped;
            columns += escaped.size();
            ++i;
            continue;
        }

        uint32_t cp = 0;
        size_t length = 1;
        if (!decode_utf8(text, i, cp, length)) {
            const std::string escaped = escaped_byte(byte);
            if (columns + escaped.size() > line_columns) {
                out += "\n  ";
                columns = indent_columns;
            }
            out += escaped;
            columns += escaped.size();
            ++i;
            continue;
        }
        // Avoid C1 and Unicode line separators being interpreted as terminal controls.
        if ((cp >= 0x80 && cp <= 0x9f) || cp == 0x2028 || cp == 0x2029) {
            for (size_t j = 0; j < length; ++j) {
                const std::string escaped = escaped_byte(static_cast<unsigned char>(text[i + j]));
                if (columns + escaped.size() > line_columns) {
                    out += "\n  ";
                    columns = indent_columns;
                }
                out += escaped;
                columns += escaped.size();
            }
            i += length;
            continue;
        }

        const int width = unicode_columns(cp);
        if (width > 0 && columns + static_cast<size_t>(width) > line_columns) {
            out += "\n  ";
            columns = indent_columns;
        }
        out.append(text, i, length);
        columns += static_cast<size_t>(width);
        i += length;
    }
    return out;
}

inline std::string filename(const std::string & path) {
    const size_t slash = path.find_last_of("/\\");
    return path.substr(slash == std::string::npos ? 0 : slash + 1);
}

inline std::string format_weight(float weight) {
    std::ostringstream out;
    out << std::setprecision(4) << weight;
    return out.str();
}

inline void write_block(const std::string & block) {
    std::lock_guard<std::mutex> lock(output_mutex());
    close_progress_line_locked();
    write_bytes(block);
}

inline void write_progress_line(uint64_t id, const std::string & text, bool final) {
    std::lock_guard<std::mutex> lock(output_mutex());
    ProgressLine & line = progress_line();
    std::string output;
    if (line.open) {
        if (line.id == id) {
            output = "\r";
            output += text;
            const size_t columns = display_columns(text);
            if (line.columns > columns) output.append(line.columns - columns, ' ');
        } else {
            output = "\n";
            output += text;
        }
    } else {
        output = text;
    }
    if (final) {
        output += '\n';
        line = {};
    } else {
        line.open = true;
        line.id = id;
        line.columns = display_columns(text);
    }
    write_bytes(output);
}

inline std::string format_progress(uint64_t id, int percent, const std::string & status,
                                   double audio_seconds, double elapsed_seconds, double speed) {
    const int filled = percent * 25 / 100;
    std::string bar(static_cast<size_t>(filled), '#');
    bar.append(static_cast<size_t>(25 - filled), '-');
    std::ostringstream out;
    out << "[合成 #" << id << "] [" << bar << "] " << std::setw(3) << percent << "% " << status
        << " | 音频 " << std::fixed << std::setprecision(2) << audio_seconds << "s"
        << " | 耗时 " << elapsed_seconds << "s"
        << " | 速度 " << speed << "x";
    return out.str();
}

} // namespace request_log_detail

inline std::string format_speech_request(uint64_t id, const std::string & source,
                                         const Models::Selection & selected,
                                         const GenRequest & request, bool parallel) {
    using namespace request_log_detail;
    std::ostringstream out;
    out << "############ 合成 #" << id << " | 来源：" << escape_inline(source)
        << " | " << (parallel ? "并行" : "串行") << " ############\n"
        << "模型ID：" << escape_inline(selected.id) << '\n';

    bool has_lora = false;
    if (!selected.path.empty() && selected.weight != 0) {
        out << "主 LoRA：" << escape_inline(filename(selected.path))
            << "（weight=" << format_weight(selected.weight) << "）\n";
        has_lora = true;
    }
    for (const auto & adapter : selected.extra_loras) {
        if (adapter.weight == 0) continue;
        out << "附加 LoRA：" << escape_inline(filename(adapter.path))
            << "（weight=" << format_weight(adapter.weight) << "）\n";
        has_lora = true;
    }
    if (!has_lora) out << "LoRA：无\n";

    out << "\n############ 语气指令 ############\n"
        << wrap_display_text(request.instruction) << "\n\n"
        << "############ 推理文本 ############\n"
        << wrap_display_text(request.text) << '\n';
    return out.str();
}

inline std::string format_speech_event(uint64_t id, const std::string & event,
                                       std::chrono::steady_clock::duration elapsed) {
    const double seconds = std::chrono::duration<double>(elapsed).count();
    std::ostringstream out;
    out << "[合成 #" << id << "] " << request_log_detail::escape_inline(event)
        << " | " << std::fixed << std::setprecision(2) << seconds << "s\n";
    return out.str();
}

inline RequestLog log_speech_request(const std::string & source, const Models::Selection & selected,
                                     const GenRequest & request, bool parallel) {
    const RequestLog log{request_log_detail::next_id().fetch_add(1, std::memory_order_relaxed),
                         std::chrono::steady_clock::now()};
    request_log_detail::write_block(format_speech_request(log.id, source, selected, request, parallel));
    return log;
}

inline void log_speech_event(const RequestLog & log, const std::string & event) {
    request_log_detail::write_block(format_speech_event(
        log.id, event, std::chrono::steady_clock::now() - log.started));
}

class SpeechProgress {
public:
    SpeechProgress(const RequestLog & log, const std::string & text, int sample_rate)
        : m_log(log), m_sample_rate(sample_rate > 0 ? sample_rate : 1),
          m_expected_seconds(estimate_seconds(text)), m_started(std::chrono::steady_clock::now()),
          m_last_draw(m_started) {
        draw_locked(m_started, false, false);
    }

    ~SpeechProgress() { finish(false); }
    SpeechProgress(const SpeechProgress &) = delete;
    SpeechProgress & operator=(const SpeechProgress &) = delete;

    void update(size_t samples) {
        std::lock_guard<std::mutex> lock(m_state_mutex);
        if (m_finished) return;
        m_total_samples += samples;
        const auto now = std::chrono::steady_clock::now();
        const bool first_chunk = samples > 0 && !m_seen_chunk;
        if (samples > 0) m_seen_chunk = true;
        if (first_chunk || now - m_last_draw >= std::chrono::milliseconds(250)) {
            draw_locked(now, false, false);
            m_last_draw = now;
        }
    }

    void finish(bool success = true) {
        std::lock_guard<std::mutex> lock(m_state_mutex);
        if (m_finished) return;
        m_finished = true;
        draw_locked(std::chrono::steady_clock::now(), true, success);
    }

private:
    void draw_locked(std::chrono::steady_clock::time_point now, bool final, bool success) {
        using namespace std::chrono;
        const double audio_seconds = static_cast<double>(m_total_samples) / m_sample_rate;
        const double elapsed_seconds = duration<double>(now - m_started).count();
        int percent = 0;
        if (std::isfinite(m_expected_seconds) && m_expected_seconds > 0) {
            percent = static_cast<int>(std::floor(audio_seconds / m_expected_seconds * 100.0));
        }
        if (final && success) percent = 100;
        else percent = std::max(0, std::min(99, percent));
        const char * status = final ? (success ? "完成" : "中断") : "预计";
        const double speed = elapsed_seconds > 0 ? audio_seconds / elapsed_seconds : 0;
        request_log_detail::write_progress_line(
            m_log.id, request_log_detail::format_progress(m_log.id, percent, status,
                                                          audio_seconds, elapsed_seconds, speed), final);
    }

    RequestLog m_log;
    int m_sample_rate;
    double m_expected_seconds;
    std::chrono::steady_clock::time_point m_started;
    std::chrono::steady_clock::time_point m_last_draw;
    size_t m_total_samples = 0;
    bool m_seen_chunk = false;
    bool m_finished = false;
    std::mutex m_state_mutex;
};

} // namespace breeze
