#include "request_log.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <io.h>
#define BREEZE_DUP _dup
#define BREEZE_DUP2 _dup2
#define BREEZE_FILENO _fileno
#define BREEZE_CLOSE _close
#else
#include <unistd.h>
#define BREEZE_DUP dup
#define BREEZE_DUP2 dup2
#define BREEZE_FILENO fileno
#define BREEZE_CLOSE close
#endif

using namespace breeze;

static void require(bool value, const char * message) {
    if (value) return;
    std::fprintf(stderr, "%s\n", message);
    std::exit(1);
}

class StdoutCapture {
public:
    StdoutCapture() : m_file(std::tmpfile()), m_saved(BREEZE_DUP(BREEZE_FILENO(stdout))) {
        require(m_file != nullptr, "could not create stdout capture file");
        require(m_saved >= 0, "could not duplicate stdout descriptor");
        std::fflush(stdout);
        require(BREEZE_DUP2(BREEZE_FILENO(m_file), BREEZE_FILENO(stdout)) >= 0,
                "could not redirect stdout");
    }

    ~StdoutCapture() {
        std::fflush(stdout);
        BREEZE_DUP2(m_saved, BREEZE_FILENO(stdout));
        BREEZE_CLOSE(m_saved);
        std::fclose(m_file);
    }

    std::string contents() {
        std::fflush(stdout);
        require(std::fflush(m_file) == 0, "could not flush capture file");
        std::rewind(m_file);
        std::string out;
        char buffer[4096];
        for (;;) {
            const size_t n = std::fread(buffer, 1, sizeof(buffer), m_file);
            out.append(buffer, n);
            if (n < sizeof(buffer)) break;
        }
        return out;
    }

private:
    FILE * m_file;
    int m_saved;
};

static void formatting() {
    Models::Selection selected;
    selected.id = "爱莉-日常";
    selected.path = "角色模型/爱莉/主音色.gguf";
    selected.weight = 0.8f;
    selected.extra_loras.push_back({"风格模型/ASMR/耳语.gguf", 0.35f});
    selected.extra_loras.push_back({"风格模型/停用.gguf", 0.0f});

    GenRequest request;
    request.text = "第一行\n第二行\t保留中文";
    request.instruction = "轻声\r缓慢\x01";
    request.cfg_scale = 1.25f;
    request.seed = 73;

    const std::string block = format_speech_request(
        99, "HTTP\n请求", "runtime/底模.gguf", selected, "voice\r1", request, true);
    require(block.find("【合成请求 #99】") != std::string::npos, "request ID missing");
    require(block.find("请求中的 LoRA（非零权重）") != std::string::npos,
            "requested adapter status was not labeled accurately");
    require(block.find("来源：\"HTTP\\n请求\"") != std::string::npos, "source newline was not escaped");
    require(block.find("底模路径：\"runtime/底模.gguf\"") != std::string::npos,
            "Chinese model path was not preserved");
    require(block.find("主音色.gguf\"") != std::string::npos, "primary LoRA missing");
    require(block.find("ASMR/耳语.gguf\"") != std::string::npos, "ASMR LoRA missing");
    require(block.find("停用.gguf") == std::string::npos, "zero-weight LoRA was logged as enabled");
    require(block.find("weight=0.8") != std::string::npos && block.find("weight=0.35") != std::string::npos,
            "active LoRA weights missing");
    require(block.find("voice_id：\"voice\\r1\"") != std::string::npos, "voice ID was not escaped");
    require(block.find("指令：\"轻声\\r缓慢\\x01\"") != std::string::npos,
            "instruction controls were not escaped");
    require(block.find("文本：\"第一行\\n第二行\\t保留中文\"") != std::string::npos,
            "multiline Chinese text was not preserved and escaped");
    require(block.find("cfg：1.25") != std::string::npos && block.find("seed：73") != std::string::npos,
            "cfg or seed missing");
    require(block.find("\\u") == std::string::npos, "Unicode text was converted to codepoint escapes");

    const auto elapsed = std::chrono::milliseconds(12);
    const std::string event = format_speech_event(99, "完成", elapsed);
    require(event.find("【合成请求 #99 事件】") != std::string::npos &&
            event.find("事件：\"完成\"，已用时：12.000 ms") != std::string::npos,
            "event ID or elapsed time missing");
}

static void concurrent_logging() {
    constexpr int threads = 8;
    constexpr int per_thread = 16;
    StdoutCapture capture;
    std::vector<RequestLog> logs(threads * per_thread);
    std::vector<std::thread> workers;
    for (int t = 0; t < threads; ++t) {
        workers.emplace_back([&, t] {
            for (int i = 0; i < per_thread; ++i) {
                const int index = t * per_thread + i;
                Models::Selection selected;
                selected.id = "model-" + std::to_string(index);
                GenRequest request;
                request.text = "request-" + std::to_string(index) + "\n中文";
                request.instruction = "instruction-" + std::to_string(index);
                logs[index] = log_speech_request("批量编号-" + std::to_string(index),
                                                 "base.gguf", selected, "voice", request, index % 2 == 0);
            }
        });
    }
    for (auto & worker : workers) worker.join();

    std::set<uint64_t> ids;
    for (const auto & log : logs) ids.insert(log.id);
    require(ids.size() == logs.size(), "concurrent request IDs collided");

    const std::string output = capture.contents();
    size_t block_count = 0;
    size_t begin = 0;
    while ((begin = output.find("【合成请求 #", begin)) != std::string::npos) {
        const size_t end = output.find("【合成请求 #", begin + 1);
        const std::string block = output.substr(begin, end == std::string::npos ? end : end - begin);
        const size_t id_start = begin + std::strlen("【合成请求 #");
        const size_t id_end = output.find_first_not_of("0123456789", id_start);
        require(id_end != std::string::npos, "request block ID is malformed");
        const uint64_t id = std::strtoull(output.substr(id_start, id_end - id_start).c_str(), nullptr, 10);
        require(ids.count(id) == 1, "output contained an unknown request ID");
        const auto matching = std::find_if(logs.begin(), logs.end(), [id](const RequestLog & log) {
            return log.id == id;
        });
        require(matching != logs.end(), "request ID was not returned");
        const size_t index = static_cast<size_t>(matching - logs.begin());
        require(block.find("模型ID：\"model-" + std::to_string(index) + "\"") != std::string::npos &&
                block.find("文本：\"request-" + std::to_string(index) + "\\n中文\"") !=
                    std::string::npos &&
                block.find("指令：\"instruction-" + std::to_string(index) + "\"") !=
                    std::string::npos,
                "concurrent request log block was interleaved");
        ++block_count;
        begin = end == std::string::npos ? output.size() : end;
    }
    require(block_count == logs.size(), "a concurrent log block was lost or split");
}

int main() {
    formatting();
    concurrent_logging();
    std::puts("request log: formatting, escaping, IDs and concurrent blocks passed");
}
