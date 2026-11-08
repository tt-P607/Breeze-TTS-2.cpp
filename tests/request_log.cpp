#include "request_log.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
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
    selected.id = "elysia";
    selected.path = "E:/private/models/voice/main.voice.gguf";
    selected.weight = 0.8f;
    selected.extra_loras.push_back({"E:/private/styles/ASMR/whisper.gguf", 0.35f});
    selected.extra_loras.push_back({"E:/private/styles/inactive.gguf", 0.0f});

    GenRequest request;
    request.instruction = "轻声\r缓慢\t\x1B";
    request.text = std::string(78, 'x') + "中\r\n\n第二行\t尾\x1B";

    const std::string block = format_speech_request(99, "HTTP batch/7\nline", selected, request, true);
    require(block.find("############ 合成 #99 | 来源：HTTP batch/7\\nline | 并行 ############") !=
                std::string::npos,
            "request header or source escaping missing");
    require(block.find("模型ID：elysia\n") != std::string::npos, "model ID missing");
    require(block.find("主 LoRA：main.voice.gguf（weight=0.8）") != std::string::npos,
            "primary LoRA filename or weight missing");
    require(block.find("附加 LoRA：whisper.gguf（weight=0.35）") != std::string::npos,
            "extra LoRA filename or weight missing");
    require(block.find("inactive.gguf") == std::string::npos, "zero-weight LoRA was printed");
    require(block.find("E:/private") == std::string::npos, "full LoRA paths were printed");
    require(block.find("voice_id") == std::string::npos && block.find("cfg") == std::string::npos &&
            block.find("seed") == std::string::npos && block.find("底模") == std::string::npos,
            "private or low-value request fields were printed");
    require(block.find('"') == std::string::npos, "request fields were wrapped in quotes");
    require(block.find("############ 语气指令 ############\n  轻声\n  缓慢\\t\\x1B\n\n") !=
                std::string::npos,
            "instruction section or control escaping failed");
    const std::string wrapped = "############ 推理文本 ############\n  " + std::string(78, 'x') +
                                "\n  中\n  \n  第二行\\t尾\\x1B\n";
    require(block.find(wrapped) != std::string::npos,
            "Chinese text wrap split a code point or changed an original line break");

    const std::string event = format_speech_event(99, "开始推理", std::chrono::milliseconds(1200));
    require(event == "[合成 #99] 开始推理 | 1.20s\n", "event was not a single readable line");
}

static std::vector<uint64_t> ids_in_progress_lines(const std::string & output) {
    std::vector<uint64_t> ids;
    size_t line = 0;
    while (line < output.size()) {
        const size_t end = output.find('\n', line);
        const std::string row = output.substr(line, end == std::string::npos ? end : end - line);
        size_t pos = 0;
        std::set<uint64_t> row_ids;
        while ((pos = row.find("[合成 #", pos)) != std::string::npos) {
            const size_t id_start = pos + std::strlen("[合成 #");
            const size_t id_end = row.find_first_not_of("0123456789", id_start);
            require(id_end != std::string::npos, "progress ID was malformed");
            row_ids.insert(std::strtoull(row.substr(id_start, id_end - id_start).c_str(), nullptr, 10));
            pos = id_end;
        }
        require(row_ids.size() <= 1, "different progress IDs shared one output line");
        if (!row_ids.empty()) ids.push_back(*row_ids.begin());
        if (end == std::string::npos) break;
        line = end + 1;
    }
    return ids;
}

static void progress_output() {
    StdoutCapture capture;

    Models::Selection selected;
    selected.id = "request-model";
    GenRequest request;
    request.instruction = "轻声";
    request.text = "正文不得被进度覆盖";
    const RequestLog first = log_speech_request("HTTP", selected, request, false);
    {
        SpeechProgress progress(first, "abcdefghij", 1000);
        progress.update(250);
        log_speech_event(first, "开始推理");
        progress.finish(true);
        progress.finish(false);
    }

    const RequestLog failed{600, std::chrono::steady_clock::now()};
    {
        SpeechProgress progress(failed, "abcdefghij", 1000);
        progress.update(100);
        progress.finish(false);
    }

    constexpr int tasks = 4;
    std::vector<RequestLog> logs;
    std::vector<std::unique_ptr<SpeechProgress>> progress;
    for (int i = 0; i < tasks; ++i) {
        Models::Selection concurrent_selection;
        concurrent_selection.id = "concurrent-" + std::to_string(i);
        GenRequest concurrent_request;
        concurrent_request.text = "并发正文-" + std::to_string(i);
        concurrent_request.instruction = "并发指令";
        logs.push_back(log_speech_request("HTTP batch/" + std::to_string(i), concurrent_selection,
                                          concurrent_request, true));
    }
    for (const auto & log : logs)
        progress.emplace_back(new SpeechProgress(log, "abcdefghij", 1000));

    std::vector<std::thread> workers;
    for (int i = 0; i < tasks; ++i) {
        workers.emplace_back([&, i] {
            progress[i]->update(static_cast<size_t>(i + 1) * 100);
            progress[i]->finish(true);
        });
    }
    for (auto & worker : workers) worker.join();

    const std::string output = capture.contents();
    require(output.find("推理文本 ############\n  正文不得被进度覆盖") != std::string::npos,
            "progress output overwrote the request body");
    require(output.find("0% 预计") != std::string::npos, "constructor did not draw initial zero progress");
    require(output.find("42% 预计") != std::string::npos, "first chunk did not update expected progress");
    require(output.find("音频 0.25s") != std::string::npos && output.find("速度 ") != std::string::npos,
            "sample count or progress timing fields missing");
    require(output.find("100% 完成") != std::string::npos, "successful finish did not draw 100 percent");
    require(output.find("17% 中断") != std::string::npos, "failed progress did not preserve its incomplete percent");
    require(output.find("[合成 #600]") != std::string::npos &&
            output.find("[合成 #600] [####---------------------]  17% 中断") != std::string::npos,
            "failed final progress line missing");
    require(ids_in_progress_lines(output).size() >= static_cast<size_t>(tasks + 3),
            "concurrent progress snapshots were lost");
    for (int i = 0; i < tasks; ++i) {
        require(output.find("并发正文-" + std::to_string(i)) != std::string::npos,
                "concurrent request body was missing");
        const std::string marker = "############ 合成 #" + std::to_string(logs[i].id);
        const std::string progress_id = "[合成 #" + std::to_string(logs[i].id) + "]";
        require(output.find(marker) != std::string::npos && output.find(progress_id) != std::string::npos,
                "concurrent request did not retain its own progress ID");
    }
}

int main() {
    formatting();
    progress_output();
    std::puts("request log: GSV-style sections, UTF-8 wrapping, progress and concurrency passed");
}
