#pragma once

#include "json.h"
#include "voices.h"

#include "breeze/generation.h"

#include <string>

namespace breeze {

struct OpenAiSpeech {
    GenRequest g;
    std::string model;
    std::string lora_path;
    bool has_model = false;
    bool has_lora_path = false;
    bool parallel = false;
    std::string voice; // empty means voice design
    std::string format = "mp3";
    bool sse = false;
};

// reads an openai style speech body. returns the error message and sets status, empty when it parsed
std::string parse_openai_speech(const std::string & body, VoiceStore & store, int split_chars,
                                OpenAiSpeech & out, int & status);
std::string parse_openai_speech(const JVal & body, VoiceStore & store, int split_chars,
                                OpenAiSpeech & out, int & status);

std::string openai_error(const std::string & msg, const char * type = "invalid_request_error");
std::string audio_base64(const std::string & bytes);
std::string json_quote(const std::string & value);

std::string sse_audio_delta(const std::string & bytes);
std::string sse_audio_done();

} // namespace breeze
