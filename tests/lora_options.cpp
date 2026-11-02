#include "models.h"
#include "json.h"
#include "breeze/model.h"
#include "breeze/scheduler.h"

#include <cstdio>
#include <stdexcept>

using namespace breeze;

static void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

int main() {
    try {
        BreezeModel model;
        GenScheduler scheduler;
        Models models(model, scheduler, "base", {}, 2, 900);
        Models::Selection selected;
        int status;
        std::string error;
#ifdef _WIN32
        const std::string role = "E:/models/role.gguf";
        const std::string effect = "E:/models/asmr.gguf";
        const std::string third = "E:/models/third.gguf";
#else
        const std::string role = "/models/role.gguf";
        const std::string effect = "/models/asmr.gguf";
        const std::string third = "/models/third.gguf";
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
        require(parse("{\"lora_weight\":0.9,\"extra_loras\":[{\"lora_path\":\"" + effect + "\",\"weight\":1.8}]}") &&
                selected.extra_loras[0].weight == 1.8f, "weight above one rejected");
        for (const auto * bad : {"-1", "null", "true", "\"1.8\"", "1e999"})
            require(!parse(std::string("{\"lora_weight\":") + bad + "}"), "invalid weight accepted");
        require(!parse("{\"extra_loras\":{}}"), "non-array accepted");
        require(!parse("{\"extra_loras\":[{\"lora_path\":\"relative.gguf\"}]}"), "relative extra path accepted");
        require(!parse("{\"extra_loras\":[{\"lora_path\":\"" + role + "\"}]}"), "duplicate role accepted");
        require(!parse("{\"extra_loras\":[{\"lora_path\":\"" + effect + "\"},{\"lora_path\":\"" + third + "\"}]}"),
                "oversized stack accepted");
        require(parse("{\"extra_loras\":[{\"lora_path\":\"" + effect + "\"},{\"lora_path\":\"" + third +
                      "\",\"weight\":0}]}"), "zero-weight extra consumes a cache slot");
        puts("LoRA request defaults, weights, paths and capacity: OK");
        return 0;
    } catch (const std::exception & e) {
        fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
