#pragma once

#include <shared_mutex>

namespace httplib { class Server; }

namespace breeze {

struct BreezeModel;
struct MimiCodec;
class GenScheduler;
class VoiceStore;
class Models;
struct ServerOptions;

void add_batch_route(httplib::Server & server, BreezeModel & model, MimiCodec & codec,
                     GenScheduler & scheduler, VoiceStore & voices, Models & models,
                     std::shared_mutex & execution_gate, const ServerOptions & options);

}
