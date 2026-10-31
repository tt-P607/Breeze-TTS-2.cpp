#pragma once

namespace httplib { class Server; }

namespace breeze {

struct BreezeModel;
struct MimiCodec;
class GenScheduler;
class VoiceStore;
class Models;
class RequestQueue;
struct ServerOptions;

void add_batch_route(httplib::Server & server, BreezeModel & model, MimiCodec & codec,
                     GenScheduler & scheduler, VoiceStore & voices, Models & models,
                     RequestQueue & execution_queue, const ServerOptions & options);

}
