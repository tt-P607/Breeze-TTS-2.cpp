#pragma once

#include "voices.h"
#include "ws.h"

#include "breeze/codec.h"
#include "breeze/model.h"

#include "breeze/scheduler.h"
#include "execution.h"
#include "models.h"

#include <string>

namespace breeze {

// handles one websocket connection until it closes. a connection that finds every slot taken
// waits its turn rather than being turned away
void ws_connection(WsConn & conn, BreezeModel & model, MimiCodec & codec, VoiceStore & store,
                   GenScheduler & sched, Models & models, RequestQueue & execution_queue,
                   int chunk_first, int chunk_max, int split_chars, const std::string & base_model);

} // namespace breeze
