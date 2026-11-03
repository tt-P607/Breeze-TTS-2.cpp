# Server API

`breeze-server` is a single binary HTTP server built on cpp-httplib. It loads
one base model and can register LoRA adapters, serves them over HTTP, and streams
raw PCM as the audio is generated rather than waiting for the whole clip.

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Liveness, sample rate, websocket port |
| `POST /v1/audio/speech` | Generate speech, streamed. Also takes OpenAI style JSON, see [OpenAI compatible requests](#openai-compatible-requests) |
| `POST /v1/audio/speech/batch` | Generate a JSON batch of audio items, see [Batch requests](#batch-requests) |
| `POST /v1/audio/convert` | Respeak a recording in another voice, see [voice-conversion.md](voice-conversion.md) |
| `POST /v1/voices` | Register or save a reference voice, see [voices.md](voices.md) |
| `GET /v1/voices` | List cached and saved voices |
| `GET /v1/voices/<id>` | One voice, with its description and metadata |
| `PATCH /v1/voices/<id>` | Change a voice's description and metadata, see [voices.md](voices.md#descriptions-and-metadata) |
| `DELETE /v1/voices/<id>` | Drop a voice from memory |

A separate WebSocket listener handles incremental text, mid stream direction
changes and interruption. It is documented in [websocket.md](websocket.md).

## Starting the server

```
breeze-server <model.gguf> [--host H] [--port P] [--webui] [--cpu]
                           [--chunk-first N] [--chunk-max N] [--verbose]
                           [--voices-dir PATH] [--ws-port P] [--split-chars N]
                           [--cors [ORIGINS]] [--no-warmup] [--parallel N]
                           [--lora-cache-size N] [--lora-idle-seconds N]
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--host` | `127.0.0.1` | Interface to bind. Use `0.0.0.0` to accept remote connections. |
| `--port` | `8080` | TCP port. |
| `--webui` | off | Also serve the browser UI at `/`. |
| `--cpu` | off | Force the CPU backend instead of Vulkan. |
| `--chunk-first` | `1` | Fewest frames in the first streamed chunk. It goes up on its own when the GPU is too busy to stay ahead with that few. |
| `--chunk-max` | `25` | Frames the chunk ramps up to. |
| `--verbose` | off | Add a per stage timing breakdown after each request. |
| `--voices-dir` | `voices` | Folder of saved `.breeze` voices to load at startup. See [voices.md](voices.md). |
| `--ws-port` | HTTP port + 1 | Port for streaming sessions. `-1` disables it. See [websocket.md](websocket.md). |
| `--split-chars` | `600` | Default length long text is broken up at. `0` sends the whole thing through in one pass. A request can still override it. |
| `--cors` | off | Let browser pages on other origins call the API. On its own it allows any origin; give a comma separated list to allow only those. See [Cross origin requests](#cross-origin-requests). |
| `--no-warmup` | off | Skip the warm up at startup. See below. |
| `--parallel` | `1` | Streams generating at once. A stream using guidance counts as two. See [performance.md](performance.md#running-several-streams). |
| `--model-id` | `default` | ID of the initial full model. |
| `--switchable-lora ID=PATH` | none | Register a LoRA adapter without loading it. See [lora.md](lora.md). |
| `--lora-cache-size` | `3` | Adapters retained between requests. Busy overflow is allowed and unloaded after its last user finishes. The base model is shared. |
| `--lora-idle-seconds` | `900` | Unload an adapter after this many seconds without a request using it. |

```
breeze-server breeze-tts-2-q4_k.gguf --port 8137 --webui
```

For example, start one base model and register two LoRA adapters:

```
breeze-server base.gguf --model-id base \
  --switchable-lora voice-a=voice-a.gguf \
  --switchable-lora voice-b=voice-b.gguf
```

`GET /v1/models` lists registered IDs and their `loaded`, `in_use`, and `idle_seconds`
state. Its `active_model` value is the default for requests that do not name a model;
`lora_cache` reports `capacity`, `idle_timeout_seconds`, and the number of loaded adapters.
Adapters load on demand and idle adapters, including the current default, unload after the
configured timeout.

Model choice is per request. Native form requests use `model_id` with optional `lora_path`;
OpenAI JSON uses the standard `model` field with optional `lora_path`; WebSocket `start` uses
`model_id` and optional `lora_path`. A `lora_path` must be an absolute path on the server and is accepted only from localhost.
Unknown IDs and failed loads return errors rather than falling back to the default or base model.
Different LoRAs can generate concurrently when requests explicitly use `execution=parallel` and
`--parallel` allows it; requests otherwise run serially.

`POST /v1/models/active` only changes the default for later requests without an explicit model.
It does not affect requests already running or WebSocket sessions that have started. It only
accepts localhost requests. Supplying `lora_path` with `model_id` registers that ID after the
adapter loads successfully. See [lora.md](lora.md) for the request fields and cache behavior.

Before it starts listening the server runs a short throwaway generation for every
mix of streams its `--parallel` setting allows, with and without guidance, and
decodes a few sizes of audio chunk. That sets up the GPU pipelines the first real
request would otherwise have to build. The first request after startup then reaches
first audio in about 200 to 350 ms on an RTX 3060, and later ones in about 80 ms. It
adds about 3 s to startup at `--parallel 1` and about 8 s at `--parallel 4`.
`--no-warmup` skips it.

## Console output

Each request prints a progress bar that fills as audio is streamed, then a
summary.

```
gen  design, 291 chars, cfg 1.0, seed 7
 87%|████████████████████▉   | 15.1/17.4s [00:11<00:01, 16.2 fps, 1.30x]
100%|████████████████████████| 19.4/19.4s [00:15, 16.1 fps, 1.29x]
     242 frames in 11 flushes, first audio 588 ms
```

The total is estimated from the text, since the model decides for itself when to
stop, so the bar can reach 100% early or run past its estimate. The last line
rewrites it with the real figure. `fps` is frames generated per second against
12.5 frames of audio per second, so the trailing number is the real time factor
and anything above `1.00x` is faster than playback.

With `--parallel` above 1 the moving bar is left out, because several streams would
write over each other. The final bar and the summary still print for each request.

There is no authentication and no rate limiting, on either the HTTP port or the
WebSocket one. Do not expose them directly to the internet; put them behind a
reverse proxy that handles both.

## Cross origin requests

The bundled web UI is served from the same origin as the API, so it needs
nothing. A page served from anywhere else, a Vite dev server or a separate front
end for instance, is blocked by the browser unless the server sends CORS
headers. It does not by default. Turn them on with `--cors`:

```
breeze-server model.gguf --cors
breeze-server model.gguf --cors http://localhost:5173,https://app.example.com
```

On its own the flag answers every origin with `Access-Control-Allow-Origin: *`.
With a list it reflects the request's `Origin` only when it matches an entry
exactly, scheme and port included, and adds `Vary: Origin` so caches keep the
answers apart. An origin that is not on the list gets no CORS headers and the
browser refuses the response.

`X-Sample-Rate` and `X-Sample-Format` are listed in
`Access-Control-Expose-Headers`, so a cross origin page can read the sample
rate the same way the bundled UI does. Preflight `OPTIONS` requests, which the
browser sends before `DELETE` and `PATCH` on `/v1/voices/<id>`, are answered
with `204` and cached for a day.

The WebSocket port is not affected. Browsers do not apply CORS to WebSockets,
and the socket accepts a connection from any page regardless of this flag, as
[websocket.md](websocket.md) notes.

## Streaming without stutter

Two numbers decide whether a stream plays cleanly, and they fail in different
ways. Getting one right does not save you from the other.

### Real time factor

The real time factor is seconds of audio produced per second of wall clock. Above
1.0 the model outruns playback, below it no amount of buffering will help because
the client drains faster than the server fills.

Measure it against your own hardware rather than assuming:

```
curl -s -o out.pcm -w "%{time_total}\n" \
  --form-string "text=<a sentence long enough to take several seconds>" \
  http://127.0.0.1:8137/v1/audio/speech
```

Audio seconds are `bytes / 2 / 24000`, so the factor is that divided by
`time_total`. On an RTX 3060 a long clip measures about 2.4x at Q4_K and 2x at
Q8_0 with the defaults. A busy GPU still eats into that, and so do other streams,
see [performance.md](performance.md#running-several-streams).

Guidance costs less than it looks like it should. `cfg_scale` above 1 adds a
second forward pass, but the depth decoder batches both branches into one graph,
so a 16 second passage at `cfg_scale` 3 measures 1.74x at Q4_K and 1.59x at
Q8_0, still comfortably above realtime. Cloning also adds a one off
reference encode of roughly 650 ms, which shows up in time to first audio rather
than in the rate.

If you land near or below 1.0, no client setting will fix it. Drop to a smaller
quantisation, lower `--parallel`, or use a faster device. Raising `--chunk-max` no
longer helps here. Quantising the depth
decoder with `--depth` also helps, but it damages the high end and the damage
grows over a long passage, so weigh it against the numbers in
[models.md](models.md).

### Queue depth

This is the one that actually causes stutter on a machine whose real time factor
looks fine.

The client's queue drains continuously but refills in one lump per chunk, so the
queue has to be deeper than the time it takes to produce a whole chunk. At
`--chunk-max 40` a chunk is 3.2 s of audio that takes roughly 2 s to generate,
so a client holding only 0.5 s of audio runs dry waiting for it, even though the
average rate is comfortably ahead.

The server now handles most of this itself. It assumes playback starts the
moment each chunk lands and keeps track of when that audio will run out. If
waiting to fill the current chunk would leave the client with nothing to play,
it sends the frames it already has early instead, and holds the chunk size until
the cushion builds back up. It only does this when the short chunk plays for
longer than it takes to generate and decode, because otherwise a run of tiny
chunks would fall further behind rather than catch up. The time it goes by includes
any other streams sharing the GPU.

Measured on an RTX 3060 at Q8_0, when early flushing went in, with a client that
starts playing on the very first packet and buffers nothing, reading a 21 second
passage at `cfg_scale` 3, the slowest common case:

| `--chunk-max` | Before | Early flushing | Plus the one graph depth decoder |
| --- | --- | --- | --- |
| 25 | 5 gaps, 222 ms of silence | none | none |
| 40 | 6 gaps, 308 ms | 1 gap, 73 ms | none |
| 60 | 7 gaps, 442 ms | 1 gap, 18 ms | none |

Without guidance there were no gaps in any of them. The early flushes cost a few
percent of throughput under guidance, 1.17x against 1.22x at the default, and the
faster depth decoder more than wins it back at 1.33x. The current build reads a 16
second passage this way at 1.59x with no gaps.

The old rule still gives the most margin: **buffer more audio than the slowest
chunk takes to produce.** The bundled UI prebuffers 1.25 s. It just matters much
less now, and a client that cannot buffer at all will mostly get away with it.

Note the difference between prebuffering and delaying. Starting the first chunk
half a second late does not create a cushion, it only shifts the start; the queue
still holds one chunk. Accumulate chunks until you are actually holding N seconds
of audio, then play them back to back. The bundled UI does this and exposes N as
the buffer slider.

### Tuning the chunk ramp

Audio is vocoded in chunks of whole frames at 12.5 frames per second. The first
chunk sets how long the client waits for sound, so the chunk starts small and
grows by a third each flush until it reaches `--chunk-max`.

The vocoder keeps its state between flushes and only decodes the new frames, so
small chunks are cheap now and raising the ceiling no longer buys anything. Measured
on an RTX 3060 at Q8_0 over an 18 second clip:

| `--chunk-max` | Flushes | Vocoder per frame |
| --- | --- | --- |
| 8 | 31 | 2.90 ms |
| 25 | 15 | 2.17 ms |
| 50 | 13 | 2.16 ms |

Lowering it is the useful direction if a client wants a smaller buffer. Going to 8
costs about 2% of total speed.

`--chunk-first` is a minimum, and the default of 1 frame is the fastest start. The
server raises the first chunk by itself when the GPU is too busy for that. Setting 4
took first audio from 278 ms to 394 ms in `breeze-cli` at Q8_0, and saved only 0.14 ms
per frame.

Setting both flags to the same value disables the ramp and streams a fixed size.
[performance.md](performance.md#tuning-settings) has the rest.

### Summary

- Leave `--chunk-max` at the default, or lower it if you want a smaller client buffer.
- The server sends early when a client is about to run dry, so a small buffer is
  enough. Around 1 s still gives the most margin under guidance.
- Prebuffer by fill level, not by delay.
- Check the real time factor first if it stutters no matter what you buffer.

## `GET /health`

Liveness probe. Returns once the model is loaded and ready.

```
curl http://127.0.0.1:8137/health
```

```json
{"status":"ok","sample_rate":24000,"ws_port":8081}
```

The server does not accept connections until loading finishes, so a successful
response also means the model is warm. Poll this after startup instead of
guessing at a delay.

`ws_port` is where the streaming socket ended up, or `0` when it is disabled, so
clients can discover it rather than being configured with it.

## `POST /v1/audio/speech`

Generates speech and streams it back.

Accepts `multipart/form-data` (needed for the reference audio upload) or
`application/x-www-form-urlencoded`.

### Fields

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `text` | string | required | Text to speak, UTF-8. |
| `model_id` | string | current default | Registered model ID for this request. |
| `lora_path` | string | none | Absolute local adapter path. With `model_id`, registers or updates that ID after a successful load. |
| `execution` | string | `serial` | `serial` excludes other TTS requests; `parallel` may overlap requests that also explicitly use `parallel`. |
| `instruction` | string | `Speak clearly and naturally.` | Voice description or delivery direction. |
| `ref_audio` | file | none | Reference WAV for cloning. Any sample rate or channel count; resampled to mono at the model rate. Identical repeated uploads (same bytes and transcript) reuse the reference encode from the in-memory hash cache instead of re-encoding; see [voices.md](voices.md). |
| `ref_text` | string | empty | Exact transcript of `ref_audio`. Required whenever `ref_audio` is present. |
| `voice_id` | string | none | A saved or cached voice to clone instead of uploading a clip. Skips the reference encode. See [voices.md](voices.md). |
| `cfg_scale` | float | `1.0` | Classifier free guidance. `1.0` disables it. |
| `seed` | int | `42` | RNG seed. |
| `temperature` | float | model default | Sampling temperature. `0` keeps whatever the GGUF was built with. |
| `top_k` | int | model default | Sampling top-k. `0` keeps the model default. |
| `top_p` | float | model default | Sampling top-p. `0` keeps the model default. |
| `repetition_penalty` | float | model default | Repetition penalty. `0` keeps the model default. |
| `split_chars` | int | `--split-chars` | Long text is split on sentence boundaries into pieces of about this size and generated one at a time. `0` generates in one pass. |
| `max_new_tokens` | int | model default | Frame cap per piece, 12.5 frames per second. `0` uses the model default of 750. |

Every sampling field treats `0` as "use the model default", so leaving them out
behaves exactly as before rather than forcing a zero.

There is no length limit on `text`. Anything past the budget is split and
generated piece by piece, each conditioned on the first so the voice does not
change partway. Generating minutes in a single pass is what `split_chars 0`
does, and the model loses track of the text well before it runs out of frames,
so leave splitting on unless you have a reason not to. Starting the server with
`--split-chars 0` turns it off for every request that does not set its own.

### Response

`200 OK` with a chunked body.

| Header | Value |
| --- | --- |
| `Content-Type` | `audio/pcm` |
| `Transfer-Encoding` | `chunked` |
| `X-Sample-Rate` | `24000` |
| `X-Sample-Format` | `s16le` |
| `Cache-Control` | `no-store` |

The body is **headerless** signed 16 bit little endian mono PCM. It is not a
WAV file. Wrap it yourself if you need one, or use the CLI which writes WAV
directly.

Chunks arrive roughly every 2 seconds of generated audio, so playback can start
long before generation finishes.

### Errors

| Status | Body | Cause |
| --- | --- | --- |
| `400` | `{"error":"text is required"}` | `text` missing or empty. |
| `403` | `{"error":"lora_path is restricted to localhost requests"}` | A request from another host supplied `lora_path`. |
| `404` | `{"error":"unknown model_id"}` or `{"error":"unknown voice_id"}` | The model ID or reference voice is not registered. |
| `422` | `{"error":"failed to load LoRA adapter"}` | The selected adapter could not be loaded. |

The server shares one base model. Requests default to `execution=serial`, which excludes
other TTS generation requests until the whole HTTP response finishes. Set `execution=parallel`
to allow overlap with other explicitly parallel requests. `--parallel` sets the GPU slot limit;
a request using guidance takes two slots, see
[performance.md](performance.md#running-several-streams). HTTP, WebSocket, batch items, and
uncached reference encoding share one server-side FIFO queue. Requests wait for their turn
and sufficient slots; clients need no local concurrency limit or busy-retry loop. A waiting
serial request blocks later parallel requests from overtaking it. Parallel requests may use
different LoRAs; the scheduler groups frames by adapter. A disconnected or cancelled waiter
is removed without generating audio. Client timeouts must include queueing time.

Because the response is streamed, a failure that happens **after** the first
chunk cannot change the status code. The connection is closed early instead, so
treat a truncated body as an error.

### Examples

Send text fields with `--form-string`, not `-F`. `curl` reads a value that starts
with `(` as the opening of a multipart group, so `-F "text=(sigh) ..."` posts an
empty field and the model ends up reading the multipart boundary out loud.

Voice design:

```
curl -X POST http://127.0.0.1:8137/v1/audio/speech \
  --form-string "text=Welcome aboard. Your journey begins now." \
  --form-string "instruction=A warm, thoughtful young woman with a clear, calm delivery." \
  --form-string "cfg_scale=1" \
  --form-string "seed=42" \
  -o speech.pcm
```

Voice clone:

```
curl -X POST http://127.0.0.1:8137/v1/audio/speech \
  --form-string "text=It is good to hear your voice again." \
  -F "ref_audio=@reference.wav" \
  --form-string "ref_text=This is the exact transcript of the reference audio." \
  -o clone.pcm
```

Voice direction, which is a clone plus an instruction:

```
curl -X POST http://127.0.0.1:8137/v1/audio/speech \
  --form-string "text=We need to discuss what happened last night." \
  --form-string "instruction=Speak slowly with a restrained, serious tone." \
  -F "ref_audio=@reference.wav" \
  --form-string "ref_text=This is the exact transcript of the reference audio." \
  --form-string "cfg_scale=1" \
  -o direction.pcm
```

Play the raw stream straight out of `curl`:

```
curl -sN -X POST http://127.0.0.1:8137/v1/audio/speech --form-string "text=Hello there." \
  | ffplay -f s16le -ar 24000 -ac 1 -nodisp -autoexit -
```

Convert to WAV:

```
ffmpeg -f s16le -ar 24000 -ac 1 -i speech.pcm speech.wav
```

## OpenAI compatible requests

`POST /v1/audio/speech` also takes the JSON body of OpenAI's speech API, so
apps built for it, like SillyTavern or the official OpenAI SDKs, can point at
the server. Set `model` to a registered ID, or omit it to use the current default.
The `Content-Type` header decides which request format you get:
`application/json` gets the behaviour described here, a form post gets the
native behaviour above.

```
curl http://127.0.0.1:8137/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"base","input":"Hello from Breeze.","voice":"narrator","response_format":"mp3"}' \
  -o speech.mp3
```

With the Python SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8137/v1", api_key="unused")
with client.audio.speech.with_streaming_response.create(
    model="base", voice="narrator", input="Streaming from the OpenAI SDK.",
) as r:
    r.stream_to_file("speech.mp3")
```

### Fields

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `input` | string | required | Text to speak, UTF-8. |
| `voice` | string or `{"id": ...}` | none | A saved or cached voice, the same ids `voice_id` takes. Leave it out to design a voice from `instructions` instead. |
| `instructions` | string | `Speak clearly and naturally.` | Voice description when designing, delivery direction when cloning. |
| `response_format` | string | `mp3` | `mp3`, `wav` or `pcm`. `opus`, `aac` and `flac` are accepted but sent as WAV. |
| `stream_format` | string | `audio` | `audio` streams the file itself, `sse` wraps it in server sent events. |
| `model` | string | current default | Registered model ID for this request. This is the standard OpenAI field; `model_id` is not accepted as an alias in JSON. |
| `lora_path` | string | none | Absolute local adapter path. With `model`, registers or updates that ID after a successful load. |
| `execution` | string | `serial` | `serial` or `parallel`, with the same behavior as the native form field. |
| `speed` | number | ignored | The model has no speed control. |

The native fields `cfg_scale`, `seed`, `temperature`, `top_k`, `top_p`,
`repetition_penalty`, `max_new_tokens` and `split_chars` are accepted as extra
JSON fields, with the same meaning and defaults as above.

OpenAI's own voice names such as `alloy` or `nova` are not built in, and a name
the server does not know is a `404`. If a client insists on those names, save
voices under them with `POST /v1/voices` (see [voices.md](voices.md)) and they
work like any other.

### Formats

| `response_format` | Sent as | `Content-Type` |
| --- | --- | --- |
| `mp3` | MP3, 80 kbps mono at 48 kHz | `audio/mpeg` |
| `wav` | 16 bit mono WAV at 24 kHz | `audio/wav` |
| `pcm` | Headerless s16le mono at 24 kHz, the same as the native endpoint and OpenAI's own `pcm` | `audio/pcm` |
| `opus`, `aac`, `flac` | WAV | `audio/wav` |

Every format streams as it is generated, so playback can start on the first
chunk. The WAV header goes out first with its length fields set to the maximum,
because the length is not known yet. Browsers and ffmpeg read that to the end
of the stream.

MP3 is encoded with [shine](https://github.com/toots/shine), which builds from
the `third_party/shine` submodule. It is upsampled to 48 kHz first, because Chrome
and Edge reject the frames shine writes at 24 kHz, so `X-Sample-Rate` says 48000
on an MP3 response. A build without it, or one configured with
`-DBREEZE_MP3=OFF`, sends WAV when asked for `mp3` as well. The console line for
each request says what actually went out, for example `openai wav for opus`.

### Server sent events

With `"stream_format": "sse"` the response is `text/event-stream`. Each chunk
arrives as a `speech.audio.delta` event carrying the next bytes of the file in
`response_format`, base64 encoded, and a `speech.audio.done` event ends it.

```
data: {"type":"speech.audio.delta","audio":"//NIxAAAAAAAAAAAAFhpbmcAAAAPAAAA..."}

data: {"type":"speech.audio.done","usage":{"input_tokens":0,"output_tokens":0,"total_tokens":0}}
```

Decoding the deltas and joining them gives the same file a plain request
returns. The token counts are always zero.

### Errors

Errors on this path use OpenAI's error shape, so clients show the message
instead of a bare status code.

```json
{"error":{"message":"input is required","type":"invalid_request_error","param":null,"code":null}}
```

| Status | Cause |
| --- | --- |
| `400` | The body is not a JSON object, `input` is missing, or a field has the wrong type or an invalid path. |
| `403` | A request from another host supplied `lora_path`. |
| `404` | `voice` is not saved or cached, or `model` is not registered. |
| `422` | The selected LoRA failed to load. |

The `Authorization` header is ignored, so any API key works. That also means
there is still no authentication, see the warning above.

## Batch requests

`POST /v1/audio/speech/batch` accepts JSON with one `execution` mode and an `items` array.
Submit between 1 and 16 items. Each item uses the OpenAI speech fields such as `input`, `model`, `voice`, `response_format`,
and `lora_path`; the top-level mode applies to the whole batch, so items do not set `execution`.
The default is serial. Explicit parallel mode may run items concurrently, subject to the
server's `--parallel` slot limit and available LoRA cache capacity.

```json
{
  "execution": "parallel",
  "items": [
    {"id": "a", "model": "voice-a", "input": "First line.", "voice": "narrator", "response_format": "wav"},
    {"id": "b", "model": "voice-b", "input": "Second line.", "voice": "narrator", "response_format": "wav"}
  ]
}
```

The JSON response contains `execution`, `elapsed_ms`, and `results` in input order.
Successful results report `id`, `model`, `status: "ok"`, `format: "wav"`, `sample_rate`,
`first_audio_ms`, `wall_ms`, `duration_seconds`, and `audio_base64`. Decode `audio_base64`
to get a complete WAV file. Output is always WAV; omit `response_format` or set it to `wav`.
Item IDs are optional and must be unique; omitted IDs are zero-based index strings.

Invalid syntax or voice settings reject the batch before generation. Model loading or generation
failures are returned for the affected item as `status: "error"`, `status_code`, and `error`;
other items continue. The default model is captured once when the batch is accepted.
Each item enters the shared server queue. Serial batches submit one item at a time;
parallel batches submit up to the slot limit. Other clients may run between batch items,
so a large batch does not reserve the service until all its audio has finished.

The batch endpoint returns after all items finish. Its `first_audio_ms` measures internal
generation timing, not when the caller receives audio. Use the single speech endpoint or
WebSocket for streamed playback. This endpoint does not mix the resulting audio files.

### SillyTavern

Open the TTS extension, pick **OpenAI Compatible** as the provider and fill in:

| Setting | Value |
| --- | --- |
| Provider Endpoint | `http://127.0.0.1:8137/v1/audio/speech` |
| API Key | Anything, it is ignored. |
| Model | A registered model ID such as `base` or `voice-a`. |
| Available Voices | The voice ids you want, comma separated, e.g. `narrator,harbour`. The list from `GET /v1/voices` has them all. |

The default voice list is OpenAI's names, which will all come back `404` until
you replace it. Speed does nothing. SillyTavern asks for MP3, which is what it
gets.

## Streaming client sketch

```python
import struct
import urllib.request

body, boundary = build_multipart({"text": "Streaming from python."})
req = urllib.request.Request(
    "http://127.0.0.1:8137/v1/audio/speech",
    data=body,
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
)

with urllib.request.urlopen(req) as r:
    rate = int(r.headers["X-Sample-Rate"])
    tail = b""
    while chunk := r.read(8192):
        chunk = tail + chunk
        n = len(chunk) // 2
        tail = chunk[n * 2:]
        samples = struct.unpack(f"<{n}h", chunk[:n * 2])
        play(samples, rate)
```

Buffer any odd trailing byte between reads, since a chunk boundary can land in
the middle of a sample.

## Web UI

`--webui` mounts a dark brutalist single page UI that mirrors the reference
Gradio demo.

| Route | Serves |
| --- | --- |
| `GET /` | `index.html` |
| `GET /style.css` | stylesheet |
| `GET /app.js` | client script |

It has three tabs matching the three generation modes, and it posts to the same
`/v1/audio/speech` endpoint.

**Stream while generating** is on by default. The UI reads the response body
incrementally and schedules each PCM chunk through the Web Audio API as it
arrives, so playback starts about a second in rather than after the whole clip is
rendered. A running counter shows how much audio has been produced, and appends a
rebuffer count if the queue ever ran dry. When generation finishes the same PCM
is wrapped into a WAV blob for the player and the download link.

**Buffer** is how much audio the UI holds before it starts playing, not how long
it waits. Chunks accumulate until that much is queued, then play back to back.
The default of 1.25 s covers the defaults here; raise it if you raised
`--chunk-max` or see rebuffers, lower it if you want a faster start and your real
time factor has room. If the queue does run dry the UI widens its own buffer for
the rest of that clip rather than stuttering repeatedly.

Turn the toggle off to buffer the whole response first and play it back from the
`<audio>` element, which is the better choice if you want to scrub the result
rather than hear it as early as possible.

The assets are compiled into the binary at build time from the
[webui](../webui) folder by `cmake/embed_webui.cmake`, so there is no static
directory to deploy. Editing a file there and rebuilding is enough to pick up
changes.

## Request logs

Speech requests print a numbered block to stdout with the full text, instruction,
base GGUF path, selected model ID, nonzero LoRA paths and weights, reference
voice ID, execution mode, CFG and seed. Queue, inference and terminal events
reuse that number and include elapsed time. WebSocket text is logged per spoken
piece with the instruction captured for that piece; batch items have separate
numbers. These logs are always enabled and flushed immediately, so they also
appear when the server is hosted by a process manager. Control characters are
escaped; Chinese text is kept readable.
