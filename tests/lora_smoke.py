"""Exercise per-request LoRA selection, concurrency, expiry, WebSocket reuse, and LRU.

Example::

    python tests/lora_smoke.py --server build/breeze-server --model base.gguf \
        --adapter-a a.gguf --adapter-b b.gguf --output-dir test-output \
        --idle-seconds 3 --extra-adapter c.gguf --extra-adapter d.gguf \
        --extra-adapter e.gguf --extra-adapter f.gguf --extra-adapter g.gguf

The test uses real local model files but does not download or modify them. It writes
PCM WAV samples, result.json, and server.log under --output-dir.
"""

import argparse
import base64
import hashlib
import io
import json
import os
import socket
import subprocess
import sys
import threading
import time
import traceback
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from websockets.sync.client import connect


SHORT_TEXT = "你好呀，今天过得怎么样？"
SHORT_INSTRUCTION = "自然温柔地说话。"
HTTP_TIMEOUT = 240


class SmokeFailure(RuntimeError):
    """A user-facing integration check failed."""


def require(condition, message):
    if not condition:
        raise SmokeFailure(message)


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("server", "model", "adapter-a", "adapter-b", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--extra-adapter",
        type=Path,
        action="append",
        default=[],
        help="additional real LoRA path; repeat to exercise the six-entry LRU",
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--cache-size", type=int, default=6)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument(
        "--idle-seconds",
        type=int,
        default=3,
        help="test server idle timeout (use 2 or 3 for a quick expiry check)",
    )
    args = parser.parse_args()
    if args.parallel < 4:
        parser.error("--parallel must be at least 4 for the concurrent model checks")
    if args.cache_size < 1:
        parser.error("--cache-size must be positive")
    if args.idle_seconds < 1:
        parser.error("--idle-seconds must be positive")
    return args


def resolved_file(path, label):
    result = path.expanduser().resolve()
    if not result.is_file():
        raise SmokeFailure(f"{label} is not an existing file: {result}")
    return result


class ServerProcess:
    def __init__(self, args, output_dir, adapters, report):
        self.args = args
        self.output_dir = output_dir
        self.adapters = adapters
        self.report = report
        self.proc = None
        self.log_file = None
        self.session = requests.Session()
        self.session.trust_env = False
        self.url = ""
        self.ws_url = ""
        self.log_path = output_dir / "server.log"

    def start(self, idle_seconds, run_name):
        http_port = free_port()
        ws_port = free_port()
        while ws_port == http_port:
            ws_port = free_port()
        self.url = f"http://127.0.0.1:{http_port}"
        self.ws_url = f"ws://127.0.0.1:{ws_port}"
        command = [
            str(self.args.server),
            str(self.args.model),
            "--model-id",
            "base",
            "--port",
            str(http_port),
            "--ws-port",
            str(ws_port),
            "--no-warmup",
            "--voices-dir",
            str((self.output_dir / "voices").resolve()),
            "--lora-cache-size",
            str(self.args.cache_size),
            "--lora-idle-seconds",
            str(idle_seconds),
            "--parallel",
            str(self.args.parallel),
            # Keep packet boundaries fixed when comparing PCM across adapter schedules.
            "--chunk-first",
            "1",
            "--chunk-max",
            "1",
        ]
        for model_id, adapter_path in self.adapters.items():
            command.extend(("--switchable-lora", f"{model_id}={adapter_path}"))
        if self.args.cpu:
            command.append("--cpu")

        env = os.environ.copy()
        server_dir = str(self.args.server.parent)
        env["PATH"] = server_dir + os.pathsep + env.get("PATH", "")
        self.log_file = self.log_path.open("a", encoding="utf-8")
        self.log_file.write(f"\n===== {run_name}: idle timeout {idle_seconds}s =====\n")
        self.log_file.flush()
        self.proc = subprocess.Popen(
            command,
            env=env,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
        )
        run = {
            "name": run_name,
            "idle_timeout_seconds": idle_seconds,
            "command": command,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        self.report["server_runs"].append(run)

        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise SmokeFailure(
                    f"server exited with {self.proc.returncode}; see {self.log_path}"
                )
            try:
                if self.session.get(self.url + "/health", timeout=1).ok:
                    return
            except requests.RequestException:
                pass
            time.sleep(0.25)
        raise TimeoutError(f"server did not become ready; see {self.log_path}")

    def stop(self):
        if self.proc is not None:
            if self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait(timeout=10)
            if self.report["server_runs"]:
                self.report["server_runs"][-1]["exit_code"] = self.proc.returncode
            self.proc = None
        if self.log_file is not None:
            self.log_file.flush()
            self.log_file.close()
            self.log_file = None

    def close(self):
        self.stop()
        self.session.close()

    def models(self):
        response = self.session.get(self.url + "/v1/models", timeout=10)
        response.raise_for_status()
        body = response.json()
        require(body.get("object") == "list", "GET /v1/models object must be 'list'")
        require(isinstance(body.get("data"), list), "GET /v1/models data must be a list")
        return body

    def model_rows(self):
        return {row["id"]: row for row in self.models()["data"]}

    def set_default(self, model_id):
        response = self.session.post(
            self.url + "/v1/models/active",
            json={"model_id": model_id},
            timeout=15,
        )
        require(
            response.ok,
            f"POST /v1/models/active {model_id!r} failed: "
            f"HTTP {response.status_code}: {response.text[:500]}",
        )
        body = response.json()
        require(body.get("active_model") == model_id, "active model response mismatch")
        return body


def open_speech(url, form, timeout=HTTP_TIMEOUT):
    client = requests.Session()
    client.trust_env = False
    try:
        response = client.post(
            url + "/v1/audio/speech",
            data=form,
            timeout=(10, timeout),
            stream=True,
        )
    except Exception:
        client.close()
        raise
    return client, response


def speech_form(
    model_id=None,
    lora_path=None,
    max_new_tokens=8,
    text=SHORT_TEXT,
    execution=None,
):
    form = {
        "text": text,
        "instruction": SHORT_INSTRUCTION,
        "cfg_scale": "1",
        "seed": "42",
        "split_chars": "0",
        "max_new_tokens": str(max_new_tokens),
    }
    if model_id is not None:
        form["model_id"] = model_id
    if lora_path is not None:
        form["lora_path"] = str(lora_path)
    if execution is not None:
        form["execution"] = execution
    return form


def save_pcm(path, pcm, sample_rate):
    with wave.open(str(path), "wb") as output:
        output.setparams((1, 2, sample_rate, 0, "NONE", "not compressed"))
        output.writeframes(pcm)


def synthesize(server, report, report_lock, label, model_id, lora_path=None,
               max_new_tokens=8, text=SHORT_TEXT, save_audio=True, timeout=HTTP_TIMEOUT,
               execution=None):
    started = time.monotonic()
    client, response = open_speech(
        server.url,
        speech_form(model_id, lora_path, max_new_tokens, text, execution),
        timeout=timeout,
    )
    try:
        if not response.ok:
            body = response.text[:1000]
            raise SmokeFailure(
                f"speech {label} failed: HTTP {response.status_code}: {body}"
            )
        sample_rate = int(response.headers.get("X-Sample-Rate", "0"))
        pcm = b"".join(response.iter_content(4096))
        require(sample_rate > 0, f"speech {label} omitted X-Sample-Rate")
        require(pcm and len(pcm) % 2 == 0, f"speech {label} returned invalid PCM16")
        elapsed = time.monotonic() - started
        item = {
            "label": label,
            "model_id": model_id,
            "lora_path": str(lora_path) if lora_path is not None else None,
            "sha256": hashlib.sha256(pcm).hexdigest(),
            "sample_rate": sample_rate,
            "pcm_bytes": len(pcm),
            "audio_seconds": len(pcm) / (sample_rate * 2),
            "wall_seconds": round(elapsed, 3),
        }
        if save_audio:
            filename = label.replace("/", "-") + ".wav"
            save_pcm(server.output_dir / filename, pcm, sample_rate)
            item["wav"] = filename
        with report_lock:
            report["audio"].append(item)
        return pcm, item
    finally:
        response.close()
        client.close()


def expect_bad_speech(server, form, label, expected_status=None):
    client, response = open_speech(server.url, form, timeout=30)
    try:
        body = response.text[:1000]
        if expected_status is not None:
            require(
                response.status_code == expected_status,
                f"{label} should return HTTP {expected_status}, got "
                f"HTTP {response.status_code}: {body}",
            )
        else:
            require(
                400 <= response.status_code < 500,
                f"{label} should return a client error, got HTTP "
                f"{response.status_code}: {body}",
            )
        return {"label": label, "status": response.status_code, "body": body}
    finally:
        response.close()
        client.close()


def append_check(report, name, details=None, optional=False):
    entry = {"name": name, "status": "passed", "details": details or {}}
    report["optional_checks" if optional else "checks"].append(entry)
    return entry


def wait_for_model(server, model_id, predicate, timeout, reason):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        rows = server.model_rows()
        last = rows.get(model_id)
        if last is not None and predicate(last):
            return last
        time.sleep(0.1)
    raise TimeoutError(f"{reason}; last state for {model_id}: {last}")


def wait_for_unload(server, model_id, idle_seconds):
    # The reaper runs at min(TTL, 30s); polling /v1/models must not extend the TTL.
    timeout = max(8, 2 * idle_seconds + 4)
    return wait_for_model(
        server,
        model_id,
        lambda row: not row["loaded"] and not row["in_use"],
        timeout,
        "adapter did not unload after its idle timeout",
    )


def receive_ws_started(connection):
    while True:
        message = connection.recv(timeout=30)
        if isinstance(message, bytes):
            continue
        event = json.loads(message)
        if event.get("type") == "error":
            raise SmokeFailure(f"WebSocket start failed: {event}")
        if event.get("type") == "started":
            return event


def receive_ws_audio_until_done(connection, timeout=HTTP_TIMEOUT):
    audio = bytearray()
    events = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            message = connection.recv(timeout=min(10, max(0.1, deadline - time.monotonic())))
        except TimeoutError:
            continue
        if isinstance(message, bytes):
            audio.extend(message)
            continue
        event = json.loads(message)
        events.append(event.get("type", ""))
        if event.get("type") == "error":
            raise SmokeFailure(f"WebSocket synthesis failed: {event}")
        if event.get("type") == "done":
            break
    else:
        raise TimeoutError("WebSocket synthesis did not send done")
    require(audio and len(audio) % 2 == 0, "WebSocket returned invalid PCM16")
    return bytes(audio), events


def websocket_sample(server, model_id, text, flush=False):
    with connect(server.ws_url, open_timeout=15, close_timeout=5) as connection:
        ready = json.loads(connection.recv(timeout=15))
        require(ready.get("type") == "ready", f"unexpected WebSocket greeting: {ready}")
        connection.send(
            json.dumps(
                {
                    "type": "start",
                    "model_id": model_id,
                    "instruction": SHORT_INSTRUCTION,
                    "seed": 42,
                    "cfg_scale": 1,
                    "split_chars": 0,
                    "max_new_tokens": 8,
                    "execution": "serial",
                }
            )
        )
        receive_ws_started(connection)
        if flush:
            connection.send(json.dumps({"type": "flush", "text": text}))
            connection.send(json.dumps({"type": "end"}))
        else:
            connection.send(json.dumps({"type": "end", "text": text}))
        return receive_ws_audio_until_done(connection)


def check_startup_registration(server, report, adapters, idle_seconds, cache_size):
    state = server.models()
    cache = state.get("lora_cache", {})
    rows = {row["id"]: row for row in state["data"]}
    expected_ids = {"base", *adapters}
    require(state.get("active_model") == "base", "server did not start with base selected")
    require(cache.get("capacity") == cache_size, "server reported the wrong LoRA cache size")
    require(
        cache.get("idle_timeout_seconds") == idle_seconds,
        "server reported the wrong LoRA idle timeout",
    )
    require(cache.get("loaded") == 0, "LoRA weights were loaded during startup")
    require(expected_ids.issubset(rows), "startup adapter IDs were not all registered")
    require(rows["base"]["loaded"], "base model should be resident at startup")
    for model_id in adapters:
        require(
            not rows[model_id]["loaded"] and not rows[model_id]["in_use"],
            f"LoRA {model_id} should be registered but not preloaded",
        )
    append_check(
        report,
        "startup registers LoRAs without preloading",
        {"registered_ids": sorted(expected_ids), "loaded_lora_count": cache["loaded"]},
    )


def test_serial_and_dynamic_paths(server, report, report_lock, adapter_paths):
    pcm_a, _ = synthesize(server, report, report_lock, "serial-a", "voice-a")
    pcm_b, _ = synthesize(server, report, report_lock, "serial-b", "voice-b")
    require(pcm_a != pcm_b, "different LoRAs produced identical serial PCM")

    alias_path = adapter_paths["voice-a"]
    alias, _ = synthesize(
        server,
        report,
        report_lock,
        "dynamic-alias-a",
        "voice-a-alias",
        lora_path=alias_path,
    )
    require(alias == pcm_a, "request-level lora_path did not reproduce its adapter")
    rows = server.model_rows()
    require("voice-a-alias" in rows, "request-level lora_path did not register its ID")
    require(rows["voice-a-alias"]["loaded"], "request-level LoRA should be resident after use")
    require(server.models()["active_model"] == "base", "per-request selection changed the default")
    append_check(
        report,
        "serial A/B baselines and request-level path registration",
        {
            "a_sha256": hashlib.sha256(pcm_a).hexdigest(),
            "b_sha256": hashlib.sha256(pcm_b).hexdigest(),
            "dynamic_alias_sha256": hashlib.sha256(alias).hexdigest(),
        },
    )
    return pcm_a, pcm_b


def test_serial_gate(server, report, report_lock):
    token_count = 16
    text = "这段短文本用于验证串行请求的执行门控。"
    with ThreadPoolExecutor(max_workers=1) as pool:
        active = pool.submit(
            synthesize,
            server,
            report,
            report_lock,
            "serial-gate-holder",
            "voice-a",
            None,
            token_count,
            text,
            False,
            HTTP_TIMEOUT,
            "serial",
        )
        wait_for_model(
            server,
            "voice-a",
            lambda row: row["loaded"] and row["in_use"],
            timeout=60,
            reason="serial holder did not acquire adapter A",
        )
        serial_rejected = expect_bad_speech(
            server,
            speech_form("voice-b", execution="serial"),
            "second serial request while serial generation is active",
            expected_status=409,
        )
        parallel_rejected = expect_bad_speech(
            server,
            speech_form("voice-b", execution="parallel"),
            "parallel request while serial generation is active",
            expected_status=409,
        )
        require(
            server.model_rows()["voice-a"]["in_use"],
            "serial holder ended before both busy responses were checked",
        )
        pcm, _ = active.result(timeout=HTTP_TIMEOUT + 30)
    require(pcm, "serial holder request returned no audio")
    append_check(
        report,
        "serial requests exclude other serial and parallel single requests",
        {
            "second_serial_status": serial_rejected["status"],
            "parallel_while_serial_status": parallel_rejected["status"],
        },
    )


def test_parallel_requests(server, report, report_lock, pcm_a, pcm_b):
    barrier = threading.Barrier(3)
    jobs = {"a": "voice-a", "b": "voice-b"}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {}
        for label, model_id in jobs.items():
            def work(job_label=label, job_model=model_id):
                barrier.wait(timeout=15)
                return synthesize(
                    server,
                    report,
                    report_lock,
                    "parallel-" + job_label,
                    job_model,
                    execution="parallel",
                )[0]

            futures[label] = pool.submit(work)
        barrier.wait(timeout=15)

        deadline = time.monotonic() + 90
        overlap = None
        while time.monotonic() < deadline:
            rows = server.model_rows()
            a_live = rows.get("voice-a", {}).get("in_use", False)
            b_live = rows.get("voice-b", {}).get("in_use", False)
            if a_live and b_live:
                overlap = {"voice-a": rows["voice-a"], "voice-b": rows["voice-b"]}
                server.set_default("voice-b")
                break
            if all(future.done() for future in futures.values()):
                break
            time.sleep(0.05)
        require(
            overlap is not None,
            "parallel requests never showed both different LoRAs in_use at once",
        )

        outputs = {
            label: future.result(timeout=HTTP_TIMEOUT + 30)
            for label, future in futures.items()
        }

    require(outputs["a"] == pcm_a, "concurrent A PCM differs from its serial fixed-seed baseline")
    require(outputs["b"] == pcm_b, "concurrent B PCM differs from its serial fixed-seed baseline")
    require(server.models()["active_model"] == "voice-b",
            "changing the default during generation did not take effect for future requests")
    append_check(
        report,
        "different LoRAs overlap without changing in-flight weights",
        {
            "both_in_use_observed": True,
            "default_changed_during_overlap": "voice-b",
            "a_batch_size": 1,
            "b_batch_size": 1,
            "a_hash": hashlib.sha256(outputs["a"]).hexdigest(),
            "b_hash": hashlib.sha256(outputs["b"]).hexdigest(),
        },
    )


def decode_batch_wav(server, result, label):
    require(result.get("status") == "ok", f"batch item {label} failed: {result}")
    require(result.get("format") == "wav", f"batch item {label} is not WAV")
    encoded = result.get("audio_base64")
    require(isinstance(encoded, str) and encoded, f"batch item {label} omitted audio_base64")
    try:
        wav_bytes = base64.b64decode(encoded, validate=True)
        with wave.open(io.BytesIO(wav_bytes), "rb") as audio:
            channels = audio.getnchannels()
            sample_width = audio.getsampwidth()
            sample_rate = audio.getframerate()
            frames = audio.readframes(audio.getnframes())
            require(audio.getcomptype() == "NONE", f"batch item {label} is compressed")
    except (ValueError, wave.Error) as error:
        raise SmokeFailure(f"batch item {label} contains invalid WAV: {error}") from error
    require(channels == 1, f"batch item {label} should be mono")
    require(sample_width == 2, f"batch item {label} should be PCM16")
    require(sample_rate == result.get("sample_rate"), f"batch item {label} sample rate mismatch")
    require(frames and len(frames) % 2 == 0, f"batch item {label} has empty or invalid PCM")
    for timing in ("first_audio_ms", "wall_ms"):
        value = result.get(timing)
        require(
            isinstance(value, (int, float)) and value >= 0,
            f"batch item {label} has invalid {timing}: {value!r}",
        )
    filename = f"{label}.wav"
    (server.output_dir / filename).write_bytes(wav_bytes)
    return {
        "label": label,
        "sha256": hashlib.sha256(wav_bytes).hexdigest(),
        "sample_rate": sample_rate,
        "pcm_bytes": len(frames),
        "first_audio_ms": result["first_audio_ms"],
        "wall_ms": result["wall_ms"],
        "wav": filename,
    }


def batch_request(server, execution, items):
    require(
        all("execution" not in item for item in items),
        "execution must be set only at the batch top level",
    )
    response = server.session.post(
        server.url + "/v1/audio/speech/batch",
        json={"execution": execution, "items": items},
        timeout=HTTP_TIMEOUT,
    )
    require(
        response.ok,
        f"batch {execution} request failed: HTTP {response.status_code}: "
        f"{response.text[:1000]}",
    )
    body = response.json()
    results = body.get("results")
    require(isinstance(results, list), f"batch {execution} response omitted results")
    require(len(results) == len(items), f"batch {execution} result count mismatch")
    return results


def test_batch_api(server, report):
    items = [
        {
            "id": "batch-a-1",
            "input": SHORT_TEXT,
            "model": "voice-a",
            "cfg_scale": 1,
            "max_new_tokens": 8,
        },
        {
            "id": "batch-a-2",
            "input": SHORT_TEXT,
            "model": "voice-a",
            "cfg_scale": 1,
            "max_new_tokens": 8,
        },
        {
            "id": "batch-b-1",
            "input": SHORT_TEXT,
            "model": "voice-b",
            "cfg_scale": 1,
            "max_new_tokens": 8,
        },
        {
            "id": "batch-b-2",
            "input": SHORT_TEXT,
            "model": "voice-b",
            "cfg_scale": 1,
            "max_new_tokens": 8,
        },
    ]
    expected_ids = [item["id"] for item in items]
    details = {}
    for execution in ("serial", "parallel"):
        results = batch_request(server, execution, items)
        ids = [result.get("id") for result in results]
        require(ids == expected_ids, f"batch {execution} changed input result order: {ids}")
        for item, result in zip(items, results):
            require(
                result.get("model") == item["model"],
                f"batch {execution} returned the wrong model for {item['id']}",
            )
        details[execution] = [
            decode_batch_wav(server, result, f"batch-{execution}-{result['id']}")
            for result in results
        ]
    report["batch_audio"] = details
    append_check(
        report,
        "batch serial and parallel modes return ordered, decodable WAV results",
        {"result_ids": expected_ids, "modes": ["serial", "parallel"]},
    )


def test_failures_and_default(server, report, report_lock, output_dir, pcm_b):
    default_pcm, _ = synthesize(
        server, report, report_lock, "default-b", None, save_audio=True
    )
    require(default_pcm == pcm_b, "request without model_id did not use the selected default")
    before = server.models()["active_model"]
    require(before == "voice-b", "unexpected default before failure-path checks")

    unknown = expect_bad_speech(
        server,
        speech_form("unregistered-model"),
        "unknown model ID",
    )
    missing_path = (output_dir / "does-not-exist.gguf").resolve()
    failed_load = expect_bad_speech(
        server,
        speech_form("failed-dynamic", missing_path),
        "missing dynamic LoRA path",
    )
    state = server.models()
    rows = {row["id"]: row for row in state["data"]}
    require(state["active_model"] == before, "failed requests changed the active default")
    require("unregistered-model" not in rows, "unknown ID polluted the model registry")
    require("failed-dynamic" not in rows, "failed path polluted the model registry")
    append_check(
        report,
        "unknown ID and failed path report errors without changing registration/default",
        {"unknown_id": unknown, "failed_path": failed_load, "active_model": before},
    )
    server.set_default("base")


def test_idle_expiry(server, report, report_lock, idle_seconds):
    sample, _ = synthesize(server, report, report_lock, "idle-expiry-a", "voice-a")
    require(sample, "pre-expiry adapter request returned no audio")
    row = server.model_rows()["voice-a"]
    require(row["loaded"] and not row["in_use"], "adapter was not idle before expiry")
    wait_for_unload(server, "voice-a", idle_seconds)
    append_check(
        report,
        "idle adapter unloads without a new request; status reads do not refresh TTL",
        {"model_id": "voice-a", "idle_timeout_seconds": idle_seconds},
    )


def test_active_lease_expiry(server, report, report_lock, idle_seconds):
    token_count = min(2048, max(16, idle_seconds * 5))
    long_text = "这是一个用于检查 LoRA 正在生成时不会被空闲清理的短接口验证。" * 12
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            synthesize,
            server,
            report,
            report_lock,
            "active-over-ttl",
            "voice-a",
            None,
            token_count,
            long_text,
            False,
            HTTP_TIMEOUT,
        )
        wait_for_model(
            server,
            "voice-a",
            lambda row: row["loaded"] and row["in_use"],
            timeout=60,
            reason="long request never acquired adapter A",
        )
        target = time.monotonic() + idle_seconds + 0.25
        while time.monotonic() < target:
            if future.done():
                future.result()
                raise SmokeFailure(
                    "long generation completed before its idle timeout elapsed; "
                    "increase --idle-seconds or use a slower backend"
                )
            row = server.model_rows()["voice-a"]
            require(row["loaded"] and row["in_use"],
                    "active adapter was unloaded before the idle timeout")
            time.sleep(0.1)
    pcm, _ = future.result(timeout=HTTP_TIMEOUT + 30)
    completed_at = time.monotonic()
    require(pcm, "long active request returned no audio")
    wait_for_model(
        server,
        "voice-a",
        lambda row: not row["in_use"],
        timeout=15,
        reason="active request did not release its adapter",
    )
    wait_for_unload(server, "voice-a", idle_seconds)
    elapsed = time.monotonic() - completed_at
    require(
        elapsed + 0.75 >= idle_seconds,
        "idle expiry appears to start before the active request completed",
    )
    append_check(
        report,
        "active generation survives past TTL and expires after release",
        {
            "requested_max_new_tokens": token_count,
            "completion_to_unload_seconds": round(elapsed, 3),
        },
    )


def test_idle_websocket_reload(server, report, idle_seconds):
    baseline, _ = websocket_sample(server, "voice-a", SHORT_TEXT)
    row = server.model_rows()["voice-a"]
    require(row["loaded"] and not row["in_use"], "WebSocket baseline did not release A")

    with connect(server.ws_url, open_timeout=15, close_timeout=5) as connection:
        greeting = json.loads(connection.recv(timeout=15))
        require(greeting.get("type") == "ready", f"unexpected WebSocket greeting: {greeting}")
        connection.send(
            json.dumps(
                {
                    "type": "start",
                    "model_id": "voice-a",
                    "instruction": SHORT_INSTRUCTION,
                    "seed": 42,
                    "cfg_scale": 1,
                    "split_chars": 0,
                    "max_new_tokens": 8,
                    "execution": "serial",
                }
            )
        )
        receive_ws_started(connection)
        row = server.model_rows()["voice-a"]
        require(row["loaded"] and not row["in_use"],
                "idle WebSocket pinned the selected LoRA")
        wait_for_unload(server, "voice-a", idle_seconds)

        connection.send(json.dumps({"type": "flush", "text": SHORT_TEXT}))
        connection.send(json.dumps({"type": "end"}))
        audio, events = receive_ws_audio_until_done(connection)

    require(audio == baseline, "idle WebSocket did not resume the originally selected LoRA")
    require("done" in events, "WebSocket flush sequence did not send done")
    row = server.model_rows()["voice-a"]
    require(row["loaded"], "WebSocket flush did not reload its selected LoRA")
    append_check(
        report,
        "idle WebSocket releases cache residency and reloads its selected A on flush",
        {
            "audio_sha256": hashlib.sha256(audio).hexdigest(),
            "received_done": "done" in events,
            "adapter_loaded_after_flush": row["loaded"],
        },
    )


def test_disconnect_isolation(server, report, report_lock, idle_seconds):
    """A slow optional check: cancel one streamed model while another completes."""
    text = "這是一段用於檢查串流斷線隔離的文字。" * 8
    token_count = min(1024, max(16, idle_seconds * 5))
    first_chunk = threading.Event()
    close_request = threading.Event()
    holder = {}
    holder_lock = threading.Lock()

    def disconnect_a():
        client, response = open_speech(
            server.url,
            speech_form(
                "voice-a",
                max_new_tokens=token_count,
                text=text,
                execution="parallel",
            ),
        )
        with holder_lock:
            holder["client"] = client
            holder["response"] = response
        try:
            if not response.ok:
                raise SmokeFailure(
                    f"disconnect request failed: HTTP {response.status_code}: "
                    f"{response.text[:500]}"
                )
            chunk = next(response.iter_content(4096), b"")
            require(chunk, "disconnect request produced no first PCM chunk")
            first_chunk.set()
            if not close_request.wait(90):
                raise TimeoutError("disconnect check did not receive its close signal")
            return len(chunk)
        finally:
            response.close()
            client.close()

    pool = ThreadPoolExecutor(max_workers=2)
    try:
        future_a = pool.submit(disconnect_a)
        future_b = pool.submit(
            synthesize,
            server,
            report,
            report_lock,
            "disconnect-peer-b",
            "voice-b",
            None,
            token_count,
            text,
            False,
            HTTP_TIMEOUT,
            "parallel",
        )
        deadline = time.monotonic() + 90
        overlap = False
        while time.monotonic() < deadline:
            rows = server.model_rows()
            if (first_chunk.is_set() and rows.get("voice-a", {}).get("in_use")
                    and rows.get("voice-b", {}).get("in_use")):
                overlap = True
                break
            if future_a.done() or future_b.done():
                break
            time.sleep(0.1)
        if not overlap:
            close_request.set()
            with holder_lock:
                response = holder.get("response")
                client = holder.get("client")
            if response is not None:
                response.close()
            if client is not None:
                client.close()
            if future_a.done() and future_b.done():
                # Fast hardware can finish before both states are sampled; preserve the
                # main integration result and make this explicitly optional.
                try:
                    future_a.result(timeout=10)
                    future_b.result(timeout=10)
                except Exception as error:
                    raise SmokeFailure(f"disconnect isolation failed: {error}") from error
                report["optional_checks"].append(
                    {
                        "name": "disconnecting one streamed request leaves another model intact",
                        "status": "skipped",
                        "details": {"reason": "requests completed before simultaneous in_use was sampled"},
                    }
                )
                return
            report["optional_checks"].append(
                {
                    "name": "disconnecting one streamed request leaves another model intact",
                    "status": "skipped",
                    "details": {"reason": "overlap/first chunk was not observed within 90 seconds"},
                }
            )
            return

        close_request.set()
        first_bytes = future_a.result(timeout=30)
        require(first_bytes > 0, "cancelled request did not receive PCM before disconnect")
        peer_pcm, _ = future_b.result(timeout=HTTP_TIMEOUT + 30)
        require(peer_pcm, "peer model request failed after another request disconnected")
        append_check(
            report,
            "disconnecting one streamed request leaves another model request intact",
            {"disconnected_model": "voice-a", "completed_model": "voice-b"},
            optional=True,
        )
    finally:
        close_request.set()
        with holder_lock:
            response = holder.get("response")
            client = holder.get("client")
        if response is not None:
            response.close()
        if client is not None:
            client.close()
        pool.shutdown(wait=True, cancel_futures=True)


def touch_adapter(server, report, report_lock, model_id, adapter_path):
    return synthesize(
        server,
        report,
        report_lock,
        "lru-touch-" + model_id,
        model_id,
        lora_path=adapter_path,
        max_new_tokens=8,
        save_audio=False,
        timeout=HTTP_TIMEOUT,
    )


def test_lru_capacity(server, report, report_lock, adapter_paths, cache_size):
    candidates = list(adapter_paths.items())
    if len(candidates) < cache_size + 1:
        report["optional_checks"].append(
            {
                "name": "six-entry cache evicts the least-recently-used idle adapter",
                "status": "skipped",
                "details": {
                    "reason": f"need {cache_size + 1} unique LoRA paths, got {len(candidates)}",
                    "hint": "pass more --extra-adapter paths",
                },
            }
        )
        return

    # The cache phase uses a longer test TTL so slow local checkpoint reads do not
    # compete with the separate short-TTL reaper checks above.
    loaded = []
    for model_id, path in candidates[:cache_size]:
        touch_adapter(server, report, report_lock, model_id, path)
        loaded.append(model_id)
    rows = server.model_rows()
    require(
        all(rows[model_id]["loaded"] and not rows[model_id]["in_use"] for model_id in loaded),
        "cache did not retain all adapters before the capacity check",
    )
    state = server.models()
    require(state["lora_cache"]["loaded"] == cache_size,
            "resident cache count did not reach its configured capacity")

    # Re-touch the cache in a known order immediately before inserting item N+1.
    # The first item is therefore the least recently used; the request lease on
    # each touch is released before the next one starts.
    for model_id in loaded:
        touch_adapter(server, report, report_lock, model_id, adapter_paths[model_id])
    victim_id, victim_path = candidates[cache_size]
    touch_adapter(server, report, report_lock, victim_id, victim_path)

    state = server.models()
    rows = {row["id"]: row for row in state["data"]}
    require(state["lora_cache"]["capacity"] == cache_size,
            "server changed cache capacity during LRU test")
    require(state["lora_cache"]["loaded"] == cache_size,
            "cache exceeded or fell below capacity after loading one extra adapter")
    require(rows[victim_id]["loaded"], "newest adapter was not retained in the cache")
    require(not rows[loaded[0]]["loaded"], "LRU did not evict the oldest idle adapter")
    for model_id in loaded[1:]:
        require(rows[model_id]["loaded"], f"LRU evicted a newer adapter: {model_id}")
    append_check(
        report,
        "cache capacity is enforced and LRU evicts only the oldest idle adapter",
        {
            "capacity": cache_size,
            "loaded_before_insert": loaded,
            "inserted": victim_id,
            "evicted": loaded[0],
            "loaded_after_insert": sorted(
                model_id for model_id, row in rows.items() if row["loaded"] and model_id != "base"
            ),
        },
    )


def build_adapters(args):
    result = {
        "voice-a": resolved_file(args.adapter_a, "--adapter-a"),
        "voice-b": resolved_file(args.adapter_b, "--adapter-b"),
    }
    for index, path in enumerate(args.extra_adapter, start=1):
        result[f"extra-{index}"] = resolved_file(path, f"--extra-adapter #{index}")
    paths = [os.path.normcase(str(path)) for path in result.values()]
    require(len(paths) == len(set(paths)), "LoRA adapter paths must be unique for this test")
    return result


def run(args):
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running",
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "backend": "CPU" if args.cpu else "GPU",
        "checks": [],
        "optional_checks": [],
        "audio": [],
        "server_runs": [],
        "server_log": str(output_dir / "server.log"),
    }
    report_lock = threading.Lock()
    server = None
    failure = None
    try:
        server_path = resolved_file(args.server, "--server")
        model_path = resolved_file(args.model, "--model")
        args.server = server_path
        args.model = model_path
        adapters = build_adapters(args)
        if (output_dir / "server.log").exists():
            (output_dir / "server.log").unlink()

        server = ServerProcess(args, output_dir, adapters, report)
        server.start(args.idle_seconds, "behavior")
        check_startup_registration(server, report, adapters, args.idle_seconds, args.cache_size)
        pcm_a, pcm_b = test_serial_and_dynamic_paths(
            server, report, report_lock, adapters
        )
        test_serial_gate(server, report, report_lock)
        test_parallel_requests(server, report, report_lock, pcm_a, pcm_b)
        test_batch_api(server, report)
        test_failures_and_default(server, report, report_lock, output_dir, pcm_b)
        test_idle_expiry(server, report, report_lock, args.idle_seconds)
        test_active_lease_expiry(server, report, report_lock, args.idle_seconds)
        test_idle_websocket_reload(server, report, args.idle_seconds)
        test_disconnect_isolation(server, report, report_lock, args.idle_seconds)

        if len(adapters) >= args.cache_size + 1:
            server.stop()
            cache_ttl = max(args.idle_seconds, 900)
            server.start(cache_ttl, "cache-lru")
            check_startup_registration(server, report, adapters, cache_ttl, args.cache_size)
        test_lru_capacity(server, report, report_lock, adapters, args.cache_size)
        report["status"] = "passed"
    except Exception as error:
        failure = error
        report["status"] = "failed"
        report["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    finally:
        if server is not None:
            server.close()
        report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        result_path = output_dir / "result.json"
        result_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failure is not None:
        print(f"Smoke test failed; details saved to {output_dir / 'result.json'}", file=sys.stderr)
        return 1
    return 0


def main():
    return run(parse_args())


if __name__ == "__main__":
    sys.exit(main())
