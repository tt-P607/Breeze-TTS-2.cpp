"""Real GPU checks for weighted stacks. Uses private ports and leaves production files untouched."""

import argparse
import base64
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import requests
from websockets.sync.client import connect

from lora_smoke import ServerProcess, require, save_pcm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("server", "model", "role", "style", "reference", "output-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = {"server_runs": [], "checks": []}
    settings = SimpleNamespace(server=args.server.resolve(), model=args.model.resolve(),
                               cache_size=2, parallel=4, cpu=False)
    server = ServerProcess(settings, out, {"role": args.role.resolve()}, report)
    extras = [{"lora_path": str(args.style.resolve()), "weight": 1.8}]
    session = requests.Session()
    session.trust_env = False
    try:
        server.start(900, "weighted-stacks")
        print(json.dumps({"url": server.url, "ws_url": server.ws_url}), flush=True)
        with args.reference.open("rb") as audio:
            result = session.post(server.url + "/v1/voices", data={
                "ref_text": "嗨，早上好，一天的好心情从见到你开始。"},
                files={"ref_audio": ("reference.wav", audio, "audio/wav")}, timeout=120)
        result.raise_for_status()
        voice = result.json()["id"]
        common = {"model_id": "role", "voice_id": voice, "text": "先把肩膀放松，慢慢呼吸。我就在这里陪着你。",
                  "instruction": "慵懒、轻声、柔和，气息明显，咬字清楚，语速舒缓，停顿自然。",
                  "max_new_tokens": "16", "split_chars": "0", "seed": "42", "cfg_scale": "2"}

        def speak(**options):
            data = {**common, **options}
            if isinstance(data.get("extra_loras"), list):
                data["extra_loras"] = json.dumps(data["extra_loras"], ensure_ascii=False)
            response = session.post(server.url + "/v1/audio/speech", data=data, timeout=180)
            response.raise_for_status()
            require("audio/pcm" in response.headers.get("Content-Type", ""), "native response is not PCM")
            require(len(response.content) > 0 and any(response.content), "empty or silent audio")
            return response.content

        normal = speak()
        require(normal == speak(lora_weight="1"), "omitted role weight differs from explicit 1")
        require(normal == speak(extra_loras=[{"lora_path": str(args.style.resolve()), "weight": 0}]),
                "zero-weight extra changes role audio")
        require(server.models()["lora_cache"]["loaded"] == 1, "zero-weight extra was loaded")
        report["checks"].append("role default 1 and zero-weight extra preserve exact PCM")

        mixed = speak(extra_loras=extras)
        style = speak(lora_weight="0", extra_loras=extras)
        require(mixed != normal and style != mixed, "role and style weights do not change output")
        require(mixed == speak(extra_loras=extras), "A-B-A stack switching contaminates generation")
        before = server.models()["lora_cache"]
        require(before["loaded"] == 2, "two-LoRA stack does not use two resident files")
        require(before["weighted_compositions"] == 0, "idle weighted graph was retained")
        require(mixed == speak(extra_loras=extras), "reused stack changes output")
        require(server.models()["lora_cache"] == before, "completed stack leaves extra cache entries")
        report["checks"].append("role + ASMR 1.8, role 0, A-B-A isolation and immediate idle graph release")

        with connect(server.ws_url, proxy=None, open_timeout=30) as ws:
            ws.recv(timeout=30)
            ws.send(json.dumps({"type": "start", **{k: v for k, v in common.items() if k != "text"},
                                "max_new_tokens": 16, "cfg_scale": 2, "seed": 42,
                                "extra_loras": extras}, ensure_ascii=False))
            started = json.loads(ws.recv(timeout=30))
            require(started["type"] == "started", f"WebSocket start failed: {started}")
            ws.send(json.dumps({"type": "end", "text": common["text"]}, ensure_ascii=False))
            pcm = bytearray()
            while True:
                msg = ws.recv(timeout=180)
                if isinstance(msg, bytes):
                    pcm.extend(msg)
                else:
                    event = json.loads(msg)
                    require(event["type"] != "error", str(event))
                    if event["type"] == "done":
                        break
        require(bytes(pcm) == mixed, "WebSocket stack differs from native HTTP")
        report["checks"].append("WebSocket stack matches native HTTP PCM")

        body = {"model": "role", "input": common["text"], "voice": voice, "instructions": common["instruction"],
                "response_format": "wav", "max_new_tokens": 16, "cfg_scale": 2, "seed": 42,
                "extra_loras": extras}
        response = session.post(server.url + "/v1/audio/speech", json=body, timeout=180)
        response.raise_for_status()
        require(response.content[44:] == mixed, "OpenAI JSON stack differs from native HTTP")
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(speak, lora_weight=w, extra_loras=extras, execution="parallel") for w in (1, 0)]
            parallel = [f.result() for f in futures]
        require(parallel == [mixed, style], "concurrent distinct stacks contaminate output")
        report["checks"].append("OpenAI JSON and parallel distinct stacks preserve exact PCM")

        batch = session.post(server.url + "/v1/audio/speech/batch", json={"execution": "parallel", "items": [
            {**body, "id": "role-asmr"}, {**body, "id": "asmr", "lora_weight": 0}]}, timeout=180)
        batch.raise_for_status()
        for result, expected in zip(batch.json()["results"], (mixed, style), strict=True):
            require(result["status"] == "ok", f"batch item failed: {result}")
            require(base64.b64decode(result["audio_base64"])[44:] == expected, "batch weights differ from single request")
        report["checks"].append("parallel batch uses per-item weights")

        for invalid in ({"lora_weight": "-1"}, {"lora_weight": "NaN"}, {"extra_loras": "{}"},
                        {"extra_loras": json.dumps([{"lora_path": str(args.role.resolve())}])}):
            response = session.post(server.url + "/v1/audio/speech", data={**common, **invalid}, timeout=20)
            require(response.status_code == 400, f"invalid stack accepted: {invalid}")
        missing = session.post(server.url + "/v1/audio/speech", data={**common,
            "extra_loras": json.dumps(extras + [{"lora_path": str(out / 'third.gguf')}])}, timeout=20)
        require(missing.status_code == 422, "missing overflow adapter did not report a load failure")
        require(server.models()["lora_cache"]["temporary_loaded"] == 0, "failed load retained an idle temporary adapter")
        report["checks"].append("invalid weights and duplicate paths rejected; missing overflow file reports load failure")
        started = time.monotonic()
        full = speak(extra_loras=extras, max_new_tokens="600")
        save_pcm(out / "角色1.0_ASMR1.8_原生叠加验证.wav", full, 24000)
        report["full_sample"] = {"seconds": len(full) / 48000, "wall_seconds": time.monotonic() - started,
                                 "pcm_sha256": hashlib.sha256(full).hexdigest()}
        server.stop()
        server.start(1, "weighted-idle-expiry")
        # A new process has no voice cache: the weight test here uses voice design.
        common.pop("voice_id")
        speak(extra_loras=extras)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and server.models()["lora_cache"]["loaded"]:
            time.sleep(0.25)
        cache = server.models()["lora_cache"]
        require(cache["loaded"] == cache["weighted_compositions"] == 0, "idle stack or its files were not unloaded")
        report["checks"].append("idle expiry unloads composition graphs and adapter files")
        report["passed"] = True
    finally:
        server.close()
        session.close()
        (out / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
