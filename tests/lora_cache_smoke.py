"""Check bounded adapter residency and immediate weighted graph release on a running server."""

import argparse
import base64
import io
import json
from pathlib import Path
import threading
import time
import urllib.request
import wave


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--models", nargs=4, required=True)
    parser.add_argument("--style", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--overflow-adapters", type=Path, nargs=3,
                        help="three additional real adapters to test a four-adapter request")
    args = parser.parse_args()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    report = {"url": args.url, "checks": [], "samples": [], "passed": False}

    def request(path, body=None):
        payload = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(args.url.rstrip("/") + path, data=payload,
                                     headers={"Content-Type": "application/json"})
        with opener.open(req, timeout=120) as response:
            return response.read()

    def state():
        return json.loads(request("/v1/models"))

    def settled():
        deadline = time.monotonic() + 5
        while True:
            value = state()
            if value["lora_cache"]["weighted_compositions"] == 0:
                assert value["lora_cache"]["loaded"] <= 3, value
                return value
            assert time.monotonic() < deadline, "finished request retained its composition graph"
            time.sleep(0.05)

    def speak(model, *, batch=False, **options):
        started = time.monotonic()
        body = {
            "model": model, "input": "你好，今天过得怎么样？", "instructions": "自然、轻柔地说话。",
            "response_format": "wav", "max_new_tokens": 8, "cfg_scale": 1, "seed": 42,
            "split_chars": 0, **options,
        }
        if batch:
            # Batch disables wall-clock-driven streaming chunk changes, allowing exact PCM comparison.
            result = json.loads(request("/v1/audio/speech/batch", {"execution": "serial", "items": [body]}))
            assert result["results"][0]["status"] == "ok", result
            audio = base64.b64decode(result["results"][0]["audio_base64"])
        else:
            audio = request("/v1/audio/speech", body)
        with wave.open(io.BytesIO(audio), "rb") as wav:
            pcm = wav.readframes(wav.getnframes())
            assert wav.getnframes() > 0 and any(pcm), "empty or silent audio"
            # Streaming WAV uses a placeholder frame count. Measure the bytes actually received.
            sample = {"model": model, "options": options, "batch": batch,
                      "audio_seconds": len(pcm) / (wav.getsampwidth() * wav.getnchannels() * wav.getframerate()),
                      "wall_seconds": round(time.monotonic() - started, 3)}
        sample["cache"] = settled()["lora_cache"]
        report["samples"].append(sample)
        print(json.dumps(sample, ensure_ascii=True), flush=True)
        return pcm

    try:
        before = state()
        report["before"] = before
        assert before["lora_cache"]["capacity"] == 3, before
        for model in args.models:
            speak(model)
        value = settled()
        loaded = {r["id"] for r in value["data"] if r["loaded"] and r["id"] != "breeze-base"}
        assert loaded == set(args.models[1:]), loaded
        report["checks"].append("four sequential real adapters retain only the three most recent")
        role = args.models[-1]
        reference_pcm = None
        for weight in (1.7, 1.8, 1.6, 1.8):
            pcm = speak(role, batch=True, extra_loras=[{"lora_path": str(args.style.resolve()), "weight": weight}])
            if weight == 1.8:
                if reference_pcm is None:
                    reference_pcm = pcm
                else:
                    assert pcm == reference_pcm, "recreating the same weighted graph changed seeded audio"
        report["checks"].append("changing role/style weights leaves zero idle composition graphs and at most three adapters")
        report["checks"].append("repeated cold weighted composition preserves exact seeded PCM")
        if args.overflow_adapters:
            snapshots = []
            done = threading.Event()

            def observe():
                while not done.is_set():
                    snapshots.append(state()["lora_cache"])
                    done.wait(0.02)

            observer = threading.Thread(target=observe)
            observer.start()
            try:
                speak(role, batch=True, max_new_tokens=32, extra_loras=[
                    {"lora_path": str(path.resolve()), "weight": 0.1} for path in args.overflow_adapters])
            finally:
                done.set()
                observer.join()
            assert any(s["loaded"] == 4 and s["temporary_loaded"] >= 1 for s in snapshots), snapshots
            after_overflow = settled()["lora_cache"]
            assert after_overflow["temporary_loaded"] == 0, after_overflow
            report["overflow"] = {"peak_loaded": max(s["loaded"] for s in snapshots),
                                  "peak_temporary": max(s["temporary_loaded"] for s in snapshots),
                                  "after": after_overflow}
            report["checks"].append("four-adapter real GPU request exceeds cache capacity and unloads temporary weights on completion")
        report["after"] = settled()
        assert report["after"]["active_model"] == before["active_model"], "default model changed"
        report["passed"] = True
    finally:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
