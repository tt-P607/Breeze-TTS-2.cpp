"""Exercise a real server with two adapters. Requires requests and websockets.

python tests/lora_smoke.py --server build/breeze-server --model base.gguf \
    --adapter-a a.gguf --adapter-b b.gguf --output-dir test-output
"""

import argparse
import hashlib
import json
import os
import socket
import subprocess
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from websockets.sync.client import connect


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("server", "model", "adapter-a", "adapter-b", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--cache-size", type=int, default=1)
    parser.add_argument("--parallel", type=int, default=1)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    http, ws = free_port(), free_port()
    while ws == http:
        ws = free_port()
    url = f"http://127.0.0.1:{http}"
    session = requests.Session()
    session.trust_env = False
    command = [
        str(args.server.resolve()),
        str(args.model.resolve()),
        "--model-id",
        "base",
        "--port",
        str(http),
        "--ws-port",
        str(ws),
        "--no-warmup",
        "--webui",
        "--voices-dir",
        str((args.output_dir / "voices").resolve()),
        "--lora-cache-size",
        str(args.cache_size),
        "--parallel",
        str(args.parallel),
    ]
    command += [
        "--switchable-lora",
        f"a={args.adapter_a.resolve()}",
        "--switchable-lora",
        f"b={args.adapter_b.resolve()}",
    ]
    if args.cpu:
        command.append("--cpu")
    env = os.environ.copy()
    env["PATH"] = str(args.server.resolve().parent) + os.pathsep + env["PATH"]
    report = {"command": command, "checks": [], "audio": []}

    def select(model, expected=200, **extra):
        response = session.post(
            url + "/v1/models/active", json={"model_id": model, **extra}, timeout=60
        )
        assert response.status_code == expected, (response.status_code, response.text)
        return response.json()

    def synth(label):
        started = time.monotonic()
        response = requests.post(
            url + "/v1/audio/speech",
            data={
                "text": "你好呀，今天过得怎么样？",
                "instruction": "自然温柔地说话。",
                "cfg_scale": "2",
                "seed": "42",
                "split_chars": "0",
                "max_new_tokens": "8",
            },
            timeout=180,
            stream=True,
        )
        response.raise_for_status()
        chunks = response.iter_content(4096)
        first = next(chunks)
        first_seconds = time.monotonic() - started
        pcm = first + b"".join(chunks)
        response.close()
        assert len(pcm) > 0 and len(pcm) % 2 == 0
        rate = int(response.headers["X-Sample-Rate"])
        with wave.open(str(args.output_dir / (label + ".wav")), "wb") as output:
            output.setparams((1, 2, rate, 0, "NONE", "not compressed"))
            output.writeframes(pcm)
        report["audio"].append(
            {
                "label": label,
                "sha256": hashlib.sha256(pcm).hexdigest(),
                "seconds": len(pcm) / (rate * 2),
                "first_audio_seconds": first_seconds,
                "wall_seconds": time.monotonic() - started,
            }
        )
        return pcm

    log_path = args.output_dir / "server.log"
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(
                        f"server exited: {proc.returncode}; see {log_path}"
                    )
                try:
                    if session.get(url + "/health", timeout=1).ok:
                        break
                except requests.RequestException:
                    pass
                time.sleep(0.25)
            else:
                raise TimeoutError("server did not become ready")
            assert session.get(url + "/").ok
            log.flush()
            assert "loaded LoRA" not in log_path.read_text(encoding="utf-8")
            report["checks"].append("adapters are registered without preloading")
            assert select("a")["changed"] is True
            assert select("a")["changed"] is False
            first_a = synth("a-first")
            with connect(f"ws://127.0.0.1:{ws}", open_timeout=10) as connection:
                assert json.loads(connection.recv(timeout=10))["type"] == "ready"
                assert select("a")["changed"] is False
                select("base", 409)
            time.sleep(0.15)
            report["checks"].append(
                "same model is a no-op; an open WebSocket prevents switching"
            )
            select(
                "missing",
                422,
                lora_path=str((args.output_dir / "missing.gguf").resolve()),
            )
            assert session.get(url + "/v1/models").json()["active_model"] == "a"
            select("b")
            other = synth("b")
            assert first_a != other, "distinct adapters produced identical samples"
            select("a")
            assert synth("a-again") == first_a, "switching changed fixed-seed output"
            report["checks"].append(
                "A/B/A changes output and reproduces A after graph/cache reset"
            )
            select("dynamic", lora_path=str(args.adapter_b.resolve()))
            assert (
                select("dynamic", lora_path=str(args.adapter_b.resolve()))["changed"]
                is False
            )
            select("a")
            if args.parallel >= 4:
                with ThreadPoolExecutor(max_workers=2) as pool:
                    futures = [pool.submit(synth, f"parallel-{i}") for i in range(2)]
                    samples = [future.result() for future in futures]
                assert all(len(sample) > 0 for sample in samples)
                assert session.get(url + "/v1/models").json()["active_model"] == "a"
                report["checks"].append(
                    "two guided requests share the active adapter at --parallel 4"
                )
            select("base")
            synth("base")
            report["checks"].append("returning to the base model still generates audio")
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
    report["backend"] = "CPU" if args.cpu else "GPU (see server.log)"
    (args.output_dir / "result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
