"""Check actual HTTP, WebSocket and batch audio together with their request logs."""

import argparse
import base64
import json
import os
import re
import socket
import subprocess
import time
from pathlib import Path

import requests
from websockets.sync.client import connect


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("server", "model", "lora", "extra-lora", "reference", "output-dir"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--reference-text", required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "server.log"
    http_port, ws_port = free_port(), free_port()
    while http_port == ws_port:
        ws_port = free_port()
    url = f"http://127.0.0.1:{http_port}"
    command = [str(args.server), str(args.model), "--model-id", "base", "--port", str(http_port),
               "--ws-port", str(ws_port), "--parallel", "2", "--no-warmup", "--voices-dir",
               str(args.output_dir / "voices"), "--switchable-lora", f"role={args.lora}"]
    env = os.environ.copy()
    env["GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM"] = "1"
    env.pop("GGML_VK_PREFER_HOST_MEMORY", None)
    env.pop("GGML_VK_ALLOW_SYSMEM_FALLBACK", None)
    session = requests.Session()
    session.trust_env = False
    with log_path.open("wb") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"server exited; see {log_path}")
                try:
                    if session.get(url + "/health", timeout=1).ok:
                        break
                except requests.RequestException:
                    pass
                time.sleep(0.5)
            else:
                raise TimeoutError("server did not become ready")
            with args.reference.open("rb") as reference:
                response = session.post(url + "/v1/voices", files={"ref_audio": reference},
                                        data={"ref_text": args.reference_text}, timeout=120)
            response.raise_for_status()
            voice = response.json()["id"]
            common = {"model": "role", "voice": voice, "response_format": "wav", "cfg_scale": 1,
                      "seed": 42, "max_new_tokens": 24, "split_chars": 0}
            response = session.post(url + "/v1/audio/speech", json={
                **common, "input": "你好。\n晚安。", "instructions": "温柔地说，句间稍停。",
                "lora_weight": 0.8, "extra_loras": [{"lora_path": str(args.extra_lora), "weight": 0.5}],
            }, timeout=180)
            response.raise_for_status()
            assert response.content.startswith(b"RIFF") and len(response.content) > 44
            response = session.post(url + "/v1/audio/speech", data={
                "model_id": "role", "voice_id": voice, "text": "谢谢。", "instruction": "自然轻声。",
                "cfg_scale": "1", "seed": "42", "max_new_tokens": "24", "split_chars": "0",
            }, timeout=180)
            response.raise_for_status()
            assert len(response.content) > 0
            with connect(f"ws://127.0.0.1:{ws_port}/", open_timeout=10) as ws:
                assert json.loads(ws.recv(timeout=10))["type"] == "ready"
                ws.send(json.dumps({"type": "start", "model_id": "role", "voice_id": voice,
                                    "cfg_scale": 1, "seed": 42, "max_new_tokens": 24}))
                assert json.loads(ws.recv(timeout=10))["type"] == "started"
                for text, instruction in [("早上好。", "明快亲切。"), ("再见。", "平静缓慢。")]:
                    ws.send(json.dumps({"type": "instruction", "instruction": instruction}))
                    ws.send(json.dumps({"type": "flush", "text": text}))
                ws.send(json.dumps({"type": "end"}))
                pcm = 0
                while True:
                    value = ws.recv(timeout=180)
                    if isinstance(value, bytes):
                        pcm += len(value)
                    else:
                        event = json.loads(value)
                        assert event["type"] != "error", event
                        if event["type"] == "done":
                            break
                assert pcm > 0
            response = session.post(url + "/v1/audio/speech/batch", json={
                "execution": "parallel", "items": [
                    {**common, "id": "first", "input": "好的。", "instructions": "语气肯定。"},
                    {**common, "id": "second", "input": "晚安。", "instructions": "轻柔舒缓。"},
                ],
            }, timeout=180)
            response.raise_for_status()
            assert all(row["status"] == "ok" and len(base64.b64decode(row["audio_base64"])) > 44
                       for row in response.json()["results"])
            text = log_path.read_text(encoding="utf-8")
            headers = re.findall(r"############ 合成 #(\d+) \|", text)
            assert len(headers) == 6 and len(set(headers)) == 6, headers
            for expected in ["HTTP JSON", "HTTP 表单", "WebSocket", "HTTP batch/first", "HTTP batch/second",
                             "你好。", "晚安。", "温柔地说，句间稍停。", "明快亲切。", "平静缓慢。",
                             "weight=0.8", "weight=0.5", "############ 语气指令 ############",
                             "############ 推理文本 ############", "0% 预计", "100% 完成", "速度"]:
                assert expected in text, f"missing log field: {expected}"
            assert "底模路径：" not in text and "voice_id：" not in text
            assert text.count("100% 完成") == 6
            for identifier in headers:
                assert re.search(rf"\[合成 #{identifier}\] \[[#-]+\]", text)
            assert args.lora.name in text and args.extra_lora.name in text
            result = {"passed": True, "real_audio_requests": 6, "unique_request_ids": headers,
                      "coverage": ["HTTP JSON weighted stack", "HTTP form", "WS instruction changes",
                                   "parallel batch"], "server_log": str(log_path)}
            (args.output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                                        encoding="utf-8")
            print(json.dumps(result, ensure_ascii=False))
        finally:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=30)
            session.close()


if __name__ == "__main__":
    main()
