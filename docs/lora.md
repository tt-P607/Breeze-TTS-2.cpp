# LoRA 请求选择与缓存

LoRA 是训练得到的一小组增量权重。本地服务共享一份底模，按每个请求选择底模或一份 LoRA。
模型选择不会切换或中断已经开始的请求；要并行执行，发起请求必须显式设置 `execution="parallel"`。

## 转换训练检查点

脚本使用 scripts/requirements.txt 中的 numpy 和 gguf。输入是 Breeze 训练脚本
保存的 adapter.safetensors，其中每层包含配对的 .lora_A 和 .lora_B。

    python scripts/convert_lora_to_gguf.py path/to/adapter.safetensors -o adapters/voice-a.gguf

alpha 只从权重同目录下的 `adapter_config.json` 的 `lora.alpha` 读取。
rank 从权重形状读取，实际缩放值为 alpha/rank。需要手动指定时使用 `--alpha`，
脚本不会搜索父目录或猜测其他配置格式。
训练保存的配置结构统一如下：

    {"schema_version": 1, "lora": {"rank": 16, "alpha": 32}}

默认输出 F16 增量权重；--dtype f32 可保留 F32。
这与底模的 Q8_0 量化是两个选择，LoRA 自身不会被转换成 Q8_0。

转换前会检查目标层、A/B 配对和权重形状，避免导出一份缺层的模型。
输出文件已存在时需要 --force 才会覆盖。写入先使用同目录临时文件，
完成后再替换目标文件。

## 注册底模和 LoRA

    breeze-server base.gguf --model-id base \
      --switchable-lora voice-a=adapters/voice-a.gguf \
      --switchable-lora voice-b=adapters/voice-b.gguf \
      --switchable-lora voice-c=adapters/voice-c.gguf \
      --lora-cache-size 6 --lora-idle-seconds 900

底模先加载，启动参数只登记 LoRA 路径，收到请求时才加载。启动注册路径可以是相对路径。
底模必须与训练这些 LoRA 时所用的模型匹配，仅有相同文件后缀或相同大小并不足够。

`GET /v1/models` 返回 `active_model`（未显式选择模型时的默认 ID）、注册模型和缓存状态：

```json
{
  "object": "list",
  "active_model": "base",
  "lora_cache": {"capacity": 6, "idle_timeout_seconds": 900, "loaded": 0},
  "data": [
    {"id": "base", "object": "model", "loaded": true, "in_use": false, "idle_seconds": 0},
    {"id": "voice-a", "object": "model", "loaded": false, "in_use": false, "idle_seconds": 0}
  ]
}
```

HTTP 原生表单使用 `model_id`，也可以直接给本机绝对路径 `lora_path`。默认 `execution=serial`；
设置 `execution=parallel` 才会与其他显式并行请求共享执行许可：

    curl http://127.0.0.1:8080/v1/audio/speech \
      --form-string "model_id=voice-a" --form-string "execution=parallel" \
      --form-string "text=Hello there."

OpenAI 风格 JSON 使用标准字段 `model`，可选 `lora_path` 和 `execution`：

    {"model":"voice-a","input":"Hello there.","voice":"narrator","execution":"parallel"}

WebSocket 在 `start` 消息中使用 `model_id`、可选的 `lora_path` 和 `execution`（默认 `serial`）。
连接开始时会固定所选路径，每段生成时按 execution 模式获取执行许可和缓存租约；
等待文本的空闲连接不会固定执行许可或 LoRA 显存。

没有显式模型字段时使用当前默认 ID。`lora_path` 必须是服务端本机绝对路径，且只接受来自 localhost 的请求；
只提供路径时该请求直接使用它，不创建注册 ID。与 `model_id` 同时提供时，成功加载后才登记或更新该 ID 的路径。
未知 ID、无效路径、
加载失败和缓存全部被占用都会明确报错，不会回退到默认模型或裸底模。

`POST /v1/models/active` 仍只允许本机调用。它只设置后续未显式选模型请求的默认 ID，
不会改变正在运行的 HTTP 请求或已开始的 WebSocket 会话：

    curl -X POST http://127.0.0.1:8080/v1/models/active \
      -H "Content-Type: application/json" \
      -d "{\"model_id\":\"voice-a\"}"

该接口也能用 `{"model_id":"voice-d","lora_path":"E:\\models\\voice-d.gguf"}` 注册本机 LoRA 并设为默认；
注册只在加载成功后生效。返回 `changed` 表示默认 ID 是否改变。

`--lora-cache-size N` 控制最多保留的 LoRA 数量（默认 6）；`--lora-idle-seconds N` 设置空闲卸载时间（默认 900 秒）。
缓存后台定时清理空闲 LoRA，未被请求使用的默认 LoRA 也会卸载；下次请求会自动重新加载。
HTTP 租约保持到整个流式响应结束，WebSocket 每段生成单独持有租约。缓存满且所有 LoRA 都在使用时，
新的适配器请求返回 `409`。加载新适配器成功后再淘汰最久未使用的空闲项，加载失败不会动已有缓存或活跃权重。

`--parallel N` 允许最多 N 个生成槽位；它决定显式并行请求的 GPU 上限，不会自动改变默认串行行为。
显式并行的不同 LoRA 请求可以参与同一轮调度，调度器按 LoRA 分组，不会把不同适配器的计算图或持久缓存混在一起。
`POST /v1/audio/speech/batch` 也接受 `execution`，由顶层模式控制整个批次，不接受 item 内的该字段。
`voice_id` 独立选择参考音频，不随模型 ID 自动改变。
`POST /v1/audio/convert` 也接受原生表单的 `model_id` 和 `lora_path`，并在转换期间保持对应租约。

模型、音色、情绪和 Bot 调用的完整例子见 [中文使用指南](usage-zh.md)。

## 验证

`tests/test_lora_converter.py` 使用很小的模拟权重核对映射、配置和缩放，不需要底模。
安装转换脚本的依赖及 pytest 后运行 `python -m pytest tests/test_lora_converter.py`。
