# LoRA 请求选择与缓存

LoRA 是训练得到的一小组增量权重。本地服务共享一份底模，每个请求可以使用底模、角色 LoRA，或角色与效果 LoRA 的叠加。
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
      --lora-cache-size 3 --lora-idle-seconds 900

底模先加载，启动参数只登记 LoRA 路径，收到请求时才加载。启动注册路径可以是相对路径。
底模必须与训练这些 LoRA 时所用的模型匹配，仅有相同文件后缀或相同大小并不足够。

`GET /v1/models` 返回 `active_model`（未显式选择模型时的默认 ID）、注册模型和缓存状态：

```json
{
  "object": "list",
  "active_model": "base",
  "lora_cache": {"capacity": 3, "idle_timeout_seconds": 900, "loaded": 0, "temporary_loaded": 0, "weighted_compositions": 0},
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
加载失败都会明确报错，不会回退到默认模型或裸底模；缓存满不会阻止加载新的适配器。

`POST /v1/models/active` 仍只允许本机调用。它只设置后续未显式选模型请求的默认 ID，
不会改变正在运行的 HTTP 请求或已开始的 WebSocket 会话：

    curl -X POST http://127.0.0.1:8080/v1/models/active \
      -H "Content-Type: application/json" \
      -d "{\"model_id\":\"voice-a\"}"

该接口也能用 `{"model_id":"voice-d","lora_path":"E:\\models\\voice-d.gguf"}` 注册本机 LoRA 并设为默认；
注册只在加载成功后生效。返回 `changed` 表示默认 ID 是否改变。

`--lora-cache-size N` 控制请求之间最多保留的 LoRA 数量（默认 3），不是正在使用数量的硬上限。
`--lora-idle-seconds N` 设置普通缓存的空闲卸载时间（默认 900 秒）。
缓存后台定时清理空闲 LoRA，未被请求使用的默认 LoRA 也会卸载；下次请求会自动重新加载。
HTTP 租约保持到整个流式响应结束，WebSocket 每段生成单独持有租约。加载前先淘汰最久未使用的空闲项，
不会卸载正在使用或本次请求需要的权重。空闲项不足时允许临时超出缓存上限，不因缓存满而等待。
超额加载的权重标为临时项，最后一个使用它的请求结束后立即卸载，即便其他普通缓存已经空闲也不会转为常驻。
`loaded` 是实际已加载总数，`temporary_loaded` 是其中的临时项数量；忙时前者可以超过 `capacity`，
临时请求结束后恢复到缓存上限内。加载失败会报错，未使用的临时项立即清理，已腾出的空闲项下次使用时重新加载。

## 请求权重与效果叠加

HTTP、WebSocket `start` 和批量请求的每一项都支持以下两个字段：

- `lora_weight`：所选角色 LoRA 的权重，未填为 `1.0`。
- `extra_loras`：额外叠加的 LoRA 数组，每项填写服务端绝对路径 `lora_path`，可选 `weight`，未填为 `1.0`。

```json
{
  "model_id": "voice-a",
  "lora_weight": 1.0,
  "extra_loras": [
    {"lora_path": "E:/styles/asmr.gguf", "weight": 1.8}
  ]
}
```

这是 WebSocket `start` 中的模型字段示例。OpenAI JSON 将 `model_id` 换成 `model`，其他字段相同；
HTTP 原生表单将 `extra_loras` 作为 JSON 字符串传入。`/v1/models/active` 只修改默认模型 ID，不保存请求权重。

权重必须为有限非负数字；`0` 表示本次不加载、不应用该项，`1` 表示原强度，超过 `1` 表示增强。
允许的数值范围不等于合适的音质范围，应根据训练数据和试听选择。效果 LoRA 需要显式填写，不会自动叠加。
这些字段不改变参考音频；音色参考仍由 `voice_id` 或参考 WAV/文本决定。
路径只接受本机请求；同一请求不能重复填写同一个文件。单次叠加也可以超过缓存数量，超额权重按临时项处理。

后端计算底模加各项增量之和，即 `base + Σ(weight × alpha/rank × B(Ax))`。
每份文件只加载一次，调整权重不生成组合 GGUF、不复制底模。组合的计算图按完整权重列表区分，
仍在使用相同列表的请求可共享；不同列表并发时分别调度。最后一个请求用完组合后立即释放
组合图及其推理缓存，不额外保留空闲组合。普通 LoRA 文件权重按缓存上限及空闲时间保留，临时权重用完立即卸载。
同一角色换不同权重不会不断积累闲置推理缓存；再次使用组合时重新创建计算图。

例如插件可把 ASMR 效果的配置默认值设为 `1.8`；选用 ASMR 但未指定权重时，由插件明确发送 `1.8`。
这是该效果的配置值，后端所有未填权重仍统一为 `1.0`。

`--parallel N` 允许最多 N 个生成槽位；它决定显式并行请求的 GPU 上限，不会自动改变默认串行行为。
HTTP、WebSocket 和批量中的每一项共用后端公平队列，超过并发额度自动等待。排队中的串行请求
会阻止后到的并行请求插队；客户端只管提交、接收与取消，不需要自己维护队列或并发额度。
显式并行的不同 LoRA 请求可以参与同一轮调度，调度器按 LoRA 分组，不会把不同适配器的计算图或持久缓存混在一起。
`POST /v1/audio/speech/batch` 也接受 `execution`，由顶层模式控制整个批次，不接受 item 内的该字段。
`voice_id` 独立选择参考音频，不随模型 ID 自动改变。
`POST /v1/audio/convert` 也接受原生表单的 `model_id` 和 `lora_path`，并在转换期间保持对应租约。

模型、音色、情绪和 Bot 调用的完整例子见 [中文使用指南](usage-zh.md)。

## 验证

`tests/test_lora_converter.py` 使用很小的模拟权重核对映射、配置和缩放，不需要底模。
安装转换脚本的依赖及 pytest 后运行 `python -m pytest tests/test_lora_converter.py`。
