# LoRA 加载与切换

LoRA 是训练得到的一小组增量权重。本地服务可以加载一个底模和多个 LoRA，
切换时只改变当前使用的 LoRA，不必重新加载整份底模。

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

## 启动一个底模和多个 LoRA

    breeze-server base.gguf --model-id base \
      --switchable-lora voice-a=adapters/voice-a.gguf \
      --switchable-lora voice-b=adapters/voice-b.gguf \
      --switchable-lora voice-c=adapters/voice-c.gguf

底模先加载，启动参数只登记 LoRA 路径，选中时才加载。底模必须与训练这些 LoRA 时所用的模型匹配，
仅有相同文件后缀或相同大小并不足够。

GET /v1/models 返回可选 ID 和 active_model。选择其中一个：

    curl -X POST http://127.0.0.1:8080/v1/models/active \
      -H "Content-Type: application/json" \
      -d "{\"model_id\":\"voice-a\"}"

返回 changed=true 表示已切换；已经在用该 ID 时返回 changed=false。
选择底模 ID（示例中的 base）即可取消当前 LoRA。

切换接口也可接受本机提供的 LoRA 绝对路径，并登记模型 ID：

    {"model_id": "voice-d", "lora_path": "E:\\models\\voice-d.gguf"}

向同一个 `/v1/models/active` 发送上述 JSON；该接口只允许本机调用。
之后可以只用 `model_id` 切换；再次选择当前 ID 和路径不会重载。
`--lora-cache-size N` 指定最多保留的 LoRA 数量（默认 1）。选过的 LoRA 会留在缓存中，
命中缓存时无需重新加载；缓存满后按最近使用顺序淘汰最久未选中的 LoRA。
为保留加载失败时的当前模型，加载新 LoRA 期间可能短暂多占一份 LoRA 显存。
缓存数量通过启动参数设置，修改后重启服务。底模只有一份，缓存数量只影响 LoRA 增量权重。
有 HTTP 合成或未关闭的 WebSocket 会话时，切换到另一模型返回 409。
调用方应先完成当前请求并关闭连接，再执行下一次切换。

`--parallel N` 允许同一活动模型同时生成多路音频。所有请求使用同一个活动 LoRA，
不支持在同一批次内混用不同 LoRA。切换时会清理引用旧 LoRA 的生成计算图；
参考音色缓存不变，下一次推理会为新的 LoRA 重建计算图。

voice_id 独立选择参考音频，不随模型 ID 自动改变。Bot 应为各模型保存自己的
参考 WAV 和准确文本，并把“选模型、取参考音色、生成全部段落”放在同一个全局队列里。

模型、音色、情绪和 Bot 调用的完整例子见 [中文使用指南](usage-zh.md)。

## 验证

`tests/test_lora_converter.py` 使用很小的模拟权重核对映射、配置和缩放，不需要底模。
安装转换脚本的依赖及 pytest 后运行 `python -m pytest tests/test_lora_converter.py`。

`tests/lora_smoke.py` 需要真实底模、两份不同的 LoRA，以及 requests 和 websockets：

    python tests/lora_smoke.py --server build/breeze-server --model base.gguf \
      --adapter-a voice-a.gguf --adapter-b voice-b.gguf --output-dir test-output

它启动独立端口，检查按需加载、同模型不重载、失败后保留当前模型、会话期间禁止切换、
A/B/A 输出复现和回到底模。`--parallel 4` 额外检查同模型并行，`--cpu` 使用 CPU。
