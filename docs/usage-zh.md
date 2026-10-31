# Breeze TTS 使用指南

## 先分清三个选择

| 选择 | 控制什么 | 如何设置 |
| --- | --- | --- |
| 模型 ID | 本次请求使用底模，或叠加哪份 LoRA 权重 | 原生表单用 `model_id`，OpenAI JSON 用 `model`，WebSocket `start` 用 `model_id` |
| 音色 ID（`voice_id`） | 使用哪段参考录音及其文字 | 上传到 `POST /v1/voices` 后取得 |
| 语气指令（`instruction`） | 这一段怎样说，例如温柔、耳语、惊喜 | 合成请求或 WebSocket 的 `instruction` 消息 |
| 执行模式 | 请求串行执行，或允许与其他显式并行请求重叠 | HTTP 表单/JSON 或 WebSocket `start` 的 `execution`，默认 `serial` |

LoRA 学习说话人的特点，参考音频提供本次合成的声音样本。两者可以一起使用。
`voice_id` 是服务给参考音频缓存起的名字，不是模型文件，也不是 LoRA ID。
相同录音和准确文字会复用缓存；临时音色在服务重启后需要重新上传。
上传时填写 `name` 会保存为 `.breeze`，下次服务启动时加载。

## 中文、情绪和声音事件

`text` 写要说的话，`instruction` 写说话方式。优先为一整段对白写连贯的自然语言指令，描述前后的情绪、语速、停顿和重音变化。
可以尝试中文；英文示例与上游用法一致。效果由模型和参考音频共同决定，需要试听。

| 目标 | 示例 |
| --- | --- |
| 自然温柔 | `instruction`: `Speak warmly and gently, with a relaxed conversational rhythm.` |
| 惊喜开心 | `instruction`: `Start with surprise, then sound delighted and affectionate.` |
| 耳语 | `instruction`: `Speak in a soft, breathy whisper.` |
| 一本正经地逗人 | `instruction`: `Speak playfully, with mock seriousness and a smile in the voice.` |
| 笑声 | `text`: `(laugh) 你终于来啦，我都等你好久了。` |
| 叹气 | `text`: `(sigh) 唉，今天有点累，陪我坐一会儿吧。` |
| 清嗓或咳嗽 | `text`: `(clears throat) 我有一件事要宣布。` 或 `(cough)` |
| 停顿 | 使用逗号、句号、破折号或省略号组织句子；时长由模型决定 |
| 拉长尾音 | `text`: `好——那就这么说定啦。`，配合 `Gently draw out the final word.` 试听 |

上游还列出了中文 `[笑]`、`[叹气]`。声音事件是模型学到的描述，不是固定命令表。
更细的 `(nervous chuckle)` 等描述可以尝试，不能保证每次都按预期出现。
`[开心]` 等普通情绪标签也不能当作已保证支持的控制指令。

普通对白先使用 `cfg_scale=1`。笑声、叹气等事件可以尝试 2，再按需要试到 3；
上游说明较高 CFG 更容易触发事件，也可能使声音更生硬。不要为每句普通对白都调高。
CFG 是整次请求或整个 WebSocket 会话的设置，`instruction` 消息只改变语气。

标点和拉长写法不能精确指定毫秒或音高。需要固定长度的静音时，应由调用方在音频之间插入静音。
连续重复很多字、很多破折号，或把复杂舞台说明塞进台词，可能让模型重复、照读或产生噪声。

## 一段话内自然转变情绪

短段可以直接在指令中描述变化：

```json
{
  "text": "你真的回来了？我还以为……没事，能再见到你，真好。",
  "instruction": "Begin with quiet surprise, hesitate briefly, then become relieved and warm."
}
```

较长对白也可以在一条指令中说明“开头轻快，中间犹豫并放慢，最后低声收尾”，
或点明某个词要拉长、某一句要加快。HTTP 请求设置 `split_chars=0` 可整段生成。
这些是表演方向，不能保证精确到毫秒、音高或某个多音字的读音。

确实需要独立生成不同段落时，再分别发送 `instruction` 和 `flush`；已经入队的段落保留原来的语气。
分段衔接可能显得突然，整段生成通常更连贯。WebSocket 仍按其协议分句，不能把它的 `split_chars=0` 当作 HTTP 的整段模式。

## Bot 的一次完整请求

单个服务共享一份底模，每个请求可以单独选择 LoRA。请求默认串行；只有设置 `execution="parallel"` 才会与其他显式并行请求重叠。
`--parallel` 仍限制 GPU 同时可用的生成槽位；
没有显式模型字段时，服务使用当前默认模型。`POST /v1/models/active` 只修改后续省略模型字段的请求，
不会切换正在运行的请求或已开始的 WebSocket 会话。

1. 在合成请求或 WebSocket `start` 中带上 `model_id`；OpenAI JSON 使用标准 `model` 字段。需要并行时显式设置 `execution="parallel"`。
2. 按该模型的配置取得参考 WAV 和准确文字，上传到 `POST /v1/voices`，保存返回的 `id`。
3. 从 `GET /health` 的 `ws_port` 取得 WebSocket 端口，连接后等待 `ready`。
4. 发送 `start`，携带 `voice_id`、整次请求的 CFG 和随机种子，等待 `started`。
5. 按顺序发送各段 `instruction` 和 `flush`；接收二进制 PCM，同时处理文本事件。
6. 发送 `end`，一直接收到 `done`，再关闭连接。

WebSocket 的音频生成可以等待 GPU，并发送 `queued`。WebSocket 在 `start` 时固定模型 ID 和路径，
但只在每段生成时持有 LoRA 缓存租约；等待文本的空闲连接不会固定显存。
缓存满且所有 LoRA 都在使用时，新的适配器请求返回 `409`。

例如，取得真实 `voice_id` 后，按顺序发送以下消息：

```json
{"type":"start","model_id":"voice-a","execution":"serial","voice_id":"example_voice","cfg_scale":2,"seed":42}
{"type":"instruction","instruction":"Sound delighted and playful."}
{"type":"flush","text":"(laugh) 你终于来啦，我正想着要不要去找你呢。"}
{"type":"instruction","instruction":"Soften into a gentle, reassuring tone."}
{"type":"flush","text":"别着急，先坐下来，今天的事情慢慢告诉我就好。"}
{"type":"end"}
```

收到的音频是 24 kHz、单声道、16 位小端 PCM；保存为 WAV 时需要添加 WAV 文件头。
`cancel` 会停止当前生成并清空待说文本，已经收到的音频仍然有效。
Bot 取消后应完成连接清理，再提交后续段落。

## HTTP 与网页

网页的声音设计、参考音色克隆和语气控制都调用 `POST /v1/audio/speech`。
网页音色菜单选择参考音频；未指定模型的请求使用服务默认值。
原生表单响应是 PCM；OpenAI 风格 JSON 可指定 `response_format="wav"` 等格式。
OpenAI JSON 的标准 `model` 字段会选择本次请求的注册模型；它不会改变服务默认值。

HTTP 的 `split_chars=0` 表示一次生成整段；WebSocket 的 0 使用默认预算 600，仍按句处理。
`flush` 只会立即提交缓冲文本，超过分段预算时仍可能拆分。
使用 curl 发送以 `(sigh)` 等开头的表单文字时，请用 `--form-string`，避免 curl 将括号当作表单语法。

流式首包时间、网页开始播放时间和整句生成速度是三个指标。网页还会预缓冲音频；
如果生成速度低于播放速度，后续仍可能等待，流式本身不能提高整句的计算速度。

## 详细协议

- [模型与 LoRA](lora.md)
- [HTTP 参数和音频格式](server.md)
- [WebSocket 消息、事件和排队](websocket.md)
- [参考音频上传与保存](voices.md)
- [上游列出的声音事件](../README.md#vocal-events)
