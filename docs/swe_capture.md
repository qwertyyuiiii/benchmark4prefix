# SWE-chat 格式对话采集与转换

## 用途与边界

将自有 Agent 的模型调用日志、ChatGPT 导出或 Claude Code 本地日志，转换为参考 SWE-chat 字段和事件类型的全文 JSONL 数据。

三层数据同时保留：

- **原始层 `transcripts/*.jsonl`**：保留传入的完整请求、工具定义、响应及日志事件。
- **分析层 `conversations.jsonl`**：正文、长文档、工具参数和工具结果全文保留，不截断。
- **调用层 `calls.jsonl`**：每次实际调用仅保存开始时间 `started_at` 和一个模型名 `model`。

要求 Python 3.10 及以上。只采集已获准共享且不涉及隐私、凭据或保密信息的内容。工具不上传原始实验日志，不自动扫描账号或本机其他目录，也不自动获取本网页的完整会话。

## 快速示例

转换包内的离线格式示例（不调用模型）：

```bash
python swe_capture.py import --format messages \
  --input examples/messages.json --output-dir output/example
python swe_capture.py validate --dataset output/example
```

输出目录须不存在或为空。示例中的时间戳和模型名留空，不代表真实调用记录。真实模型调用的接入方式见下文和 `examples/instrument_agent.py`。

## 一 在 Agent 调用处采集

将 `swe_capture.py` 放入实验项目，在现有的模型调用处添加几行：

```python
from swe_capture import Recorder

recorder = Recorder(
    "logs/math-session-001.jsonl",
    session_id="math-session-001",
    agent="your-agent",
)

# payload 必须是这一调用实际提交的全部输入；不能只放最新用户消息。
payload = {
    "model": MODEL_NAME,
    "messages": complete_messages,
    "tools": complete_tool_definitions,
    "stream": False,
}

response = recorder.call(
    "openai-chat",
    payload,
    lambda: client.chat.completions.create(**payload),
)
```

Anthropic Messages 对应写法：

```python
response = recorder.call(
    "anthropic", payload,
    lambda: client.messages.create(**payload),
)
```

`payload` 中保留实际的 `system`、`messages`、`tools`、采样参数等。不要把 SDK client、鉴权 header 或 API key 对象交给 recorder。代码只记录你显式传入的参数，不会自动脱敏文本。

每次真正发出的调用单独记录；应用重试也单独记录。输入在调用前落盘，成功后记录完整响应，失败时保留请求和异常类型。只在调用开始时读取 UTC 时钟；该调用的输入、输出及错误记录统一使用这个开始时间，不采集结束时间，也不据此计算调用耗时。

模型名只保留一个 `model` 字段，优先取本次请求使用的名称；请求未提供时采用服务端返回的名称。两者均未知时留空，不另设请求模型、响应模型或模型来源字段。原始请求和响应仍全文保留，其中服务端原有的字段保持原样。token usage 仅使用实际返回值。

调用记录示例（以下为字段示意，名称和时间不是本对话的真实元数据）：

```json
{
  "session_id": "math-session-001",
  "call_id": "local-call-id",
  "provider": "openai-chat",
  "started_at": "2026-11-08T01:00:00Z",
  "model": "my-model",
  "status": "completed"
}
```

API 分析行的 `model` 表示本次调用使用的模型；`timestamp` 统一表示本次调用开始时间，不表示用户最初输入时间或模型输出完成时间。`capture_annotations.jsonl` 使用 `call_id` 关联调用，并将 `timestamp_source` 标为 `call_started`。`calls.jsonl` 始终逐调用保留开始时间，不受分析表去重影响。单独导入历史对话时仍保留源消息明确给出的时间，不补造调用时间。

调用工具后的结果通常进入下一次请求的历史；需要保证其中的结果未被采集端额外截断。如果应用主动压缩或裁剪历史，可补记：

```python
recorder.context_event("history_compaction", {"reason": "context_budget"})
```

紧随其后的调用仍需记录真实的新输入。没有事件时，脚本只报告“观察到历史结构变化”，不自行断言发生了压缩。

初版支持非流式 OpenAI-compatible Chat Completions 和 Anthropic Messages，暂不提供 Responses API、SSE 分片拼装或网络代理。流式接入应由应用在**完整拼装响应后**调用 `finish_call`；异步调用可按下面方式接入：

```python
cid = recorder.start_call("openai-chat", payload)
try:
    result = await async_client.chat.completions.create(**payload)
    recorder.finish_call(cid, result)
except Exception as exc:
    recorder.fail_call(cid, exc)
    raise
```

单个文件对应一个 session。POSIX 下追加写有文件锁；其他系统应每个 writer 使用独立文件。文件默认权限 0600。完整请求中的多模态内容也会原样保留。

转换采集日志：

```bash
python swe_capture.py import --format api \
  --input logs/math-session-001.jsonl --output-dir dataset/math-session-001
python swe_capture.py validate --dataset dataset/math-session-001
```


## 二 转换已有对话

### 通用 messages JSON

```json
{
  "session_id": "math-session-001",
  "agent": "your-agent",
  "messages": [
    {"role": "user", "content": "请验证这个证明", "timestamp": null},
    {"role": "assistant", "content": "我们先检查假设", "timestamp": null}
  ]
}
```

```bash
python swe_capture.py import --format messages \
  --input dialogue.json --output-dir dataset/dialogue
```

支持 OpenAI 的 `tool_calls` / `tool_call_id`，以及 Anthropic 的 `tool_use` / `tool_result` 内容块。工具结果即使位于 `user` role 的原始消息中，也不会误计为人类新提示。System/developer 等显式传入的内容映射到非 conversational 的 metadata 事件。

### ChatGPT 本人导出

对 ChatGPT 账户导出的 `conversations.json`：

```bash
python swe_capture.py import --format chatgpt \
  --input conversations.json --conversation-id YOUR_CONVERSATION_ID \
  --output-dir dataset/chatgpt-session
```

省略 `--conversation-id` 会转换文件中所有对话；建议明确选择目标。脚本沿 `current_node` 的 parent 链取当前分支，不混入被编辑、重生成的其他分支；不提取隐藏推理或内部上下文条目。不会登录、调用未公开接口或抓取其他会话。导出缺少哪些字段，输出就保留哪些空值。

### Claude Code 本地原生 JSONL

```bash
python swe_capture.py import --format claude \
  --input /your/explicit/session.jsonl --output-dir dataset/claude-session
```

支持常见 `type/user/assistant/message/content` 结构及 metadata 事件；原生日志不是完整 API payload 的保证，工具 schema 或真正送入模型的上下文可能缺失。新 CLI 版本格式需要先抽样验证；不宣称兼容所有 native 版本。

## 三 输出字段

| 文件 | 内容 |
| --- | --- |
| `conversations.jsonl` | 与 SWE-chat 文档对齐的 39 个字段，如 turn_id、role、turn_type、content、model、timestamp、tool_call_id |
| `sessions.jsonl` | 会话标识、agent、模型、对话轮数、可获得的 usage、transcript 路径等核心字段 |
| `session_logs.jsonl` | session_id 到原始 transcript 的关联 |
| `calls.jsonl` | 逐调用的 session_id、call_id、provider、started_at、model、status |
| `transcripts/*.jsonl` | 采集到的原始数据，或明确标记的导入记录 |
| `capture_annotations.jsonl` | call_id、时间含义、原文长度等扩展信息；content_truncated 始终为 false |
| `capture_quality.json` | 来源、缺失字段、历史变化及 transcript SHA-256 |
| `schema_manifest.json` | 字段清单、兼容范围及来源说明 |
