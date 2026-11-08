# benchmark4prefix

用于 Prefix KV Cache 替换研究的 Agent 对话数据采集与整理。研究背景、数据需求和协作方式见 [合作提案](Proposal.md)。

## 对话采集与转换

`swe_capture.py` 将完整 API 调用日志、通用 messages JSON、ChatGPT 导出或 Claude Code 本地日志转换为 SWE-chat 风格的全文 JSONL。仅依赖 Python 3.10+ 标准库。

在 WSL/Linux 中，从仓库根目录运行离线示例（不调用模型）：

```bash
python3 swe_capture.py import --format messages \
  --input examples/messages.json --output-dir output/example
python3 swe_capture.py validate --dataset output/example
```

输出目录必须不存在或为空；再次运行时请选择新的输出目录。示例未提供真实调用时间和模型，验证报告中的相关 warning 为预期提示。

在自有 Agent 中使用 `from swe_capture import Recorder` 记录实际请求和响应；OpenAI-compatible Chat Completions、Anthropic Messages 的接入示例见 [examples/instrument_agent.py](examples/instrument_agent.py)。完整接入方法、导入命令、输出字段及兼容边界见 [采集与转换说明](docs/swe_capture.md)（命令均在仓库根目录执行）。

导出保留三层数据：`transcripts/*.jsonl` 保存原始记录，`conversations.jsonl` 保存全文分析行，`calls.jsonl` 保存逐调用的开始时间、单个模型名和状态。实际 API payload 对前缀复用研究更有价值；历史对话导入无法补齐未记录的请求上下文，也不代表可精确重放的 token/KV trace。

只采集有权共享的数据。Recorder 会全文记录显式传入的请求与响应，不自动脱敏；不要传入 API key、鉴权 header 或 SDK client。生成的数据和原始日志默认不纳入版本控制。

## 验证

```bash
python3 -m unittest discover -s tests -v
```

测试使用本地模拟响应与临时目录，无需 API key 或网络连接。
