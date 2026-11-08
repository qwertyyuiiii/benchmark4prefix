# smolagents 接入测试

使用 Hugging Face 开源 [smolagents 1.22.0](https://huggingface.co/docs/smolagents/v1.22.0/reference/agents) 的 `ToolCallingAgent`，由模型选择工具并驱动 Agent 的完整循环。接入脚本为 [`examples/smolagents_capture.py`](../examples/smolagents_capture.py)。所有命令在仓库根目录运行。

## 安装

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-agent-test.txt
```

采集器本身仍只依赖标准库；这些依赖仅用于可选的 Agent 测试。

## 真实模型测试

在本机通过安全方式配置 `DEEPSEEK_API_KEY` 环境变量，不要将密钥填入脚本、文档或 Git。然后运行：

```bash
.venv/bin/python examples/smolagents_capture.py \
  --output-dir output/smolagents-live
python3 swe_capture.py validate --dataset output/smolagents-live/dataset
```

默认使用 [DeepSeek 官方 Chat Completions 接口](https://api-docs.deepseek.com/)的 `deepseek-flash`，可用 `--model`、`--api-base` 修改。测试固定让模型调用 `multiply(17, 23)`，再通过 `final_answer` 返回结果；最多 3 个 Agent 步骤。SDK 自动重试关闭，非流式调用，每次 SDK 尝试单独记录。运行会产生少量模型调用费用。已有输出目录会在发起调用前被拒绝。

每次运行生成 `raw.jsonl`、`dataset/` 和 `report.json`。报告检查答案、调用完成状态、工具 schema、工具结果、人类提示计数和全文保留。原始日志与生成数据位于被 Git 忽略的 `output/`。

## 离线复现

```bash
.venv/bin/python examples/smolagents_capture.py --offline \
  --output-dir output/smolagents-offline
.venv/bin/python -m unittest discover -s tests -v
```

离线模式运行相同的 Agent 和工具，但以明确标记的模拟响应替代 SDK 网络调用，不需要密钥。它验证框架集成，不表示真实模型推理或实际 API 到达轨迹；原始记录及质量报告均带有 synthetic 标记。标准库环境下，原有测试照常运行，可选 Agent 测试会跳过。

## 接入要点与范围

- Recorder 包装最终 SDK `create` 调用，保留实际消息、工具 schema、采样参数及响应；`extra_body` 展平为实际请求体字段。密钥只存在于 SDK client，未交给 Recorder。
- smolagents 将工具结果包装成 `user` 消息。适配器从框架记忆取得明确的工具信息，以消息索引和正文 SHA-256 写入 `replay_metadata`，分析层据此映射为 `tool_result`，原始请求不变。
- 框架重绘的工具调用历史映射为 metadata，避免重复计数。合并或无法唯一关联的工具观察保留为框架事件，不猜测工具 ID。
- 示例限定顺序运行、单个 Agent/模型实例、非流式、单工具观察、smolagents 1.22.0。没有测试并行、多工具批处理、多模态、流式或其他框架版本。

## 实际运行结果（2026-11-08，WSL Ubuntu-22.04）

| 项目 | 结果 |
| --- | --- |
| Agent | smolagents 1.22.0 ToolCallingAgent |
| 模型 | deepseek-flash，真实 API |
| 任务 | multiply(17, 23) → final_answer |
| 最终答案 | 391 |
| 模型调用 | 2 次，均 completed |
| 分析行 | 9 |
| 人类提示 / assistant_response | 1 / 2 |
| 工具调用 / 工具结果 | 2 / 1（调用含 final_answer） |
| 校验 | valid=true，errors=[]，warnings=[] |
| 报告检查 | 7 项全部通过 |

该结果是接入烟雾测试，不是模型能力评测或缓存性能 benchmark。未将任何真实原始会话日志提交到仓库。
