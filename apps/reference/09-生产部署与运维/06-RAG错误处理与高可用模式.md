# RAG 系统错误处理与高可用模式

> 来源综合：
> - [AI Agent Error Handling: Retries, Fallbacks, and Circuit Breakers](https://www.aimadetools.com/blog/ai-agent-error-handling) (AimadeTools, 2026-04)
> - [Error Handling Patterns for AI Agent Chains](https://mentiko.com/blog/agent-chain-error-handling-patterns) (Mentiko, 2026-03)
> - [Handling Timeouts and Retries in LLM Systems](https://dasroot.net/posts/2026/02/handling-timeouts-retries-llm-systems) (DasRoot, 2026-02)

## 故障分类

| 故障类型 | 示例 | 频次 | 处理策略 |
|---------|------|------|---------|
| 速率限制 (429) | API 请求过多 | 常见 | 指数退避重试 |
| 服务错误 (500/503) | 供应商宕机 | 偶尔 | 回退到备用模型 |
| 超时 | 复杂推理超时 | 偶尔 | 增加超时或简化请求 |
| 无效输出 | 返回非法 JSON | 常见 | 更严格 Prompt + 重试 |
| 幻觉 | 调用不存在的工具 | 常见 | 执行前验证 |
| 无限循环 | 重复调用同一工具 | 罕见但危险 | 断路器 |
| 上下文溢出 | 超过 Token 限制 | 逐渐累积 | 摘要压缩 |
| 预算超限 | Token 消费达上限 | 计划内 | 优雅降级 |

## 重试策略

### 指数退避 + 抖动
基础延迟按指数增长，加入随机抖动避免惊群效应。

### 模型回退链
主模型不可用时按优先级链回退：Claude → GPT-4o → Gemini 2.0

## 断路器模式

当错误率达到阈值时断开电路，快速失败而非继续重试浪费资源。支持半开状态自动恢复检测。

## 优雅降级

根据检索质量动态调整响应策略：
- 高质量检索 → 完整回答
- 低质量检索 → 谨慎回答 + 存疑标注
- 检索失败 → 明确告知无法回答

## RTO/RPO 规划建议

- RTO（恢复时间目标）：RAG 系统建议 ≤ 5 分钟
- RPO（恢复点目标）：索引数据建议 ≤ 1 小时
- 关键组件（向量 DB、LLM 网关）必须主备部署
