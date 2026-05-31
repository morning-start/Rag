# RAG 五种架构对比分析：从 Naive 到 Agentic

**来源**: ChannelTel Blog
**链接**: https://www.channel.tel/blog/rag-evolved-five-architectures-replaced-basic-pipeline
**类型**: 英文深度对比（2026年3月）
**适用章节**: Ch2

## 核心内容

### Naive RAG 的三大失败模式

1. **多跳推理失败**: "比较政策A和政策B"需要检索两个文档，但向量搜索只返回与查询最相似的一个
2. **过时或冲突来源**: 知识库有三个版本的同一文档时，Naive RAG 无法优先选择最新的
3. **不需要检索的查询**: "伦敦现在几点？"不需要文档查找，但 Naive RAG 仍然检索并注入无关上下文

### 五种架构概览

| 架构 | 解决的核心问题 | 关键机制 | 准确率提升 |
|------|---------------|---------|-----------|
| Self-RAG | 模型不质疑检索结果 | 反思 Token（[Retrieve]/[IsRel]/[IsSup]） | +20-35% |
| Corrective RAG (CRAG) | 检索结果不可靠 | 验证+修正循环 | +20-30% |
| Adaptive RAG | 不必要的检索开销 | 路由器判断是否检索 | 成本降 40%+ |
| GraphRAG | 全局性问题回答差 | 知识图谱社区摘要 | 全局查询准确率 +35% |
| Agentic RAG | 所有上述问题 | 自主 Agent 规划+执行 | 综合最优 |

### 选择建议

- FAQ/简单问答 → Naive RAG
- 需要事实准确性 → Self-RAG / CRAG
- 高并发、成本敏感 → Adaptive RAG
- 复杂关系推理 → GraphRAG
- 多步复杂任务 → Agentic RAG