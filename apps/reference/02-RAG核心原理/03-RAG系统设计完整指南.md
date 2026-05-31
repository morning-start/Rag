# RAG System Design - Complete Guide

**来源**: GitHub - amitgambhir/rag-system-design-guide
**链接**: https://github.com/amitgambhir/rag-system-design-guide
**作者**: Amit Gambhir (企业级 RAG 系统工程师)
**类型**: 完整系统设计指南（1077 行，覆盖全栈）
**适用章节**: 所有章节

## 内容总览

### Part I — Foundations（基础篇）
- Foundation Models & LLM Pitfalls（大模型基础与局限）
- What is RAG & How It Works（RAG 原理与工作机制）
- RAG vs. Prompt Engineering vs. Fine-Tuning（三种方案对比）

### Part II — System Design（系统设计篇）
- Problem Framing（问题界定）
- When NOT to Use RAG（什么时候不该用 RAG）
- Failure Scenarios & Challenges（六大失败模式）
- Data & Ingestion Layer（数据摄入层）
- Chunking Strategy（分块策略）
- Embeddings Strategy（嵌入策略）
- Searching, Indexing & Vector Databases（搜索索引与向量数据库）
- Retrieval Design（检索设计）
- Reranking & Context Selection（重排序与上下文选择）
- Prompt & Grounding Strategy（提示词与地基策略）
- Generation Layer（生成层）
- Reducing Hallucinations Through Prompting（通过提示词减少幻觉）

### Part III — Operations & Architecture（运维架构篇）
- Evaluation Metrics（评估指标）
- Observability & Debugging（可观测性与调试）
- Scaling & Performance（扩展与性能）
- Infrastructure & Kubernetes（基础设施与 K8s）
- Security & Compliance（安全与合规）
- Enterprise RAG Architecture（企业级 RAG 架构）

### Part IV — Advanced Topics（高级主题篇）
- RAG vs. MCP vs. AI Agents（RAG vs MCP vs AI Agent）
- Advanced RAG Patterns（高级 RAG 模式）
- GraphRAG and Knowledge Graphs（图谱 RAG）
- Multi-Modal RAG（多模态 RAG）
- Guardrails and Safety（护栏与安全）
- Agentic RAG（智能体 RAG）
- RAG in Production（生产运维）

### 附录
- Quick Reference: Design Pitfalls & Best Practices（设计陷阱与最佳实践速查）
- Appendix A: The RAG Developer Stack (2026)
- Appendix B: Recommended Tools & Technologies

## 核心价值

1. **实践驱动**：作者基于多年企业级 RAG 系统设计经验撰写
2. **结构完整**：涵盖从基础到生产的完整链路
3. **避坑指南**：每个环节都包含常见的失败模式和解决方案
4. **时效性强**：2026 年发布，包含最新的技术栈推荐

## 关键数据（来自原文）

### 六大失败模式
1. **正确答案来自正确文档但生成错误** — 生成器幻觉
2. **正确答案来自错误文档** — 源数据质量差
3. **高相似度但低实际相关性** — 嵌入模型语义偏差
4. **不一致的回答** — 温度参数过高
5. **冷启动慢** — 首次查询延迟高
6. **成本随时间上升** — 语料库增长

### 分块策略推荐
- 产品文档、法律条文 → 512-1024 tokens
- FAQ、短问答 → 128-256 tokens
- 技术文档 → 按标题层级语义分块

### Embedding 模型选型
- 通用场景 → OpenAI text-embedding-3-large / BGE-M3
- 专业领域 → 领域微调模型
- 多语言 → BGE-M3 / M3E