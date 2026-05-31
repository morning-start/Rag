# Production RAG Systems: Complete Guide (2025)

**来源**: GitHub - ivesh/ai-interview
**链接**: https://github.com/ivesh/ai-interview/blob/main/production-rag-systems-guide.md
**类型**: 生产级 RAG 系统完整参考（2571 行）
**适用章节**: Ch3, Ch5, Ch9, Ch10, Ch11

## 内容总览

### 1. 引言与 RAG 基础
- RAG 解决的问题：知识截止、幻觉、微调成本高、私有数据
- RAG 演进史：Naive RAG (2020) → Advanced RAG (2022) → Modular RAG (2024) → Agentic RAG (2025)
- 2025 年现状：混合检索是标配、重排序是生产必备、Graph RAG 提升 35%

### 2. RAG 架构基础
- **索引阶段（离线）**：文档 → 分块 → 嵌入 → 向量库
- **检索阶段（在线）**：查询 → 嵌入 → 相似度搜索
- **生成阶段**：上下文 + 查询 → Prompt → LLM → 回答

### 3. 生产架构模式
- 基础 RAG 架构
- 高级 RAG（查询转换 + 混合检索 + 重排序）
- 模块化 RAG（多路检索器）
- Agentic RAG（自适应检索）

### 4. 评估指标
- **Context Precision**：检索块是否包含答案
- **Context Recall**：检索器是否找到所有必要信息
- **Faithfulness**：答案是否基于上下文
- **Answer Relevance**：是否回答了问题
- **RAGAS 完整评估框架**

### 5. 框架对比（2025-2026）

| 框架 | 优势 | 劣势 | 最佳场景 |
|------|------|------|----------|
| LangChain | 生态最大、多工具集成 | 抽象层多、调试复杂 | 通用 RAG |
| LlamaIndex | 索引能力最强、数据理解好 | 社区相对较小 | 复杂数据管道 |
| Haystack | 生产级搜索优化 | 灵活性较差 | 企业搜索 |
| LangGraph | Agent 构建最强 | 学习曲线陡峭 | Agentic RAG |

### 6. 生产最佳实践清单
- 向量化前清洗数据
- 混合检索（BM25 + 向量）
- 始终使用重排序
- 元数据过滤
- 监控检索质量
- Prompt 强制"仅基于上下文回答"
- 缓存热门查询
- 设置温度 0 或接近 0

### 7. Embedding 模型推荐

| 模型 | 参数量 | 维度 | MTEB | 延迟 | 推荐场景 |
|------|--------|------|------|------|----------|
| BGE-base-en-v1.5 | 110M | 768 | 67 | 20ms | 生产默认 |
| E5-base-v2 | 110M | 768 | 65 | 18ms | 通用 |
| GTE-large | 335M | 1024 | 67 | 45ms | 高质量 |
| all-MiniLM-L6-v2 | 23M | 384 | 58 | 10ms | 速度优先 |

### 8. 常见陷阱与解决方案
- 分块策略不当 → 语义分块 + 父子块
- 纯向量检索 → 混合检索
- 无重排序 → Cross-Encoder
- 提示词没有约束 → 强制引用
- 无评估框架 → RAGAS
- 无监控 → 全链路可观测性