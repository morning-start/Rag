# RAG 从入门到生产实践

> 检索增强生成（Retrieval-Augmented Generation）全链路中文教程

## 概览

本教程系统讲解 RAG 技术的完整知识体系，从基础概念到生产部署，覆盖 13 个核心章节（含 3 个拓展章节）。每个章节配有可运行的代码示例、架构图解和生产最佳实践。

## 章节结构

```
认知篇
├── 第1章  RAG 概述与核心价值         — 大模型局限、RAG 原理、RAG vs 微调/提示工程
├── 第2章  RAG 技术架构演进（五代范式）— Naive → Advanced → Modular → Graph → Agentic

基础篇
├── 第3章  环境准备与工具链            — Ollama + Chroma + LangGraph + 项目模板
├── 第4章  数据处理与索引管线           — 文档解析、分块策略、向量化存储
├── 第5章  检索与生成管线              — 语义/关键词/混合检索、查询增强、重排序
├── 第5A章 LangGraph 完整 RAG 实战     — StateGraph 驱动的全链路 RAG 系统

进阶篇
├── 第6章  检索质量优化               — 分块调优、元数据过滤、上下文压缩
├── 第7章  生成质量优化               — Prompt 工程、引用溯源、Self-RAG、多轮对话
├── 第8章  评估体系                   — RAGAS 指标、评估框架对比、人工评测 SOP

生产篇
├── 第9章  生产架构设计               — 容器化部署、向量索引调优、高可用设计
├── 第9A章 错误处理与高可用           — 重试/降级/熔断、错误隔离
├── 第10章 监控与运维                 — OpenTelemetry 链路追踪、Grafana 监控
├── 第11章 安全与合规                — Prompt 注入防护、数据脱敏、OWASP LLM Top 10

前沿篇
├── 第12章 高级 RAG 范式              — GraphRAG + Agentic RAG 架构与实现
├── 第12A章 多模态 RAG 展望           — 视觉/音频/表格多模态检索生成
└── 第13章 RAG 与 Agent 生态融合      — MCP 协议、工具调用、多 Agent 协作
```

## 技术栈

| 层次 | 技术选型 |
|------|---------|
| 本地 LLM | Ollama（Qwen2.5 / Llama3） |
| 向量数据库 | ChromaDB（开发/轻量）、Milvus / Qdrant（生产） |
| 检索增强 | BM25 + 向量混合（RRF 融合）、Cross-Encoder 重排序 |
| 编排框架 | LangGraph（StateGraph + TypedDict + conditional_edges） |
| 评估工具 | RAGAS（RAG 指标）、TruLens（Tracing + Eval） |
| 监控 | OpenTelemetry + Jaeger / Grafana |
| 部署 | Docker Compose、Kubernetes、Helm |

## 快速开始

```bash
# 安装依赖
bun install

# 本地预览教程网站
bun run dev

# 构建静态站点
bun run build
```

> 前提条件：Node.js >= 18，bun >= 1.3.14

## 项目结构

```
RAG/
├── apps/
│   ├── web/            # 🌐 教程网站（Astro + Starlight）
│   ├── content/        # 📝 章节内容源（Markdown）
│   └── reference/      # 📚 参考资料（每章配套）
├── 大纲.md             # 全书大纲
├── AGENTS.md           # Agent 工作流配置
├── package.json        # 工作区根配置
├── turbo.json          # Turborepo 流水线配置
└── bun.lock            # 依赖锁定
```

## 网页预览

教程内容通过 Astro + Starlight 构建为静态网站，支持搜索、导航和移动端适配：

```bash
cd apps/web
bun run dev     # 开发模式（热更新）
bun run build   # 生产构建
bun run preview # 预览构建结果
```

> **注意**：构建前需创建 `apps/web/src/content/docs/chapters` 目录链接到 `apps/content`：
> ```powershell
> New-Item -ItemType Junction -Path "apps/web/src/content/docs/chapters" -Target "apps/content"
> ```

## 许可

MIT License
