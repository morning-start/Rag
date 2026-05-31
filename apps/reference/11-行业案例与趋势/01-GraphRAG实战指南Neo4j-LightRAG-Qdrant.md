# GraphRAG 实战指南：Neo4j + LightRAG + Qdrant

> 来源: https://neo4j.com/blog/genai/what-is-graphrag/ + https://blog.csdn.net/gitblog_00171/article/details/151129297 + https://qdrant.tech/documentation/advanced-tutorials/graphrag-qdrant-neo4j/
> 获取时间: 2026-05-25
> 类型: T1 官方文档 + T4 社区教程 | Graph RAG

## 核心内容摘要

Neo4j 官方 GraphRAG 博客 + LightRAG 中文实战 + Qdrant+Neo4j 混合架构教程。覆盖 GraphRAG 从概念到生产的完整路径。

## 关键数据与论点

### GraphRAG vs 向量 RAG 核心差异

| 维度 | 向量 RAG | Graph RAG |
|------|---------|----------|
| 检索原理 | 语义相似度匹配 | 实体关系遍历 + 语义搜索 |
| 回答范围 | 局限于单个/多个文本片段 | 可跨实体关系链综合归纳 |
| 适用问题 | "产品 X 的功能是什么？" | "公司整体风险趋势如何？" |
| 构建成本 | 低（Embedding + 存储） | 高（需 LLM 抽取实体关系） |
| 更新成本 | 低（增量索引） | 高（可能需重建受影响子图） |

### Neo4j GraphRAG Pipeline 三步走

```python
from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.embeddings import OpenAIEmbeddings

driver = GraphDatabase.driver("neo4j://localhost:7687", auth=("neo4j", "password"))
embedder = OpenAIEmbeddings(model="text-embedding-3-large")
retriever = VectorRetriever(driver, "index-name", embedder)
llm = OpenAILLM(model_name="gpt-4o")
graph_rag = GraphRAG(retriever=retriever, llm=llm)

result = graph_rag.search("公司的风险趋势是什么？")
print(result.answer)
print(result.context)
```

### LightRAG 快速上手（30分钟构建企业知识图谱）

LightRAG 是轻量级 GraphRAG 框架，核心流程：
1. **实体提取**：LLM 从文本中识别 Person/Organization/Product 等实体
2. **关系识别**：推断实体间的关系类型和权重
3. **图存储**：写入 Neo4j/MongoDB 等图数据库

```python
from lightrag import LightRAG
from lightrag.llm.openai import gpt_4o_mini_complete

rag = LightRAG(
    working_dir="./auto_kg",
    llm_model_func=gpt_4o_mini_complete
)
await rag.insert("CompanyA develops ProductX...")
result = await rag.query("Who developed ProductX?")
```

### Qdrant + Neo4j 混合架构优势

Qdrant 负责高效向量检索，Neo4j 负责关系推理：
- **Ingestion 阶段**: 文档同时写入 Qdrant（向量）和 Neo4j（图谱）
- **Retrieval 阶段**: 先向量检索候选集，再沿图谱扩展关系
- **Generation 阶段**: 结合两种上下文生成更完整的回答

### 知识图谱构建组件（neo4j-graphrag）

| 组件 | 功能 | 必要性 |
|------|------|--------|
| Data Loader | 从文件提取文本 | 必须 |
| Text Splitter | 分块处理 | 必须 |
| Chunk Embedder | 计算 chunk 向量 | 推荐 |
| Schema Builder | 定义节点/关系类型 | 推荐 |
| Entity & Relation Extractor | LLM 提取实体关系 | **必须** |
| Knowledge Graph Writer | 写入 Neo4j | 必须 |
| Entity Resolver | 合并重复实体 | 推荐 |
