# ChromaDB 原生 Python SDK 使用指南

> 来源: https://www.datacamp.com/tutorial/chromadb-tutorial-step-by-step-guide
> 获取时间: 2026-05-25
> 类型: T1 官方文档 | 向量数据库教程

## 核心内容摘要

DataCamp 2026 年 3 月发布的 ChromaDB 完整教程，涵盖安装、核心概念、Embedding 选择、CRUD 操作、过滤查询和 RAG 集成。完全使用原生 Python SDK，不依赖 LangChain。

## 关键数据与论点

### 安装与初始化

```bash
pip install chromadb
```

```python
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
```

### 核心概念映射

| 传统数据库概念 | ChromaDB 概念 |
|--------------|-------------|
| Table | Collection |
| Row | Document |
| Column | Metadata / Embedding |
| Index | HNSW (内置) |

### 创建 Collection 与添加文档

```python
from chromadb.utils import embedding_functions

embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = client.create_collection(
    name="rag_documents",
    embedding_function=embedding_func,
    metadata={"hnsw:space": "cosine"}
)

collection.add(
    documents=["文本内容..."],
    metadatas=[{"source": "doc1.pdf", "page": 1}],
    ids=["doc1_page1"]
)
```

### 语义搜索

```python
results = collection.query(
    query_texts=["如何部署 RAG 系统"],
    n_results=5,
    where={"source": "doc1.pdf"}
)

results['documents'][0]
```

### 元数据过滤

```python
collection.query(
    query_texts=["query"],
    where={
        "$and": [
            {"source": {"$eq": "report.pdf"}},
            {"page": {"$gte": 10}}
        ]
    }
)
```

### 更新与删除

```python
collection.update(ids=["id1"], documents=["新内容"])
collection.delete(ids=["id1"])
collection.delete(where_document={"$contains": "deprecated"})
```

### Embedding 模型选择

| 模型 | 维度 | 大小 | 适用场景 |
|------|------|------|---------|
| all-MiniLM-L6-v2 | 384 | 80MB | 默认选择，多语言 |
| nomic-embed-text | 768 | 274MB | 高质量，8K上下文 |
| BGE-M3 | 1024 | ~600MB | 多语言+长文本 |
| Qwen3-Embedding | 1024 | 1.2GB | MTEB 最高分(70.7) |

### ChromaDB vs 竞品

| 特性 | ChromaDB | Pinecone | Weaviate | Milvus |
|------|---------|---------|---------|-------|
| 托管方式 | 本地/云端 | 仅云端 | 自托管/云端 | 自托管/云端 |
| 安装复杂度 | pip install | API Key | Docker | Docker/K8s |
| 免费额度 | 本地无限 | 有限 | 有限本地 | 自托管免费 |
| 最佳场景 | 原型→中型 | 生产规模 | 复杂查询 | 企业级海量 |
