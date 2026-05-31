---
title: "第5A章 LangGraph 完整 RAG 实战"
editUrl: true
---

# 第5A章 LangGraph 完整 RAG 实战

本章的重点——使用 **LangGraph StateGraph** 构建完整的 RAG 管线。LangGraph 相比传统 LangChain LCEL Chain 的核心优势在于：图结构支持循环、状态内置共享、条件路由原生支持（来源: [04-LangGraph入门到实践全指南.md](reference/04-工具链与环境/04-LangGraph入门到实践全指南.md)）。

## 5A.1 定义 RAG 状态（State）

LangGraph 的核心是 **State（状态）**——一个贯穿整个图执行的共享数据结构。每个节点可以读取和更新状态。

```python
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

class RAGState(TypedDict):
    """RAG 管线的共享状态定义"""
    question: str                    # 用户原始问题
    rewritten_question: str          # 改写后的查询
    context: str                     # 检索到的上下文
    answer: str                      # 最终回答
    messages: Annotated[list, add_messages]  # 多轮对话消息（带 reducer）
    sources: list[str]               # 检索来源列表
    relevance_score: float           # 检索相关性评分
```

**关键点解析**：
- `messages` 字段使用 `Annotated[list, add_messages]` 类型标注，这是 LangGraph 的 **reducer 模式**
- `add_messages` reducer 会自动追加新消息到列表，而非覆盖整个列表
- 这使得多轮对话的状态管理变得极其简洁（来源: [03-LangGraph官方AgenticRAG教程.md](reference/04-工具链与环境/03-LangGraph官方AgenticRAG教程.md)）

## 5A.2 定义节点函数（Nodes）

节点是图中的基本处理单元，每个节点接收当前状态，处理后返回状态更新。

```python
def retrieve_node(state: RAGState) -> dict:
    """
    检索节点：执行混合检索 + 重排序

    从 state 中读取 question，执行检索后将结果写入 context 和 sources
    """
    question = state.get("rewritten_question") or state["question"]

    # 1. 混合检索（BM25 + 向量 + RRF）
    raw_results = hybrid_search(question, collection, bm25, chunks, k=20)

    # 2. 提取候选文档
    candidate_docs = [text for text, metadata, score in raw_results]

    # 3. Cross-Encoder 重排序
    reranked = rerank_documents(question, candidate_docs, top_k=5)

    # 4. 组装上下文，使用实际文档元数据作为来源标识
    docs_with_sources = []
    for doc_text, rerank_score in reranked:
        # 从原始检索结果中找到对应的元数据
        source_info = "未知来源"
        for text, metadata, rrf_score in raw_results:
            if text == doc_text and metadata:
                source = metadata.get("source", "未知来源")
                page = metadata.get("page", "")
                if page:
                    source_info = f"{source} (第{page}页)"
                else:
                    source_info = source
                break
        docs_with_sources.append((doc_text, source_info))

    context = format_context(docs_with_sources)
    sources = [source for _, source in docs_with_sources]
    relevance_score = reranked[0][1] if reranked else 0.0

    return {
        "context": context,
        "sources": sources,
        "relevance_score": relevance_score
    }


def generate_node(state: RAGState) -> dict:
    """
    生成节点：基于上下文调用 LLM 生成回答

    从 state 中读取 context 和 question，生成 answer
    """
    context = state["context"]
    question = state["question"]

    # 构建 Prompt
    prompt = build_rag_prompt(context, question)

    # 调用 LLM 生成
    answer = call_llm(prompt, temperature=0.1)

    return {"answer": answer}


def query_rewrite_node(state: RAGState) -> dict:
    """
    查询改写节点：可选的查询增强步骤
    """
    original = state["question"]
    rewritten = rewrite_query(original)
    return {"rewritten_question": rewritten}
```

## 5A.3 构建与编译图（Graph Assembly）

将节点组装成有向图，定义执行顺序和边（Edges）。

```python
from langgraph.graph import StateGraph, START, END

# 创建 StateGraph 实例
graph = StateGraph(RAGState)

# 添加节点
graph.add_node("rewrite", query_rewrite_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", generate_node)

# 添加边：定义执行顺序
graph.add_edge(START, "rewrite")      # 开始 → 查询改写
graph.add_edge("rewrite", "retrieve")  # 查询改写 → 检索
graph.add_edge("retrieve", "generate") # 检索 → 生成
graph.add_edge("generate", END)        # 生成 → 结束

# 编译图（生成可执行的应用）
app = graph.compile()
```

**图结构可视化**：

```text
START → [rewrite] → [retrieve] → [generate] → END
```

## 5A.4 执行 RAG 管线

编译后的 `app` 对象可以通过 `invoke()` 方法执行，传入初始状态即可触发完整流程。

```python
# 执行 RAG 查询
initial_state = {
    "question": "检索增强生成的核心优势是什么？",
    "messages": []  # 初始空消息列表（多轮对话时填充历史）
}

# invoke 触发完整流程：START → rewrite → retrieve → generate → END
result = app.invoke(initial_state)

# 输出结果
print("=" * 60)
print("问题:", result["question"])
print("-" * 60)
print("改写查询:", result.get("rewritten_question", "N/A"))
print("-" * 60)
print("来源文档:")
for source in result["sources"]:
    print(f"  - {source}")
print("-" * 60)
print("相关性评分:", f"{result['relevance_score']:.4f}")
print("-" * 60)
print("回答:\n", result["answer"])
print("=" * 60)
```

**输出示例**：

```text
============================================================
问题: 检索增强生成的核心优势是什么？
------------------------------------------------------------
改写查询: 检索增强生成系统的核心优势和主要特点
------------------------------------------------------------
来源文档:
  - paper/rag-intro.pdf (第3页)
  - tutorial/rag-architecture.md (第1页)
  - blog/rag-vs-finetuning.txt
------------------------------------------------------------
相关性评分: 0.8923
------------------------------------------------------------
回答:
 根据提供的资料，检索增强生成（RAG）的核心优势包括：

1. **知识时效性强**：RAG 无需重新训练模型即可获取最新知识，
   通过更新外部知识库即可实现知识的实时更新。[文档1]

2. **可解释性好**：每个回答都可以追溯到具体的源文档，
   用户可以验证信息的准确性。[文档2]

3. **领域适应快**：针对新领域只需准备领域文档，
   无需大规模微调模型。[文档3]

（来源：[文档1]、[文档2]、[文档3]）
============================================================
```

## 5A.5 多轮对话支持

LangGraph 的 `add_messages` reducer 使得多轮对话的实现变得非常自然。历史消息会自动累积在 `state["messages"]` 中，无需外部 Memory 组件。

```python
def chat_rag_node(state: RAGState) -> dict:
    """
    支持多轮对话的生成节点

    利用 state["messages"] 中的历史对话上下文，
    结合检索到的 context 生成更具连贯性的回答
    """
    context = state["context"]
    question = state["question"]
    history = state.get("messages", [])

    # 构建带历史上下文的 Prompt
    history_context = ""
    if history:
        history_context = "\n\n历史对话：\n"
        for msg in history[-6:]:  # 最近 3 轮对话（6 条消息）
            # 统一使用 dict 格式的消息，通过 "role" 字段区分
            role = "用户" if msg.get("role") == "user" else "助手"
            content = msg.get("content", "")
            history_context += f"{role}: {content}\n"

    prompt = f"""你是一个专业的知识助手。请基于以下上下文和历史对话回答问题。

{history_context}
参考文档：
{context}

问题：{question}

回答："""

    answer = call_llm(prompt)

    # 返回回答，同时添加到 messages（add_messages reducer 会自动追加）
    return {
        "answer": answer,
        "messages": [AIMessage(content=answer)]
    }


# 多轮对话示例
conversation_state = {
    "question": "那它的缺点呢？",  # 追问（依赖上一轮上下文）
    "messages": [
        HumanMessage(content="检索增强生成的核心优势是什么？"),
        AIMessage(content="RAG 的核心优势包括知识时效性强、可解释性好..."),
        HumanMessage(content="能详细说说知识时效性吗？"),
        AIMessage(content="RAG 的知识时效性体现在无需重新训练模型...")
    ]
}

result = app.invoke(conversation_state)
print(result["answer"])  # 会基于历史对话理解"它"指代的是 RAG
```

**多轮对话 vs 传统方案对比**：

| 维度 | LangChain 方案 | LangGraph 方案 |
|------|---------------|---------------|
| 状态管理 | ConversationBufferMemory 外部附加 | State 内置，`add_messages` reducer |
| 历史传递 | 手动加载/保存 memory | 自动累积在 `state["messages"]` |
| 代码复杂度 | 需要 Memory + Chain + Callbacks | 仅需 State 定义 + 节点函数 |
| 可扩展性 | 受限于线性 Chain | 图结构支持任意复杂流程 |

## 5A.6 完整可运行代码汇总

以下是完整的 RAG 管线代码，可直接复制运行（前提：已安装依赖且 Ollama 服务已启动）：

```python
"""
完整 RAG 管线：LangGraph StateGraph 实现
技术栈：Chroma + rank_bm25 + CrossEncoder + Ollama + LangGraph

数据格式说明：
- 第4章（索引管线）输出的 chunks 格式为 [{"text": str, "metadata": dict}, ...]
- metadata 包含 source（文件名）、page（页码）等字段
- 本章所有代码均使用 chunk["text"] 和 chunk["metadata"] 访问数据
"""

import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import httpx
import json
import numpy as np
from nltk.tokenize import word_tokenize
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


# ============================================================
# 0. 消息类型定义（统一使用 dict 格式）
# ============================================================

def HumanMessage(content: str) -> dict:
    """用户消息"""
    return {"role": "user", "content": content}


def AIMessage(content: str) -> dict:
    """助手消息"""
    return {"role": "assistant", "content": content}


# ============================================================
# 1. 全局初始化（实际项目中应拆分为独立模块）
# ============================================================

client = chromadb.PersistentClient(path="./chroma_db")
ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    model_name="nomic-embed-text",
    url="http://localhost:11434/api"
)
collection = client.get_or_create_collection(
    name="rag_documents",
    embedding_function=ollama_ef
)
cross_encoder = CrossEncoder('BAAI/bge-reranker-base')

# 假设这些变量已在索引阶段准备好
# chunks: list[dict]  格式 [{"text": str, "metadata": dict}, ...]
# bm25: BM25Okapi  BM25 模型
# all_documents: list[str]  所有文档文本（用于 BM25）

# ============================================================
# 2. 工具函数
# ============================================================

def call_llm(prompt: str, model: str = "qwen2.5:7b", temperature: float = 0.1) -> str:
    """调用 Ollama REST API"""
    url = "http://localhost:11434/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False,
               "options": {"temperature": temperature}}
    return httpx.post(url, json=payload, timeout=120.0).json()["response"].strip()

def format_context(docs_with_sources: list) -> str:
    """格式化上下文"""
    parts = [f"[{i+1}] 来源: {s}\n{c}" for i, (c, s) in enumerate(docs_with_sources)]
    return "\n\n".join(parts)

def build_rag_prompt(context: str, question: str) -> str:
    """构建 RAG Prompt"""
    return f"""你是一个专业的知识助手。请基于上下文回答问题。

上下文：
{context}

要求：必须基于上下文回答，引用来源文档编号，如无相关信息请明确说明。

问题：{question}
回答："""

def get_source_label(metadata: dict) -> str:
    """
    从文档元数据中提取来源标识

    Args:
        metadata: 文档元数据，包含 source、page 等字段

    Returns:
        格式化的来源标识字符串
    """
    source = metadata.get("source", "未知来源")
    page = metadata.get("page")
    if page is not None:
        return f"{source} (第{page}页)"
    return source

# ============================================================
# 3. 状态定义
# ============================================================

class RAGState(TypedDict):
    question: str
    context: str
    answer: str
    messages: Annotated[list, add_messages]
    sources: list[str]

# ============================================================
# 4. 节点函数
# ============================================================

def retrieve_node(state: RAGState) -> dict:
    """检索节点：混合检索 + 重排序"""
    question = state["question"]

    # 混合检索
    vector_results = collection.query(
        query_texts=[question],
        n_results=10,
        include=["documents", "metadatas", "distances"]
    )

    candidates = vector_results["documents"][0] or []
    candidate_metas = vector_results["metadatas"][0] or []

    # 重排序
    if candidates:
        pairs = [(question, doc) for doc in candidates]
        scores = cross_encoder.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:5]

        # 使用实际文档元数据作为来源标识
        docs_with_sources = []
        for doc, score in ranked:
            # 找到对应的元数据
            source_label = "未知来源"
            for i, c in enumerate(candidates):
                if c == doc and i < len(candidate_metas):
                    source_label = get_source_label(candidate_metas[i])
                    break
            docs_with_sources.append((doc, source_label))
    else:
        docs_with_sources = []

    return {
        "context": format_context(docs_with_sources),
        "sources": [s for _, s in docs_with_sources]
    }

def generate_node(state: RAGState) -> dict:
    """生成节点：LLM 生成回答"""
    prompt = build_rag_prompt(state["context"], state["question"])
    answer = call_llm(prompt)
    return {"answer": answer, "messages": [AIMessage(content=answer)]}

# ============================================================
# 5. 图组装与执行
# ============================================================

graph = StateGraph(RAGState)
graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", generate_node)
graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)

app = graph.compile()

# 执行
if __name__ == "__main__":
    result = app.invoke({
        "question": "什么是检索增强生成？",
        "messages": [HumanMessage(content="什么是检索增强生成？")]
    })
    print(result["answer"])
```

---

## 本章小结

| 模块 | 旧方案（LangChain） | 新方案（LangGraph + 原生库） | 核心收益 |
|------|-------------------|-------------------------|---------|
| 语义检索 | Chroma retriever | `collection.query()` | 去除抽象层，直接控制参数 |
| 关键词检索 | BM25Retriever | `rank_bm25.BM25Okapi()` | 轻量化，无 LangChain 依赖 |
| MMR 去重 | MMR Retriever wrapper | 自定义 MMR 函数 | 完全可控的相关性-多样性权衡 |
| 混合检索 | EnsembleRetriever | 自定义 RRF 融合函数 | 统一 ID 映射，完全可控的融合逻辑 |
| 查询增强 | MultiQueryRetriever / HyDE Chain | 纯 Python 函数 + `call_llm()` | 灵活组合，易于调试 |
| 子问题分解 | 无内置支持 | 自定义 `decompose_query()` | 复杂查询拆解，提升召回覆盖率 |
| 重排序 | BGE-Reranker wrapper | `CrossEncoder.predict()` | 直接调用，零封装开销 |
| 生成模块 | ChatOllama + ChatPromptTemplate | Ollama REST API + f-string | 去框架依赖，性能更优 |
| **编排层** | **LCEL Chain** | **LangGraph StateGraph** | **图结构、状态管理、条件路由** |
| 多轮对话 | ConversationBufferMemory | `state["messages"]` + `add_messages` | 内置支持，代码减半 |

**迁移核心价值**：

1. **去除过度抽象**：不再受限于 LangChain 的 Retriever/Chain/Memory 封装，直接操作底层库
2. **图结构编排**：StateGraph 支持条件路由、循环、并行执行，远超线性 Chain 的表达能力
3. **统一状态管理**：TypedDict State + Reducer 模式取代外部 Memory 组件，多轮对话实现大幅简化
4. **生产级就绪**：LangGraph 内置 checkpointer 持久化、interrupt() 人机协同等企业级特性（来源: [04-LangGraph入门到实践全指南.md](reference/04-工具链与环境/04-LangGraph入门到实践全指南.md)）

本章完成了从 LangChain LCEL 到 LangGraph StateGraph 的全栈迁移。你现在掌握了一个完全解耦、高度可控的生产级 RAG 管线实现。

从下一章开始，我们进入进阶篇——如何进一步提升检索质量和生成质量，让 RAG 系统达到生产级水平。
