# LangGraph 从入门到实践全指南

> 来源: https://blog.csdn.net/sunxuehai1/article/details/159554767
> 获取时间: 2026-05-25
> 类型: T4 社区分析 | LangGraph 实操指南

## 核心内容摘要

2026 年 5 月发布的中文 LangGraph 全指南，覆盖核心概念、快速开始、状态持久化、RAG 实战和人工介入等完整内容。基于通义千问模型演示，代码可直接运行。

## 关键数据与论点

### LangGraph 三大核心概念

1. **图结构（Graph Structure）**: 节点(Nodes)代表操作，边(Edges)定义执行顺序，支持普通边和条件边
2. **状态管理（State Management）**: 共享状态贯穿图执行，每个节点可读写状态
3. **循环能力（Cyclical Workflows）**: 区别于 LangChain LCEL 的线性链，支持循环逻辑

### 最小可用 RAG 实现

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]

def retrieval(state: MessagesState):
    user_query = state["messages"][-1]
    docs = retriever.invoke(str(user_query))
    context = "\n".join([doc.page_content for doc in docs])
    return {"messages": [HumanMessage(content=f"上下文:\n{context}")]}

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

graph_builder = StateGraph(MessagesState)
graph_builder.add_node("retrieval", retrieval)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "retrieval")
graph_builder.add_edge("retrieval", "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()
```

### 记忆管理对比

| 存储类型 | 用途 | 适用场景 |
|---------|------|---------|
| InMemoryStore | 短期快速访问 | 单会话临时数据 |
| MemorySaver | 状态持久化 | 跨会话恢复、长期存储 |
| RedisSaver/PostgresSaver | 生产级持久化 | 企业部署（推荐替换 InMemory） |

### 人工介入（Human-in-the-Loop）

```python
from langgraph.types import interrupt, Command

def ask_human(state: MessagesState):
    human_response = interrupt({"question": user_query})
    return {"messages": [AIMessage(human_response)]}
```

### 环境依赖（2026年最新）

```
pip install -U langgraph langchain langchain-community langchain-tavily pymupdf faiss-cpu
```
