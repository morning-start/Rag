# LangGraph 官方 Agentic RAG 教程

> 来源: https://docs.langchain.com/oss/python/langgraph/agentic-rag
> 获取时间: 2026-05-25
> 类型: T1 官方文档 | 完整 RAG 教程

## 核心内容摘要

LangChain 官方的 LangGraph Agentic RAG 完整教程，涵盖从文档预处理到图组装的全流程。这是目前最权威的 LangGraph RAG 实现参考。

## 关键数据与论点

### 核心架构模式

LangGraph RAG 的标准架构包含以下节点：

| 节点 | 功能 | 类型 |
|------|------|------|
| `generate_query_or_respond` | LLM 决定是否需要检索 | Agent 节点 |
| `retrieve` (ToolNode) | 执行向量检索 | 工具节点 |
| `grade_documents` | 评估检索结果相关性 | 条件路由 |
| `rewrite_question` | 检索不相关时重写查询 | 重试节点 |
| `generate_answer` | 基于上下文生成最终回答 | 生成节点 |

### State 定义（MessagesState）

```python
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]
```

### 图组装核心代码

```python
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

workflow = StateGraph(MessagesState)
workflow.add_node(generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([retriever_tool]))
workflow.add_node(rewrite_question)
workflow.add_node(generate_answer)

workflow.add_edge(START, "generate_query_or_respond")
workflow.add_conditional_edges("generate_query_or_respond", route_on_tool_calls, {...})
workflow.add_conditional_edges("retrieve", grade_documents)
workflow.add_edge("generate_answer", END)
workflow.add_edge("rewrite_question", "generate_query_or_respond")

graph = workflow.compile()
```

### 与传统 LangChain RAG 的关键差异

| 维度 | LangChain LCEL | LangGraph |
|------|--------------|-----------|
| 编排模型 | 线性 Chain/DAG | 有向图（支持循环） |
| 状态管理 | Memory 外部附加 | State 内置共享 |
| 条件路由 | RouterChain | conditional_edges |
| 工具调用 | AgentExecutor | ToolNode + bind_tools |
| 人机协同 | 不支持原生 | interrupt() 原生支持 |
| 持久化 | 需额外配置 | checkpointer 内置 |

### 版本信息

- 教程基于 **LangGraph 最新版**（2026年）
- 使用 `init_chat_model()` 统一模型初始化
- 使用 `MessagesState` 预定义状态（推荐）
- 使用 `@tool` 装饰器定义检索工具
