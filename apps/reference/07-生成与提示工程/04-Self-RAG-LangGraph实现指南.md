# Self-RAG LangGraph实现指南

## 标题
Self-RAG: 基于LangGraph的自反思RAG系统完整实现指南

## 来源链接
- [LangChain AI: LangGraph Self-RAG Tutorial](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_self_rag/)
- [DataCamp: Self-RAG: A Guide With LangGraph Implementation](https://www.datacamp.com/tutorial/self-rag)
- [Elegant Software Solutions: Advanced RAG Patterns - Self-RAG, CRAG, and Agentic Retrieval](https://www.elegantsoftwaresolutions.com/blog/building-rag-systems-advanced-patterns)

---

## 核心要点提炼

### 1. Self-RAG 概述

**Self-RAG** 是由Carnegie Mellon University和IBM Research于2023年提出的RAG增强范式。其核心创新在于引入**自反思/自评分机制**，使系统能够在检索和生成的每个关键节点进行质量评估和动态调整。

#### 传统RAG vs Self-RAG

| 维度 | 传统RAG | Self-RAG |
|------|---------|----------|
| **流程** | 单次检索→生成 | 迭代检索→评估→生成→验证循环 |
| **质量控制** | 无 | 四决策框架实时评估 |
| **错误处理** | 无反馈机制 | 自动重试或查询改写 |
| **延迟** | 低 | 中等（增加1-3次迭代） |
| **准确性** | 依赖检索质量 | 自适应优化 |

### 2. Self-RAG 四决策框架

原始论文定义了四个关键决策点，通过特殊token控制模型行为：

#### 决策1: Retrieve（是否需要检索）

**输入**: 用户问题 x
**输出**: yes / no / continue

**判断逻辑**:
- `yes`: 问题需要外部知识支持，执行检索
- `no`: 问题可由模型参数化知识回答，跳过检索
- `continue`: 需要结合已有信息继续推理

**价值**: 避免不必要的检索开销，对简单问题直接生成答案

---

#### 决策2: ISREL（检索段落相关性）

**输入**: (问题 x, 检索段落 d)
**输出**: relevant / irrelevant

**判断标准**:
- `relevant`: 段落包含解决问题的关键信息或语义相关
- `irrelevant`: 段落与问题无关或信息量不足

**实现方式**: 二分类评分器，基于关键词匹配+语义相似度

---

#### 决策3: ISSUP（生成支持度）

**输入**: (问题 x, 段落 d, 生成结果 y)
**输出**: fully supported / partially supported / no support

**判断维度**:
- `fully supported`: 所有声明均有文档证据
- `partially supported`: 部分声明有证据，部分缺乏支持
- `no support`: 主要声明无文档依据

**核心作用**: 检测幻觉，确保事实准确性

---

#### 决策4: ISUSE（回答有用性）

**输入**: (问题 x, 生成结果 y)
**输出**: 1-5评分

**评分标准**:
- **5分**: 完美回答，直接解决问题
- **4分**: 良好回答，轻微补充即可
- **3分**: 可用但需用户进一步澄清
- **2分**: 相关性弱，仅部分有用
- **1分**: 无关或不正确

**应用场景**: 决定是否需要重新生成或终止流程

---

## LangGraph 实现架构

### 系统架构图

```
                    ┌─────────────────┐
                    │    START        │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   retrieve      │ ← 检索文档
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ grade_documents │ ← ISREL: 过滤不相关文档
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
     ┌─────────────┐  ┌──────────┐  ┌──────────────┐
     │ transform_  │  │ generate │  │ (filtered     │
     │ query       │  │          │  │  docs)        │
     └──────┬──────┘  └────┬─────┘  └──────────────┘
            │               │
            │               ▼
            │     ┌───────────────────┐
            │     │ grade_generation  │ ← ISSUP + ISUSE
            │     └────────┬──────────┘
            │              │
            │    ┌─────────┼─────────┐
            │    ▼         ▼         ▼
            │ not      useful    not useful
            │supported│         │
            │    │         │         │
            │    ▼         ▼         ▼
            └───→generate   END   transform_query
                (retry)                 │
                                        │
                                        └→retrieve (重新检索)
```

### Graph State 定义

```python
from typing import List, TypedDict
from typing_extensions import TypedDict

class GraphState(TypedDict):
    """
    表示图中流转的状态

    Attributes:
        question: 用户问题
        generation: LLM生成的答案
        documents: 检索到的文档列表
    """
    question: str
    generation: str
    documents: List[str]
```

---

## 关键组件详解

### 组件1: Retrieval Grader（检索评分器）

**功能**: 判断单个文档是否与问题相关（二分类）

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

class GradeDocuments(BaseModel):
    """用于相关性检查的二分类评分"""
    binary_score: str = Field(
        description="文档是否与问题相关，'yes'或'no'"
    )

# 配置LLM（使用结构化输出）
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# Prompt设计
system = """你是一个评分器，负责评估检索文档与用户问题的相关性。
不需要过于严格的测试。目标是过滤掉错误的检索结果。
如果文档包含与问题相关的关键词或语义含义，则评为相关。
给出二分类分数'yes'或'no'以指示文档是否相关。"""

grade_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "检索文档:\n\n {document} \n\n 用户问题: {question}")
])

retrieval_grader = grade_prompt | structured_llm_grader
```

**使用示例**:
```python
question = "agent memory"
docs = retriever.invoke(question)
doc_txt = docs[1].page_content
result = retrieval_grader.invoke({"question": question, "document": doc_txt})
# 输出: GradeDocuments(binary_score='no')
```

---

### 组件2: Generation Grader（幻觉检测器）

**功能**: 检测生成内容是否被检索文档支持（二分类）

```python
class GradeHallucinations(BaseModel):
    """用于幻觉检测的二分类评分"""
    binary_score: str = Field(
        description="答案是否基于事实，'yes'或'no'"
    )

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeHallucinations)

system = """你是一个评分器，负责评估LLM生成的内容是否基于/支持于一组检索到的事实。
给出二分类分数'yes'或'no'。
'yes'表示答案是基于/支持这组事实的。"""

hallucination_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "事实集合:\n\n {documents} \n\n LLM生成: {generation}")
])

hallucination_grader = hallucination_prompt | structured_llm_grader
```

**判断逻辑**:
- `yes`: 生成内容中的所有主张都有文档支撑 → 可信
- `no`: 存在未被支持的声明 → 可能存在幻觉，需重新生成

---

### 组件3: Answer Grader（答案相关性评分器）

**功能**: 评估生成答案是否真正解决了用户问题

```python
class GradeAnswer(BaseModel):
    """用于评估答案是否解决问题的二分类评分"""
    binary_score: str = Field(
        description="答案是否解决了问题，'yes'或'no'"
    )

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeAnswer)

system = """你是一个评分器，负责评估答案是否解决/回应了问题。
给出二分类分数'yes'或'no'。
'yes'表示答案解决了该问题。"""

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "用户问题:\n\n {question} \n\n LLM生成: {generation}")
])

answer_grader = answer_prompt | structured_llm_grader
```

---

### 组件4: Question Rewriter（问题改写器）

**功能**: 当检索失败时，优化查询以提高检索质量

```python
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

system = """你是一个问题改写器，将输入问题转换为更适合向量库检索的优化版本。
观察输入并尝试推理底层的语义意图/含义。"""

re_write_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "这是初始问题:\n\n {question} \n 制定一个改进后的问题。")
])

question_rewriter = re_write_prompt | llm | StrOutputParser()

# 使用示例
result = question_rewriter.invoke({"question": "agent memory"})
# 输出: "What is the role of memory in an agent's functioning?"
```

**触发条件**: 当所有检索文档都被判定为不相关时自动调用

---

## 完整代码骨架

### 节点函数实现

```python
def retrieve(state: GraphState) -> GraphState:
    """
    检索文档

    Args:
        state: 当前图状态

    Returns:
        更新后的状态，包含检索到的文档
    """
    print("---RETRIEVE---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def generate(state: GraphState) -> GraphState:
    """
    生成答案

    Args:
        state: 当前图状态

    Returns:
        更新后的状态，包含LLM生成结果
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # RAG生成
    generation = rag_chain.invoke({
        "context": documents,
        "question": question
    })
    return {
        "documents": documents,
        "question": question,
        "generation": generation
    }

def grade_documents(state: GraphState) -> GraphState:
    """
    判断检索文档的相关性

    Args:
        state: 当前图状态

    Returns:
        更新后的状态，仅保留相关文档
    """
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke({
            "question": question,
            "document": d.page_content
        })
        if score.binary_score == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")

    return {"documents": filtered_docs, "question": question}

def transform_query(state: GraphState) -> GraphState:
    """
    改写查询以获得更好的检索结果

    Args:
        state: 当前图状态

    Returns:
        更新后的状态，包含改写后的问题
    """
    print("---TRANSFORM QUERY---")
    question = state["question"]
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": state["documents], "question": better_question}
```

### 边函数（条件路由）

```python
def decide_to_generate(state: GraphState) -> str:
    """
    决定是生成答案还是重新检索

    Returns:
        下一个要调用的节点名称
    """
    print("---ASSESS GRADED DOCUMENTS---")
    filtered_documents = state["documents"]

    if not filtered_documents:
        print("---DECISION: ALL DOCS IRRELEVANT, TRANSFORM QUERY---")
        return "transform_query"
    else:
        print("---DECISION: GENERATE---")
        return "generate"

def grade_generation_v_documents_and_question(state: GraphState) -> str:
    """
    评估生成内容的质量（幻觉检查 + 问题匹配）

    Returns:
        'useful' / 'not supported' / 'not useful'
    """
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    # 幻觉检查
    score = hallucination_grader.invoke({
        "documents": documents,
        "generation": generation
    })

    if score.binary_score == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")

        # 问题匹配检查
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({
            "question": question,
            "generation": generation
        })

        if score.binary_score == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION NOT GROUNDED, RETRY---")
        return "not supported"
```

### 构建工作流

```python
from langgraph.graph import END, StateGraph, START

workflow = StateGraph(GraphState)

# 定义节点
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)

# 构建边
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")

# 条件边：根据文档相关性决定下一步
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)

workflow.add_edge("transform_query", "retrieve")

# 条件边：根据生成质量决定下一步
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",      # 重试生成
        "useful": END,                     # 完成
        "not useful": "transform_query",   # 改写查询
    },
)

# 编译应用
app = workflow.compile()
```

---

## DataCamp 实现步骤总结

### Step 1: Setup and Installation

```bash
%pip install -U langchain_community tiktoken langchain-openai \
                langchainhub chromadb langchain langgraph
```

配置API密钥:
```python
import os
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
```

### Step 2: Set up Knowledge Base

加载文档、切分、向量化、存储到ChromaDB:
```python
urls = ["https://example.com/doc1", ...]
docs = [WebBaseLoader(url).load() for url in urls]
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=OpenAIEmbeddings(),
)
retriever = vectorstore.as_retriever()
```

### Step 3: Helper Functions

创建可复用的辅助函数:
```python
def create_structured_llm(model, schema):
    llm = ChatOpenAI(model=model, temperature=0)
    return llm.with_structured_output(schema)

def create_grading_prompt(system_message, human_template):
    return ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", human_template),
    ])

class BinaryScoreModel(BaseModel):
    binary_score: str = Field(description="Binary score: 'yes' or 'no'")
```

### Step 4: Build Core LLM Components

- Retrieval evaluator（检索评估器）
- Hallucination grader（幻觉评分器）
- Answer grader（答案评分器）
- Question rewriter（问题改写器）
- RAG chain（RAG链）

### Step 5: Set up LangGraph Workflow

- Define graph state（定义图状态）
- Define node functions（定义节点函数）
- Define edge functions（定义边函数）
- Connect nodes and edges（连接节点和边）

### Step 6: Testing the Workflow

```python
inputs = {"question": "Explain how agent memory works?"}
for output in app.stream(inputs):
    for key, value in output.items():
        print(f"Node '{key}':")

print(value["generation"])
```

---

## 局限性与注意事项

### ⚠️ 已知局限

| 局限 | 描述 | 缓解策略 |
|------|------|----------|
| **延迟增加** | 多次迭代导致响应时间增长2-5倍 | 设置最大迭代次数（建议≤3次） |
| **Token开销大** | 每次评分都需要额外LLM调用 | 使用轻量级模型（如gpt-4o-mini）做评分 |
| **Grading Prompt调优难** | 评分质量高度依赖prompt设计 | A/B测试不同prompt变体 |
| **简单查询过度工程** | 对FAQ类问题引入不必要复杂度 | 添加Retrieve决策层过滤简单查询 |
| **检索质量依赖性强** | 如果初始检索就很差，改写也难以挽救 | 结合多种检索策略（混合检索） |

### 生产环境优化建议

1. **分层模型策略**:
   - 评分步骤: gpt-4o-mini（成本低、速度快）
   - 最终生成: gpt-4o（质量高）

2. **早停机制**:
   - 首次生成support_score > 0.9时直接返回
   - 避免不必要的迭代

3. **缓存优化**:
   - 相似查询的检索决策通常稳定
   - 实现语义缓存减少重复计算

4. **监控指标**:
   - 平均迭代次数（>3次需排查上游问题）
   - 各节点的通过率分布
   - 端到端延迟P99

---

## 与其他RAG范式的定位关系

### Self-RAG vs CRAG (Corrective RAG)

| 维度 | Self-RAG | CRAG |
|------|----------|------|
| **关注点** | 生成阶段质量控制 | 检索阶段质量评估 |
| **核心机制** | 自反思+重试 | 检索质量评估+纠正动作 |
| **典型场景** | 检测生成幻觉 | 处理知识库缺失 |
| **可组合性** | ✅ 可与CRAG组合 | ✅ 可与Self-RAG组合 |

**组合模式** (生产推荐):
```
检索 → CRAG评估(相关?) → 生成 → Self-RAG反思(有支撑?)
                                    ↓ 不支持
                               重新生成
```

### Self-RAG vs Agentic RAG

| 特征 | Self-RAG | Agentic RAG |
|------|----------|-------------|
| **自主性** | 固定决策树 | 动态规划 |
| **工具使用** | 仅检索+生成 | 多工具调用(API/搜索/数据库) |
| **复杂度** | 中等 | 高 |
| **适用场景** | 结构化QA | 开放式复杂任务 |

**演进路径**: Self-RAG → CRAG → Agentic RAG（复杂度递增）

---

## 适用场景速查表

| 场景 | 推荐度 | 理由 |
|------|--------|------|
| 混合查询类型(部分需检索) | ⭐⭐⭐⭐⭐ | Retrieve决策节省延迟 |
| 高风险领域(法律/医疗/金融) | ⭐⭐⭐⭐⭐ | 幻觉检测至关重要 |
| 知识库质量参差不齐 | ⭐⭐⭐⭐ | Relevance grading过滤噪声 |
| 面向用户的应用 | ⭐⭐⭐⭐ | 自我批判提升交付质量 |
| 所有查询都需检索 | ⭐⭐ | 决策步骤增加无用开销 |
| 严格延迟要求(<2s) | ⭐⭐ | 迭代循环可能超时 |
| 高度精选的知识库 | ⭐⭐ | 检索质量已足够好 |

---

## 参考数据来源

- Akram et al. (2023). "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection". CMU & IBM Research.
- LangChain AI (2025). "Self-RAG Tutorial with LangGraph".
- DataCamp (2025). "Self-RAG: A Guide With LangGraph Implementation".
- Elegant Software Solutions (2025). "Advanced RAG Patterns: Self-RAG, CRAG, and Agentic Retrieval".

---

*文件创建日期: 2026-05-25*
*资料分类: 07-生成与提示工程*
