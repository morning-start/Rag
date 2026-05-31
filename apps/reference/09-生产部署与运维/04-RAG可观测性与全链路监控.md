# RAG 可观测性与全链路监控指南 (2026)

> 来源: https://ai-proxy-git-thinking-params.preview.braintrust.dev/articles/best-rag-observability-tools-2026 + https://docs.base14.io/instrument/apps/auto-instrumentation/langgraph/
> 获取时间: 2026-05-25
> 类型: T3 行业报告 + T1 官方文档 | 监控运维

## 核心内容摘要

Braintrust 2026 年 5 月发布的 RAG 可观测性工具对比 + base14.io LangGraph OpenTelemetry 集成官方指南。

## 关键数据与论点

### RAG 四大核心 Trace 类型

| Trace 类型 | 记录内容 | 失败模式 |
|-----------|---------|---------|
| **Retrieval traces** | 检索到的 chunks、scores、排序顺序 | 召回语义相近但主题错误的内容 |
| **Reranker traces** | 排名变化（初始→最终） | 重排器移除了唯一正确的 chunk |
| **Context-assembly traces** | chunks 如何拼接为 final prompt | 拼接逻辑丢失关键上下文 |
| **Generation traces** | LLM 调用输入/输出/延迟/成本 | LLM 基于缺失/错误上下文产生幻觉 |

### RAG Observability vs 其他 discipline 的区别

| Discipline | Primary Question | Limitation for Production RAG |
|-----------|------------------|-------------------------------|
| RAG Evaluation | Does system meet quality bar before deployment? | Pre-deployment evals 不反映线上实时质量 |
| Generic LLM Monitoring | How are LLM calls performing on cost/latency? | 不显示检索质量或重排是否降级了正确 chunk |
| APM Observability | Are services healthy from infra perspective? | 基础设施指标不衡量 answer quality 或 grounding |

### LangGraph OpenTelemetry 集成核心代码

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

provider = TracerProvider()
processor = BatchSpanProcessor(OTLPSpanExporter(endpoint="http://otel-collector:4318/v1/traces"))
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("rag-app")

# 在每个 LangGraph Node 中注入 span
def retrieve_node(state):
    with tracer.start_as_current_span("rag.retrieve") as span:
        span.set_attribute("query", state["question"])
        span.set_attribute("top_k", 5)
        results = collection.query(query_texts=[state["question"]], n_results=5)
        span.set_attribute("retrieved_count", len(results["documents"][0]))
        return {"context": results["documents"][0]}
```

### 关键延迟归因字段

| 字段 | 含义 |
|------|------|
| `llm.request.duration` | 端到端推理耗时（含排队、prefill、decode）|
| `retriever.query.latency` | 向量检索 P95 延迟 |
| `outputparser.parse.time` | 结构化解析耗时 |

### 主流工具对比

| 工具 | 原生摄入路径 | 框架覆盖 | 自托管 |
|------|------------|---------|--------|
| Phoenix/Langfuse | OpenTelemetry + OpenInference | 广泛 | ✅ |
| Braintrust | Vendor SDK (支持 OTEL) | 全覆盖 | ❌ |
| LangSmith | Vendor SDK | LangChain/LangGraph 优先 | ❌ |
| Galileo | OTEL + LangGraph Callback | LangGraph 原生 | ✅ |
