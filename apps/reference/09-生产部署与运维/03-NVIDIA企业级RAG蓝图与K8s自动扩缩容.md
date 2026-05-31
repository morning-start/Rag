# NVIDIA 企业级 RAG 蓝图与 Kubernetes 自动扩缩容

> 来源: https://build.nvidia.com/nvidia/build-an-enterprise-rag-pipeline + https://developer.nvidia.com/blog/enabling-horizontal-autoscaling-of-enterprise-rag-components-on-kubernetes/
> 获取时间: 2026-05-24
> 类型: T2 官方教程 + 企业实践

## 核心内容摘要

NVIDIA 官方企业级 RAG Blueprint 的架构设计与生产部署方案，以及 K8s 环境下各组件的水平自动扩缩容最佳实践。

## 关键数据与论点

### NVIDIA RAG Blueprint 架构特性
- **Agent 生态系统支持**: MCP Server、Data Catalog、Reasoning Budget 配置
- **多模态**: VLM 图像理解/描述/图像感知回答
- **混合检索**: Dense + Sparse + GPU 加速索引查询
- **可插拔向量库**: ElasticSearch、Milvus 等
- **内置可观测性**: OpenTelemetry 集成 + RAGAS 评估脚本
- **部署方式**: Docker 或 Kubernetes，支持 NIM Operator GPU 共享

### K8s HPA 自动扩缩容（2025年12月数据）

**扩缩指标选择**:
| 服务 | 主要指标 | 说明 |
|------|---------|------|
| LLM NIM | TTFT p90 + concurrency | 延迟敏感型（客服聊天 ISL<2s） |
| Reranking NIM | GPU 利用率 | 吞吐量优先 |
| Embedding NIM | GPU 利用率 | 吞吐量优先 |

**实测效果**:
- 动态扩展至 **6 个 LLM Pod**
- 对应 reranking/embedding Pod 同步增加
- GPU 加速 Milvus + CAGRA 索引缓解检索瓶颈
- GenAI-Perf 脚本验证 HPA 行为

### 六层企业级 RAG 架构（IJRAI 2026 论文）
1. **数据摄入与编排层**: 分布式 ETL 处理百万级文档
2. **混合检索机制层**: 向量相似度 + BM25 + 知识图谱遍历
3. **AI 计算与推理层**: 高级推理引擎实现 2-3× GPU 利用率提升
4. **上下文工程层**: 分层检索策略实现 90% 召回准确率
5. **安全与治理层**: Agent 安全执行模式
6. **基础设施自动化层**: IaC 可复现部署

## 适用章节

Ch9 生产架构设计（四层架构、Docker/K8s 部署、性能优化）
