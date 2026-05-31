# Ragas完整评估流水线指南

> **来源**: [Ragas官方Improve RAG教程](https://docs.ragas.io/en/stable/howtos/applications/evaluate-and-improve-rag/) + [ML Journey: How to Evaluate a RAG Pipeline](https://mljourney.com/how-to-evaluate-a-rag-pipeline-metrics-tools-and-what-to-fix/)
>
> **整理日期**: 2026-05-25
>
> **适用场景**: RAG系统迭代优化、评估体系搭建、检索+生成质量诊断

---

## 核心要点

### 1. Ragas完整工作流（六步循环）

Ragas提供从搭建到改进的端到端评估流水线：

```
Set up RAG → Create eval dataset → Set metrics → Run experiment → Analyze errors → Improve → Re-run
```

**各阶段核心任务**：

| 阶段 | 关键动作 | 产出物 |
|------|---------|--------|
| **Set up RAG** | 搭建基础RAG系统（BM25/向量检索 + LLM生成） | 可运行的RAG pipeline |
| **Create eval dataset** | 准备QA对评估集（人工或合成） | question/ground_truth/contexts |
| **Set metrics** | 选择评估指标（faithfulness/relevancy等） | metrics配置 |
| **Run experiment** | 批量运行评估，收集分数和trace | CSV结果 + MLflow traces |
| **Analyze errors** | 分析失败模式，定位瓶颈（检索? 生成?） | 错误分类报告 |
| **Improve → Re-run** | 针对性优化后重新评估 | 对比baseline的改进数据 |

---

### 2. 四大核心指标详解

#### (1) Faithfulness（忠实度）—— 检测幻觉

**定义**: 答案中的每个声明是否都被检索上下文支持？

- **阈值**: < 0.8 表示存在幻觉风险
- **机制**: LLM-as-Judge将答案分解为独立声明，逐条与context比对
- **低分原因**: 模型依赖参数知识（训练时学到的）而非提供的context
- **修复方向**: 改进prompt（强制"仅基于给定文档回答"）+ 提升context recall

```python
from ragas.metrics import faithfulness
# Faithfulness分解声明逐条检查：
# 1. LLM提取答案中的所有factual claims
# 2. 对每个claim检查是否在retrieved contexts中找到支持证据
# 3. score = supported_claims / total_claims
```

#### (2) Answer Relevancy（答案相关性）—— 检测跑题

**定义**: 答案是否实际回答了用户的问题？

- **机制**: LLM从答案逆向生成候选问题，测其与原问题的语义相似度
- **低分原因**: 
  - 检索到离谱context → 模型忠实但答非所问
  - 模型忽略query直接总结context
- **修复方向**: 先查context precision（若低则修检索器），再强化system prompt的指令遵循

```python
from ragas.metrics import answer_relevancy
# Answer relevancy生成候选问题测相似度：
# 1. LLM基于answer生成N个候选question
# 2. 计算候选Q与原始Q的embedding cosine similarity
# 3. 取最高相似度作为score
```

#### (3) Context Precision（上下文精度）—— 检索噪声控制

**定义**: 检索结果中相关文档的比例（加权累积精度DCG风格）

$$Precision@K = \frac{1}{K} \sum_{k=1}^{K} v_k \cdot \frac{|relevant@k|}{k}$$

其中 $v_k \in \{0,1\}$ 表示第k个doc是否相关。

- **低分原因**: 检索返回过多无关chunk
- **修复方向**: 
  - 加入cross-encoder reranker（如ms-marco-MiniLM-L-6-v2）
  - 元数据过滤（按文档类型/日期/类别预过滤）
  - 调整top-k或similarity threshold

#### (4) Context Recall（上下文召回）—— 检索完整性

**定义**: ground truth中的相关信息是否出现在检索结果里？

- **阈值**: < 0.80 表示严重漏检
- **低分原因**: chunking策略不当、embedding模型不匹配领域、索引配置错误
- **诊断方法**: 多种chunk size（256/512/1024 tokens）对比实验找最优值
- **修复方向**: 
  - 调整chunk策略（大小/overlap）
  - 切换domain-specific embedding模型
  - 尝试hybrid search（BM25 + vector）

---

### 3. 合成数据集生成（降低标注成本）

**流程**: LLM从每个document chunk自动生成3-5个QA对

```python
# 合成数据生成prompt模板（简化版）
synthesis_prompt = """
Given the following document chunk, generate {n} diverse question-answer pairs.
Questions must be answerable SOLELY from this chunk.
Output format: JSON list of {{"question": ..., "answer": ...}}

Document:
{chunk}
"""

# 质量保障：人工抽检10-20%验证质量
# 常见问题：问题过于简单、答案包含chunk外信息、多样性不足
```

**注意事项**:
- 合成问题倾向于well-formed、unambiguous，与真实用户查询分布有gap
- 生产后尽快补充真实user queries（50-100条手动标注）
- 使用databricks/dolly-15k等公开benchmark作为补充

---

### 4. LLM-as-Judge机制详解

| 指标 | Judge方式 | LLM调用次数 | 成本级别 |
|------|----------|------------|---------|
| Faithfulness | 分解声明→逐条验证 | O(claims) × 1 | 中高 |
| Answer Relevancy | 生成候选Q→相似度计算 | 1次生成 + embedding | 中 |
| Context Precision | 二元相关判断 per doc | K次（top-k docs） | 中 |
| Context Recall | claim-level coverage check | O(claims) × 1 | 中高 |

**成本优化策略**:
- **抽样评估**: LLM指标跑100-500例随机样本
- **全量监控**: embedding similarity指标覆盖全量
- **分层频率**: 核心指标每日抽检，回归测试每deploy必跑

---

### 5. 错误诊断决策树

```
                    metric异常？
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
    context_recall    faithfulness   answer_relevancy
         low             low              low
          │               │               │
          ▼               ▼               ▼
     修检索器        先修检索           context_precision
     (chunking/      再修prompt         也低？→修检索器
      embedding)                       否→强化指令prompt
```

**具体操作清单**:

| 场景 | 首选方案 | 备选方案 |
|------|---------|---------|
| context_recall < 0.80 | 调chunk size实验 | 换domain-tuned embedding |
| context_precision < 0.75 | 加reranker | 加metadata filter |
| faithfulness < 0.85 | 提升recall + prompt约束 | 加self-check步骤 |
| answer_relevancy < 0.80 | 查precision | 强化answering instruction |

---

### 6. 完整Python代码示例

```python
import asyncio
from ragas import evaluate, Dataset
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset as HFDataset

# 1. 准备评估数据（Ragas标准格式）
eval_data = {
    "question": ["What is the refund policy?", "How do I reset password?"],
    "answer": ["30-day full refund.", "Click Forgot Password link."],
    "contexts": [
        ["Refund within 30 days for full amount."],
        ["Visit login page, click Forgot Password."],
    ],
    "ground_truth": [
        "Customers can get full refund within 30 days.",
        "Reset via Forgot Password on login page.",
    ],
}

dataset = HFDataset.from_dict(eval_data)

# 2. 运行评估
results = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ],
)

print(results)
# 输出示例:
# {'context_precision': 0.92, 'context_recall': 0.88,
#  'faithfulness': 0.95, 'answer_relevancy': 0.91}
```

---

## 关键数据与经验法则

| 维度 | 经验值 | 说明 |
|------|-------|------|
| Faithfulness安全线 | ≥ 0.85 | 低于此值幻觉风险显著上升 |
| Context Recall底线 | ≥ 0.80 | 低于需优先修检索器 |
| Context Precision底线 | ≥ 0.75 | 低于需加reranking |
| Answer Relevancy底线 | ≥ 0.80 | 结合precision判断根因 |
| 抽样规模 | 100-500例 | LLM-judge的成本平衡点 |
| 全量监控 | embedding指标 | 低成本覆盖全量queries |
| 回归阈值 | 下降 > 0.05 | 触发告警的退化幅度 |

---

## 适用场景说明

✅ **推荐使用本指南的场景**:

1. **首次搭建RAG评估体系**: 从零建立可重复的评估pipeline
2. **RAG性能调优迭代**: 系统性定位检索/生成瓶颈并改进
3. **生产监控体系建设**: 构建nightly batch eval + dashboard alerting
4. **Agentic RAG升级**: 从naive RAG迁移到agentic multi-hop检索

⚠️ **不适用场景**:

- 纯生成任务评估（无检索组件）
- 需要human preference ranking的场景（需额外RLHF流程）
- 多模态RAG（图像/表格检索需扩展指标）

---

## 参考资源

- Ragas官方文档: https://docs.ragas.io/en/stable/
- Ragas GitHub Examples: https://github.com/vibrantlabsai/ragas/tree/main/examples
- ML Journey原文: https://mljourney.com/how-to-evaluate-a-rag-pipeline-metrics-tools-and-what-to-fix/
