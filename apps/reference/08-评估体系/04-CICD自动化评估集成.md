# CI/CD自动化评估集成

> **来源**: [CircleCI + Ragas Tutorial](https://circleci.com/blog/automated-rag-pipeline-evaluation-and-benchmarking-with-ragas/) + [DZone: RAG Accuracy Metrics Automated Evaluation](https://dzone.com/articles/rag-accuracy-metrics-automated-evaluation) + [ChatNexus: RAG System Testing Automation CI/CD](https://articles.chatnexus.io/knowledge-base/rag-system-testing-automation-cicd-for-conversatio/)
>
> **整理日期**: 2026-05-25
>
> **适用场景**: RAG系统CI/CD流水线搭建、自动化回归测试、持续质量保障

---

## 核心要点

### 1. CircleCI + Ragas完整流程（七步）

```
安装依赖 → 设置RAG Pipeline → 构建Eval Dataset → 运行Ragas评估 → 本地Orchestration → CircleCI配置 → 自动触发
```

#### Step 1: 安装依赖

```txt
# requirements.txt
ragas==0.2.15
faiss-cpu==1.11.0
langchain==0.3.25
together=1.5.8
datasets==3.6.0
numpy==2.2.6
pandas==2.2.3
langchain-together==0.3.0
```

**LLM Provider选择**: TogetherAI（$25免费额度，适合CI环境）

#### Step 2: API Key安全管理

```bash
# .env (本地开发用，不提交git)
TOGETHER_API_KEY=<YOUR_KEY>

# CircleCI环境变量设置:
# Project Settings → Environment Variables → TOGETHER_API_KEY
# ⚠️ 永远不要硬编码API Key到代码中
```

#### Step 3: 构建RAG Pipeline

```python
# rag_pipeline/pipeline.py
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA

class RAGPipeline:
    def __init__(self, llm, embedding_provider):
        self.llm = llm
        self.embedding = embedding_provider
        self.vectorstore = None
        self.qa_chain = None

    def build_vector_store(self, documents):
        """FAISS向量索引构建"""
        self.vectorstore = FAISS.from_documents(documents, self.embedding)

    def setup_qa_chain(self):
        """配置RetrievalQA链"""
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever()
        )

    def run_queries(self, dataset, query_sample_size=30):
        """批量执行查询，收集Ragas所需数据"""
        results = []
        for item in dataset.select(range(query_sample_size)):
            query = item["instruction"]
            ground_truth = item["response"]
            
            response = self.qa_chain.invoke({"query": query})
            answer = response["result"]
            
            retrieved_docs = self.qa_chain.retriever.invoke(query)
            contexts = [doc.page_content for doc in retrieved_docs]
            
            results.append({
                "question": query,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": ground_truth
            })
        return results
```

#### Step 4: 准备评估数据集

使用 **databricks/dolly-15k** 作为benchmark：

```python
# rag_pipeline/dataloader.py
from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DollyDataLoader:
    def __init__(self, sample_size=None):
        self.sample_size = sample_size
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=50
        )

    def load_data(self):
        dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
        # 过滤closed_qa类别（最适合RAG eval）
        dataset = dataset.filter(lambda x: x['category'] == 'closed_qa')
        
        if self.sample_size:
            dataset = dataset.shuffle(seed=42).select(
                range(min(self.sample_size, len(dataset)))
            )
        
        documents = [Document(page_content=item["context"]) for item in dataset]
        chunks = self.splitter.split_documents(documents)
        return chunks, dataset
```

**参数化设计**: `DOCUMENTS_SAMPLE_SIZE`和`QUERY_SAMPLE_SIZE`可通过CI pipeline parameters调整。

#### Step 5: 运行Ragas评估

```python
# eval_script.py
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset as HFDataset

def run_ragas_evaluation(results):
    """将RAG输出转为Ragas格式并运行评估"""
    eval_dataset = HFDataset.from_dict({
        "question": results["question"],
        "answer": results["answer"],
        "contexts": results["contexts"],
        "ground_truth": results["ground_truth"],
    })

    evaluation_result = evaluate(
        eval_dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )
    
    return evaluation_result
```

#### Step 6: 本地Orchestration

```python
# main.py - 本地测试入口
from rag_pipeline.dataloader import DollyDataLoader
from rag_pipeline.pipeline import RAGPipeline
from rag_pipeline.model_provider import LLMProvider, EmbeddingProvider

def main():
    # 配置
    DOCUMENTS_SAMPLE_SIZE = 200  # 向量库文档数
    QUERY_SAMPLE_SIZE = 30       # 评估查询数
    
    # 加载数据
    loader = DollyDataLoader(sample_size=DOCUMENTS_SAMPLE_SIZE)
    documents, hf_dataset = loader.load_data()
    
    # 构建pipeline
    llm = LLMProvider("meta-llama/Llama-3-70b-chat-hf").get_llm()
    embedding = EmbeddingProvider("togethercomputer/m2-bert-80M-8k-retrieval").get_embedding_provider()
    
    pipeline = RAGPipeline(llm, embedding)
    pipeline.build_vector_store(documents)
    pipeline.setup_qa_chain()
    
    # 执行查询+收集结果
    results = pipeline.run_queries(hf_dataset, QUERY_SAMPLE_SIZE)
    
    # Ragas评估
    scores = run_ragas_evaluation(results)
    print(scores)
    
    # 断言检查（CI gate）
    assert scores['faithfulness'] > 0.85, f"Faithfulness too low: {scores['faithfulness']}"
    assert scores['context_recall'] > 0.80, f"Context Recall too low: {scores['context_recall']}"

if __name__ == "__main__":
    main()
```

#### Step 7: CircleCI配置

```yaml
# .circleci/config.yml
version: 2.1

jobs:
  rag-evaluation:
    docker:
      - image: cimg/python:3.11
    steps:
      - checkout
      - restore_cache:
          keys:
            - v1-deps-{{ checksum "requirements.txt" }}
      - run:
          name: Install dependencies
          command: |
            python -m venv venv
            . venv/bin/activate
            pip install -r requirements.txt
      - save_cache:
          key: v1-deps-{{ checksum "requirements.txt" }}
          paths:
            - "./venv"
      - run:
          name: Run RAG evaluation
          command: |
            . venv/bin/activate
            python main.py
      - store_artifacts:
          path: experiments/
          destination: rag-eval-results

workflows:
  version: 2
  evaluate-rag:
    jobs:
      - rag-evaluation
```

---

### 2. GitHub Actions Workflow示例

```yaml
# .github/workflows/rag-evaluation.yml
name: RAG Accuracy Check

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  accuracy-check:
    runs-on: ubuntu-latest
    env:
      TOGETHER_API_KEY: ${{ secrets.TOGETHER_API_KEY }}
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          
      - name: Install dependencies
        run: pip install -r requirements.txt
        
      - name: Run Ragas evaluation
        run: python eval_pipeline.py
        
      - name: Assert quality gates
        run: |
          python -c "
          import json
          with open('eval_results.json') as f:
              data = json.load(f)
          
          assert data['context_precision'] > 0.75, \
              f'Context precision below threshold: {data[\"context_precision\"]}'
          assert data['faithfulness'] > 0.85, \
              f'Faithfulness below threshold: {data[\"faithfulness\"]}'
          assert data['critical_errors'] == 0, \
              'Critical errors detected in logs!'
          
          print('✅ All quality gates passed')
          "
          
      - name: Upload evaluation report
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: rag-eval-report
          path: experiments/
```

---

### 3. CI/CD评估检查项清单

| 检查维度 | 具体指标 | 阈值建议 | 触发动作 |
|---------|---------|---------|---------|
| **高级指标断言** | Faithfulness | > 0.85 | ❌ Fail build |
| | Answer Relevancy | > 0.80 | ⚠️ Warning |
| | Context Precision | > 0.75 | ❌ Fail build |
| | Context Recall | > 0.80 | ❌ Fail build |
| | MRR (Mean Reciprocal Rank) | > threshold | ❌ Fail build |
| **Schema合规性** | 引用标注存在性 | 必须有 | ❌ Fail build |
| | 页脚/操作按钮 | 必须有 | ❌ Fail build |
| | 响应格式JSON校验 | valid JSON | ❌ Fail build |
| **日志质量** | Critical errors | **= 0** | ❌ Fail build |
| | Error rate | < 1% | ⚠️ Warning |
| **性能SLA** | P95 latency | < 2000ms | ❌ Fail build |
| | Throughput (并发) | > X qps | ⚠️ Warning |

---

### 4. 回归防护策略

#### (1) Prompt变更触发完整Test Suite

```
每次system prompt或instruction template修改时：
  ├── 跑全量eval set（非抽样）
  ├── 对比变更前后各指标delta
  └── 若任何指标下降 > 0.05 → 阻止合并
```

#### (2) 文档库更新即时重评

```
当corpus发生更新时：
  ├── 不等待nightly batch job
  ├── 立即触发full re-index + eval
  └── 文档变化是检索质量骤降的最常见原因
```

#### (3) 嵌入模型切换验证

```
更换embedding model前：
  ├── 构建新索引（shadow mode）
  ├── 同一eval set跑新旧两版对比
  ├── 关注recall@k和precision@k的变化
  └── 验证通过后才切流量
```

---

### 5. 性能测试集成（k6/Locust）

```javascript
// k6 RAG load test script (load_test.js)
import http from 'k6/http';
import { check } from 'k6';
import { Rate } from 'k6/metrics';

const errorRate = new Rate('errors');

export const options = {
  stages: [
    { duration: '30s', target: 10 },   // ramp-up
    { duration: '1m', target: 50 },     // sustained load
    { duration: '20s', target: 0 },     // ramp-down
  ],
  thresholds: {
    http_req_duration: ['p(95)<2000'],  // P95 latency < 2s
    errors: ['rate<0.05'],              // error rate < 5%
  },
};

export default function () {
  const payload = JSON.stringify({
    question: 'What is the refund policy?',
  });
  
  const res = http.post('http://localhost:8000/chat', payload, {
    headers: { 'Content-Type': 'application/json' },
  });
  
  errorRate.add(res.status !== 200);
  
  check(res, {
    'status is 200': (r) => r.status === 200,
    'response has answer': (r) => 
      JSON.parse(r.body).answer.length > 0,
    'response has sources': (r) => 
      JSON.parse(r.body).contexts.length > 0,
  });
}
```

**集成到CI**: 在evaluation job后并行运行load test job，两者都通过才允许deploy。

---

### 6. 安全与合规扫描

```yaml
# CI中的安全扫描步骤
security-scan:
  steps:
    # 容器镜像漏洞扫描
    - name: Trivy image scan
      uses: aquasecurity/trivy-action@master
      with:
        image-ref: 'my-rag-app:latest'
        severity: 'CRITICAL,HIGH'
        exit-code: '1'  # 发现严重漏洞则fail
      
    # SAST静态分析
    - name: Bandit security check
      run: bandit -r rag_pipeline/ -ll
      
    # PII检测（确保日志不含敏感信息）
    - name: PII leak detection
      run: |
        python -c "
        import re
        with open('logs/eval.log') as f:
            log = f.read()
        # 检测email、phone、SSN等模式
        patterns = [r'\b[\w.-]+@[\w.-]+\.\w+\b', r'\d{3}-\d{2}-\d{4}']
        for p in patterns:
            matches = re.findall(p, log)
            assert len(matches) == 0, f'PII detected: {matches[:3]}'
        "
```

---

### 7. Canary/Blue-Green部署策略

```
CI Pass → Build Image → Deploy to Staging
                          │
                    ┌─────┴─────┐
                    ▼           ▼
              Canary(5%)   Stable(95%)
                    │           │
               监控指标:      持续服务
               - P95 latency
               - Error rate
               - User thumbs down rate
               - Eval metrics (shadow eval)
                    │
              全部健康? ──Yes──→ Full Rollout
                   │
                  No → Auto Rollback
```

---

## 最佳实践总结

| 实践 | 说明 | 收益 |
|------|------|------|
| **Prompt作为代码管理** | 版本控制prompt模板，纳入Git | 可追溯、可回滚 |
| **合成测试套件** | 维护覆盖common/edge/fail的query集 | 快速发现regression |
| **CI中自动化Index刷新** | staging部署时reindex精简文档集 | 验证ingestion链路完整性 |
| **反馈闭环** | 生产环境收集thumbs up/down → 反哺CI分析 | 缩短feedback loop |
| **并行化测试阶段** | unit test / static analysis / security scan并行 | 缩短总feedback时间 |
| **Feature Branch环境** | 为每个branch创建ephemeral stack | 开发者preview类prod效果 |
| **不可变基础设施** | GitOps方式管理infra变更 | 所有变更可审计、可回滚 |
| **Pipeline健康监控** | 追踪success rate / build time / flakiness | 及时发现脆弋试验 |

---

## 适用场景说明

✅ **推荐使用**:
- 已有RAG系统需要建立自动化质量门禁
- 团队采用GitHub Actions/GitLab CI/CircleCI等CI平台
- 需要防止prompt变更/文档更新导致的性能回退
- 有SLA要求的production RAG服务

⚠️ **前提条件**:
- 有固定的eval dataset（至少50-100 QA对）
- LLM provider支持API调用（CI环境无GPU）
- 团队接受每次code change增加2-5分钟build时间

---

## 参考资源

- CircleCI官方教程: https://circleci.com/blog/automated-rag-pipeline-evaluation-and-benchmarking-with-ragas/
- DZone原文: https://dzone.com/articles/rag-accuracy-metrics-automated-evaluation
- ChatNexus CI/CD指南: https://articles.chatnexus.io/knowledge-base/rag-system-testing-automation-cicd-for-conversatio/
- Databricks Dolly-15k: https://huggingface.co/datasets/databricks/databricks-dolly-15k
- TogetherAI: https://www.together.ai/ ($25免费额度)
