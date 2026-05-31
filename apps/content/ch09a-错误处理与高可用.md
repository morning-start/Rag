# 第9A章 错误处理与高可用

## 9A.1 错误处理与容错

### 9A.1.1 重试机制与指数退避

分布式系统中，网络抖动、服务暂时不可用是常态而非异常。重试机制是应对瞬态故障的第一道防线。但朴素的重试（固定间隔、无限次）可能适得其反——在服务真正宕机时，大量客户端同时重试会引发"重试风暴"，反而加剧故障。

**指数退避（Exponential Backoff）** 是重试的黄金标准：每次重试的等待时间按指数增长，并加入随机抖动（Jitter）避免多个客户端同步重试。该算法最早由以太网 CSMA/CD 协议采用，后被广泛用于 TCP 重传、分布式系统等领域。

```python
import asyncio
import random
import logging
from typing import Callable, TypeVar

T = TypeVar("T")
logger = logging.getLogger(__name__)

class RetryConfig:
    """重试配置"""
    max_retries: int = 3           # 最大重试次数
    base_delay: float = 1.0        # 初始延迟（秒）
    max_delay: float = 60.0        # 最大延迟上限
    exponential_base: float = 2    # 指数基数
    jitter: bool = True            # 是否添加随机抖动
    retryable_exceptions: tuple = (
        ConnectionError,
        TimeoutError,
        asyncio.TimeoutError,
    )

async def retry_with_backoff(
    func: Callable[..., T],
    config: RetryConfig = RetryConfig(),
    **kwargs
) -> T:
    """带指数退避和抖动的异步重试装饰器"""
    last_exception = None

    for attempt in range(config.max_retries + 1):
        try:
            return await func(**kwargs)
        except config.retryable_exceptions as e:
            last_exception = e
            if attempt == config.max_retries:
                logger.error(
                    f"重试耗尽 ({config.max_retries} 次), 最后错误: {e}"
                )
                raise

            # 计算退避时间：base * exponential_base ^ attempt
            delay = min(
                config.base_delay * (config.exponential_base ** attempt),
                config.max_delay
            )

            # 添加抖动：±50% 随机偏移
            if config.jitter:
                delay = delay * (0.5 + random.random())

            logger.warning(
                f"第 {attempt + 1} 次重试，{delay:.1f}s 后执行，错误: {e}"
            )
            await asyncio.sleep(delay)

    raise last_exception  # 理论上不会到达


# 使用示例
async def call_llm(prompt: str, max_tokens: int = 2048) -> str:
    """调用 LLM，带自动重试"""
    return await retry_with_backoff(
        lambda **kw: llm.ainvoke(kw["prompt"]),
        prompt=prompt
    )
```

**退避时间序列示例**（base=1s, exponential_base=2, jitter）：
- 第 1 次重试：~1s（0.5-1.5s）
- 第 2 次重试：~2s（1.0-3.0s）
- 第 3 次重试：~4s（2.0-6.0s）
- 第 4 次重试：~8s（4.0-12.0s）

### 9A.1.2 熔断器模式

当后端服务持续故障时，重试只会浪费资源并延长用户等待时间。**熔断器（Circuit Breaker）** 模式借鉴了电路中的保险丝概念：当错误率超过阈值时"断开电路"，直接拒绝请求（快速失败），避免级联故障扩散。

熔断器有三种状态：
```text
        错误率超阈值              冷却期结束且探测成功
  ┌──────────────┐              ┌──────────────┐
  │              │──────────────→│              │
  │   关闭       │              │   半开       │
  │ (正常放行)    │←──────────────│ (试探性放行)  │
  │              │  探测失败      │              │
  └──────┬───────┘              └──────────────┘
         │ 错误率超阈值
         ▼
  ┌──────────────┐
  │              │
  │   打开       │
  │ (直接拒绝)    │
  │              │
  └──────────────┘
         │ 冷却期结束
         ▼
      进入半开状态
```

```python
import time
import threading
from enum import Enum
from collections import deque

class CircuitState(Enum):
    CLOSED = "closed"       # 正常放行
    OPEN = "open"           # 熔断，直接拒绝
    HALF_OPEN = "half_open" # 试探性放行

class CircuitBreaker:
    """熔断器实现"""

    def __init__(
        self,
        failure_threshold: int = 5,     # 连续失败 5 次触发熔断
        recovery_timeout: float = 30.0, # 熔断后 30 秒进入半开状态
        half_open_max_calls: int = 3,   # 半开状态最多放行 3 个请求
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        self.half_open_success_count = 0
        self._lock = threading.Lock()

    def can_execute(self) -> bool:
        """判断是否允许执行请求"""
        with self._lock:
            if self.state == CircuitState.CLOSED:
                return True

            if self.state == CircuitState.OPEN:
                # 检查是否到达冷却期
                if time.time() - self.last_failure_time >= self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_success_count = 0
                    return True
                return False  # 仍在冷却期，拒绝请求

            if self.state == CircuitState.HALF_OPEN:
                return self.half_open_success_count < self.half_open_max_calls

        return False

    def record_success(self):
        """记录成功调用"""
        with self._lock:
            self.failure_count = 0
            if self.state == CircuitState.HALF_OPEN:
                self.half_open_success_count += 1
                if self.half_open_success_count >= self.half_open_max_calls:
                    self.state = CircuitState.CLOSED  # 探测成功，恢复

    def record_failure(self):
        """记录失败调用"""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN  # 探测失败，重新熔断
            elif self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN

# 使用示例
llm_circuit = CircuitBreaker(failure_threshold=5, recovery_timeout=30)

async def call_llm_with_circuit(prompt: str) -> str:
    if not llm_circuit.can_execute():
        raise Exception("LLM 服务熔断中，请稍后重试")
    try:
        result = await llm.ainvoke(prompt)
        llm_circuit.record_success()
        return result
    except Exception as e:
        llm_circuit.record_failure()
        raise
```

### 9A.1.3 降级策略

当核心服务不可用时，降级策略确保系统仍能提供基本功能，而非完全不可用。降级是"优雅失败"（Graceful Degradation）理念的具体实践。

| 降级场景 | 降级策略 | 用户体验影响 |
|---------|---------|------------|
| LLM 服务不可用 | 返回缓存的历史回答 + 提示"当前回答可能非最新" | 可接受 |
| 向量数据库不可用 | 降级为 BM25 关键词检索（ES） | 检索质量下降但可用 |
| 重排序模型不可用 | 跳过重排序，直接使用向量检索结果 | 精度下降但可用 |
| Embedding 服务不可用 | 返回"系统维护中"页面 + 预设 FAQ | 最小可用 |

```python
class RAGServiceWithFallback:
    """带降级策略的 RAG 服务"""

    def __init__(self, primary_llm, fallback_llm, cache, bm25_retriever):
        self.primary_llm = primary_llm
        self.fallback_llm = fallback_llm     # 备用 LLM（如更小的模型）
        self.cache = cache
        self.bm25_retriever = bm25_retriever
        self.llm_circuit = CircuitBreaker(failure_threshold=5, recovery_timeout=30)

    async def query(self, question: str) -> dict:
        # 第一层降级：缓存命中
        cached = self.cache.get_cached_result(question)
        if cached:
            cached["from_cache"] = True
            return cached

        # 第二层降级：检索降级（向量检索 → BM25）
        try:
            docs = await retriever.ainvoke(question)
        except Exception:
            docs = self.bm25_retriever.search(question, top_k=5)

        # 第三层降级：LLM 降级（主模型 → 备用模型 → 模板回答）
        if not self.llm_circuit.can_execute():
            answer = self._template_answer(question, docs)
        else:
            try:
                answer = await self.primary_llm.ainvoke(
                    rag_prompt.format(context=format_docs(docs), question=question)
                )
                self.llm_circuit.record_success()
            except Exception:
                try:
                    # 降级到备用模型
                    answer = await self.fallback_llm.ainvoke(
                        rag_prompt.format(context=format_docs(docs), question=question)
                    )
                except Exception:
                    # 最终降级：模板回答
                    answer = self._template_answer(question, docs)
                    self.llm_circuit.record_failure()

        return {"answer": answer, "sources": [d.metadata for d in docs]}

    def _template_answer(self, question: str, docs: list) -> str:
        """模板降级回答：直接返回检索到的文档片段"""
        if not docs:
            return "抱歉，当前服务暂时不可用，请稍后重试。"
        context = "\n\n".join([f"参考资料 {i+1}: {d.page_content[:200]}" for i, d in enumerate(docs[:3])])
        return f"以下是与您的问题相关的参考资料（AI 生成服务暂时不可用）：\n\n{context}"
```

---

## 9A.2 数据备份与灾难恢复

### 9A.2.1 向量数据库备份策略

向量数据库是 RAG 系统的核心资产——丢失向量索引意味着需要重新处理所有文档，耗时可能从数小时到数天。备份策略的设计取决于 RPO（Recovery Point Objective，可容忍的数据丢失时间窗口）和 RTO（Recovery Time Objective，可接受的恢复时间）。

| 备份方案 | RPO | RTO | 复杂度 | 适用场景 |
|---------|-----|-----|--------|---------|
| **快照备份** | 小时级 | 分钟级 | 低 | 中小规模（<100 万向量） |
| **增量同步** | 分钟级 | 分钟级 | 中 | 大规模，实时性要求高 |
| **多副本复制** | 秒级 | 秒级 | 高 | 金融/医疗等关键场景 |

```bash
# Qdrant 快照备份与恢复
# 1. 创建快照
curl -X POST 'http://localhost:6333/collections/rag_docs/snapshots'

# 2. 列出快照
curl 'http://localhost:6333/collections/rag_docs/snapshots'

# 3. 下载快照（备份到远程存储）
curl -o rag_docs_snapshot.tar 'http://localhost:6333/collections/rag_docs/snapshots/{snapshot_name}'

# 4. 恢复：将快照文件放入 Qdrant 的 snapshots 目录后执行
curl -X PUT 'http://localhost:6333/collections/rag_docs/snapshots/upload?priority=snapshot' \
    --data-binary @rag_docs_snapshot.tar
```

```bash
# Milvus 备份（使用 milvus-backup 工具）
# 1. 配置备份目标（S3 / 本地路径）
cat > backup.yaml <<EOF
configs:
  milvus:
    address: localhost
    port: 19530
  storage:
    type: local
    rootPath: /data/milvus_backup
EOF

# 2. 创建备份
./milvus-backup create -n rag_backup_20260529

# 3. 恢复备份
./milvus-backup restore -n rag_backup_20260529
```

### 9A.2.2 索引重建方案

当灾难发生且备份不可用时，需要从原始文档全量重建索引。一个健壮的索引重建方案应满足：可中断续跑、进度可追踪、重建期间不影响在线服务。

```python
class IndexRebuilder:
    """索引重建管理器：支持断点续跑和进度追踪"""

    def __init__(self, doc_source, vectorstore, embedding_model, checkpoint_store):
        self.doc_source = doc_source          # 文档来源（S3 / 本地目录 / 数据库）
        self.vectorstore = vectorstore
        self.embedding_model = embedding_model
        self.checkpoint_store = checkpoint_store  # 进度持久化（Redis / SQLite）

    def rebuild(self, batch_size: int = 100):
        """全量重建索引，支持断点续跑"""
        # 1. 检查是否有未完成的重建任务
        checkpoint = self.checkpoint_store.get("rebuild_progress")
        start_offset = checkpoint.get("last_offset", 0) if checkpoint else 0

        if start_offset > 0:
            print(f"检测到未完成的重建任务，从偏移量 {start_offset} 继续")

        # 2. 清空旧索引（仅在首次启动时）
        if start_offset == 0:
            self.vectorstore.delete_collection()
            self.vectorstore.create_collection()

        # 3. 分批处理文档
        all_docs = self.doc_source.list_all()
        total = len(all_docs)
        processed = 0

        for i in range(start_offset, total, batch_size):
            batch = all_docs[i:i + batch_size]

            # 加载并处理文档
            chunks = []
            for doc_path in batch:
                pages = self.doc_source.load(doc_path)
                chunks.extend(text_splitter.split_documents(pages))

            # 批量 Embedding 和写入
            texts = [c.page_content for c in chunks]
            embeddings = self.embedding_model.embed_batch(texts)
            self.vectorstore.add_embeddings(
                texts=texts,
                embeddings=embeddings,
                metadatas=[c.metadata for c in chunks]
            )

            processed += len(batch)

            # 4. 更新检查点（每批完成后持久化进度）
            self.checkpoint_store.set("rebuild_progress", {
                "last_offset": i + batch_size,
                "total": total,
                "processed": processed,
                "timestamp": datetime.utcnow().isoformat()
            })

            print(f"进度: {processed}/{total} ({processed/total*100:.1f}%)")

        # 5. 完成：清除检查点
        self.checkpoint_store.delete("rebuild_progress")
        print(f"索引重建完成，共处理 {total} 篇文档")
```

**灾难恢复清单**：

| 检查项 | 说明 | 验证方式 |
|--------|------|---------|
| 向量数据库快照 | 每日自动快照，保留 7 天 | 定期恢复测试 |
| 元数据数据库备份 | PostgreSQL 每日全量 + WAL 归档 | `pg_restore` 验证 |
| 原始文档存储 | S3 跨区域复制 / 本地 NAS 双副本 | 文件完整性校验 |
| 配置文件版本管理 | Git 仓库管理所有配置 | CI/CD 配置变更检测 |
| 索引重建脚本 | 定期演练，确保可在一小时内完成 | 混沌工程测试 |
| 监控告警 | 备份失败、存储空间不足时立即告警 | PagerDuty / 飞书告警 |

### 9A.2.3 RTO / RPO 规划

灾难恢复的核心指标是 RTO（Recovery Time Objective，恢复时间目标）和 RPO（Recovery Point Objective，恢复点目标）。针对 RAG 系统的各组件，建议的 RTO/RPO 如下：（来源: [RAG错误处理与高可用模式](reference/09-生产部署与运维/06-RAG错误处理与高可用模式.md)）

| 组件 | 建议 RTO | 建议 RPO | 恢复策略 |
|------|---------|---------|---------|
| 向量数据库 | ≤ 5 分钟 | ≤ 1 小时 | 主备部署 + WAL 日志 |
| 关系数据库（元数据） | ≤ 2 分钟 | ≤ 5 分钟 | 主从复制 + WAL 归档 |
| LLM 网关 | ≤ 1 分钟 | 无状态 | 多副本负载均衡 |
| Embedding 服务 | ≤ 5 分钟 | 无状态 | 多副本 + 缓存预热 |
| 原始文档存储 | ≤ 30 分钟 | ≤ 24 小时 | 跨区域复制 + 定期校验 |
| 索引重建服务 | ≤ 1 小时 | 不适用 | 定期演练 + 断点续跑 |

**关键原则**：
- 核心在线服务（检索、生成）RTO 应 ≤ 5 分钟，确保对用户影响最小化
- 索引数据建议每小时快照，RPO ≤ 1 小时，最多丢失 1 小时的增量数据
- 定期进行混沌工程测试，验证 RTO/RPO 指标是否达标

## 9A.3 断路器模式与优雅降级

### 9A.3.1 断路器（Circuit Breaker）

重试机制在瞬态故障时有效，但在服务真正宕机时，大量并发重试会引发"重试风暴"加剧故障。断路器模式通过检测错误率达到阈值时"断开电路"，快速失败而非继续重试：

```python
import asyncio
import time
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    CLOSED = "closed"       # 正常状态，请求通过
    OPEN = "open"           # 断开状态，快速失败
    HALF_OPEN = "half_open" # 半开状态，尝试恢复

class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_requests: int = 3
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_requests = half_open_max_requests
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.half_open_requests = 0

    async def call(self, func, *args, **kwargs):
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.half_open_requests = 0
                logger.info("断路器进入半开状态，尝试恢复")
            else:
                raise CircuitBreakerOpenError("断路器已断开，请求被拒绝")

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        if self.state == CircuitState.HALF_OPEN:
            self.half_open_requests += 1
            if self.half_open_requests >= self.half_open_max_requests:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                logger.info("断路器恢复为关闭状态")
        else:
            self.failure_count = 0

    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"失败次数达到阈值 {self.failure_threshold}，断路器断开")

class CircuitBreakerOpenError(Exception):
    pass
```

### 9A.3.2 模型回退链（Model Fallback Chain）

当主模型不可用或响应质量下降时，按优先级链回退到备用模型：

```python
MODEL_FALLBACK_CHAIN = [
    {"model": "qwen2.5:72b", "provider": "local", "priority": 1},
    {"model": "gpt-4o", "provider": "openai", "priority": 2},
    {"model": "claude-sonnet-4", "provider": "anthropic", "priority": 3},
    {"model": "gemini-2.0-flash", "provider": "google", "priority": 4},
]

async def call_with_fallback(prompt: str, max_retries: int = 2) -> str:
    for tier in MODEL_FALLBACK_CHAIN:
        for attempt in range(max_retries):
            try:
                return await call_llm(prompt, model=tier["model"])
            except (ConnectionError, TimeoutError, RateLimitError) as e:
                logger.warning(f"模型 {tier['model']} 失败: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                continue
        logger.info(f"回退到下一级模型: {tier['model']}")
    raise AllModelsFailedError("所有模型均不可用")
```

### 9A.3.3 优雅降级策略

当系统整体负载过高或部分组件不可用时，通过动态调整响应策略实现降级：

| 降级级别 | 触发条件 | 行为 | 用户体验影响 |
|---------|---------|------|-------------|
| **L0 正常** | 全部组件正常 | 完整 RAG 管线：检索 + 重排 + 生成 | 最佳体验 |
| **L1 精简** | 重排服务不可用 | 跳过重排，直接使用向量检索 Top-K 结果 | 回答质量可能下降 5-15% |
| **L2 降级** | LLM 服务压力大 | 使用缓存命中替代 LLM 生成，或返回检索摘要 | 响应更快但精度下降 |
| **L3 容灾** | 向量数据库不可用 | 使用备用 BM25 关键词检索，或返回静态 FAQ | 仅支持简单问答 |
| **L4 熔断** | 核心组件全不可用 | 返回友好提示："系统维护中，请稍后再试" | 服务不可用但系统不崩溃 |

```python
class GracefulDegradation:
    def __init__(self):
        self.health_checks = {
            "vector_db": False,
            "reranker": False,
            "llm": False,
            "bm25": True,  # BM25 通常本地可用
        }

    def get_degradation_level(self) -> int:
        if all(self.health_checks.values()):
            return 0  # L0 正常
        if not self.health_checks["reranker"]:
            return 1  # L1 精简
        if not self.health_checks["llm"]:
            return 2  # L2 降级
        if not self.health_checks["vector_db"]:
            return 3  # L3 容灾
        return 4  # L4 熔断

    async def handle_query(self, query: str) -> str:
        level = self.get_degradation_level()
        if level == 0:
            return await self.full_rag_pipeline(query)
        elif level == 1:
            return await self.skip_rerank_pipeline(query)
        elif level == 2:
            return await self.cache_or_summary_pipeline(query)
        elif level == 3:
            return await self.bm25_fallback_pipeline(query)
        else:
            return "系统正在维护升级，请稍后再试。如需紧急帮助，请联系 support@example.com"
```

**核心设计哲学**：降级不是"功能裁剪"，而是**可预期的行为切换**。每个降级级别都有明确的触发条件、行为定义和用户体验声明，让团队在故障发生时能快速判断当前系统的服务能力边界。

---



