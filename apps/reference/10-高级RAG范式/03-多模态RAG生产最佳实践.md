# 多模态 RAG 生产最佳实践

> 来源综合：
> - [Guide to Multimodal RAG for Images and Text](https://medium.com/kx-systems/guide-to-multimodal-rag-for-images-and-text-10dab36e3117) (KX Systems, 2025-06)
> - [How to Build a Multimodal RAG Pipeline](https://mixpeek.com/guides/multimodal-rag-pipeline-architecture) (Mixpeek, 2026-04)
> - [Multimodal RAG Development: 12 Best Practices for Production](https://www.augmentcode.com/guides/multimodal-rag-development-12-best-practices-for-production-systems) (Augment Code, 2026-01)
> - [Multimodal RAG with Python, PDFs, Images and LLMs](https://blog.stackademic.com/mastering-retrieval-augmented-generation-with-multimodal-data-93067cf6f60e) (Stackademic, 2025-09)

## 为什么多模态 RAG 困难

多模态 RAG 在生产环境的失败率达 73%，主要是因为单模态检索模式无法处理跨模态协调需求：

1. **不同模态需要不同的分块策略**：45 分钟的视频需要时间分割，产品图像需要感兴趣区域检测，播客需要说话人分割
2. **Embedding 空间是模态特定的**：CLIP 图像向量与 Whisper 文本向量不在同一空间，需要对齐或多阶段融合
3. **Context Window 有硬限制**：不能把原始视频传给 LLM，需要提取关键帧/字幕片段并结构化为上下文

## 五阶段管线

| 阶段 | 输入 | 处理 | 输出 |
|------|------|------|------|
| Ingest | 原始媒体文件 | 格式归一化、编解码 | 标准化媒体文件 |
| Perceive | 标准化媒体 | 提取特征/Embedding/元数据 | 模态特定表示 |
| Index | 特征向量 | 存储到向量库 | 可检索索引 |
| Retrieve | 用户查询 | 跨模态检索 | Top-K 多模态结果 |
| Generate | 检索结果 + 查询 | VLM 生成 | 最终回答 |

## 两种检索方法

### 方法 1：统一多模态 Embedding
使用 CLIP/SigLIP 等模型将文本和图像编码到同一向量空间，支持直接跨模态检索。

### 方法 2：模态转换 + 文本检索
先用 VLM 总结图像/音频/视频内容为文本，再用文本 Embedding 模型检索。实现简单但信息有损。

## 生产建议

1. **Late Fusion 策略**：分别检索文本和图片结果，在生成阶段融合
2. **分层存储**：表格元数据（表名、列名、行数）存入向量库用于粗筛，完整内容按需加载
3. **渐进式落地**：先解决 80% 的非结构化文本场景，再逐步引入多模态能力
4. **原生多模态向量数据库**：关注 Qdrant Multimodal Capabilities、Weaviate 等
