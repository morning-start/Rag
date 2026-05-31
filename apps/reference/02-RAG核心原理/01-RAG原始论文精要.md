# Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks

**来源**: arXiv:2005.11401 (NeurIPS 2020)
**作者**: Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela
**链接**: https://arxiv.org/abs/2005.11401

## 核心摘要

大型预训练语言模型已被证明可以在参数中存储事实知识，并在下游 NLP 任务上进行微调时取得最先进的结果。然而，它们访问和精确操作知识的能力仍然有限，因此在知识密集型任务上，它们的性能落后于特定任务架构。此外，为其决策提供依据和更新其世界知识仍然是开放的研究问题。

具有可微访问机制到显式非参数记忆的预训练模型可以克服这个问题，但迄今为止仅针对提取式下游任务进行了研究。我们探索了一种用于检索增强生成（RAG）的通用微调方法——该模型结合了预训练的**参数记忆**和**非参数记忆**用于语言生成。

## 核心贡献

1. **RAG 模型定义**: 参数记忆是预训练的 seq2seq 模型（BART），非参数记忆是 Wikipedia 的密集向量索引，通过预训练的神经检索器访问。

2. **两种 RAG 公式**:
   - **RAG-Sequence**: 整个生成序列使用相同的检索文档
   - **RAG-Token**: 每个 token 可以使用不同的检索文档

3. **实验结果**:
   - 在三个开放域 QA 任务上取得了最先进的结果
   - 优于参数化 seq2seq 模型和特定任务的检索-提取架构
   - 生成更具体、多样化和事实性的语言

## 关键概念

- **参数记忆 (Parametric Memory)**: 存储在模型权重中的知识
- **非参数记忆 (Non-parametric Memory)**: 外部知识库（如 Wikipedia 索引）
- **密集段落检索 (Dense Passage Retrieval, DPR)**: 用于高效检索的神经检索器
- **知识密集型任务 (Knowledge-Intensive Tasks)**: 需要外部知识的 NLP 任务

## 引用信息

```
@inproceedings{lewis2020retrieval,
  title={Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks},
  author={Lewis, Patrick and Perez, Ethan and Piktus, Aleksandra and Petroni, Fabio and Karpukhin, Vladimir and Goyal, Naman and K\"{u}ttler, Heinrich and Lewis, Mike and Yih, Wen-tau and Rockt\"{a}schel, Tim and Riedel, Sebastian and Kiela, Douwe},
  booktitle={Advances in Neural Information Processing Systems},
  year={2020}
}
```