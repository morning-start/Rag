# 训练层优化：SFT/DPO/RLHF 在 RAG 场景中的应用

> 来源综合：
> - [What SFT, DPO, RLHF, and RAG Actually Do in an AI Agent](https://pub.towardsai.net/what-sft-dpo-rlhf-and-rag-actually-do-in-an-ai-agent-d5b8daf0aedb) (Towards AI, 2026-03)
> - [Post-Training LLMs Guide: SFT, RLHF, DPO & GRPO](https://www.sundeepteki.org/advice/the-complete-guide-to-post-training-llms-how-sft-rlhf-dpo-and-grpo-shape-llms) (Sundeep Teki, 2026-05)
> - [Post-Training Large Language Models: SFT, DPO, and RLHF Explained](https://python.plainenglish.io/post-training-large-language-models-sft-dpo-and-rlhf-explained-3e0d17a84914) (Python in Plain English, 2025-09)
> - [Fine-Tuning Techniques - Choosing Between SFT, DPO, and RFT](https://developers.openai.com/cookbook/examples/fine_tuning_direct_preference_optimization_guide) (OpenAI Cookbook, 2025-06)
> - [Conditional Equivalence of DPO and RLHF](https://arxiv.org/abs/2605.20834) (arXiv, 2026-05)

## 概述

SFT、DPO、RLHF 是三种不同层次的训练优化方法，它们与 RAG 构成互补关系：

- **SFT (Supervised Fine-Tuning)**：使用标注的指令-回答对训练模型，学习正确的回答风格和行为模式。最基础的训练优化方法。
- **DPO (Direct Preference Optimization)**：直接基于偏好对（preferred/rejected pairs）优化模型，无需训练独立的奖励模型。计算效率高、训练稳定。
- **RLHF (Reinforcement Learning from Human Feedback)**：通过奖励模型 + 强化学习（PPO/GRPO）优化模型。最强大的对齐方法，但训练复杂度和计算成本最高。

## 三种方法的对比

| 维度 | SFT | DPO | RLHF |
|------|-----|-----|------|
| 数据需求 | 指令-回答对 | 偏好对（好/坏回答） | 偏好对 + 奖励模型训练数据 |
| 训练复杂度 | 低 | 中 | 高 |
| 计算成本 | 低 | 中 | 高 |
| 对齐效果 | 基础行为学习 | 偏好对齐 | 深度对齐 |
| 训练稳定性 | 稳定 | 稳定 | 需要精细调参 |
| 典型工具 | LoRA, QLoRA | TRL, Axolotl | TRL, DeepSpeed |

## 2026 年最新趋势

1. **三阶段管线成为标准**：SFT → Preference Alignment (DPO/RLHF) → RL with Verifiable Rewards (GRPO/DAPO)
2. **GRPO 兴起**：DeepSeek R1 采用的 Group Relative Policy Optimization，无需 Critic Model，降低 RLHF 训练门槛
3. **SFT + RAG 组合成为生产基线**：先通过 SFT 让模型理解领域语言和格式规范，再用 RAG 提供实时知识注入
4. **CPO (Constrained Preference Optimization)**：arXiv 2026 年 5 月最新研究，在 DPO 基础上增加约束确保对齐的可靠性

## 对 RAG 的价值

- SFT 让模型更好地理解"基于上下文回答"的指令格式
- DPO/RLHF 让模型学习"不确定时拒绝回答"的行为偏好
- 训练层优化可以与 Prompt 层、管线层、参数层的优化叠加使用，效果相乘
