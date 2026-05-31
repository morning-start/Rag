# LangChain 完整 RAG 课程

**来源**: GitHub - sourangshupal/simple-rag-langchain
**链接**: https://github.com/sourangshupal/simple-rag-langchain
**类型**: 7 个渐进式 Jupyter Notebook 教程
**适用章节**: Ch3, Ch4, Ch5

## 课程结构

| Notebook | 主题 | 难度 | 时长 |
|----------|------|------|------|
| 01 | LangChain 基础、LCEL、首次 LLM 调用 | 🔰 入门 | 45min |
| 02 | 文档加载（PDF/CSV/JSON/HTML） | 🔰 入门 | 60min |
| 03 | 文本分块策略 | 🔰→🎓 | 45min |
| 04 | Embeddings（OpenAI/Gemini） | 🔰→🎓 | 45min |
| 05 | 向量存储（InMemory/FAISS/Chroma） | 🎓 进阶 | 60min |
| 06 | 检索策略（Similarity/MMR） | 🎓 进阶 | 45min |
| 07 | 完整 RAG 管线 | 🎓 进阶 | 90min |
| 10 | RAGAS 评估 | 🎓 进阶 | — |
| 11 | LLM as Judge 评估 | 🎓 进阶 | — |
| 12 | RAGAS 指标深入 | 🎓 进阶 | — |

## 技术要点

### LangChain 1.0.5+ 语法
- LCEL 管道操作符 `|`
- `.invoke()` 方法
- 现代包导入

### 样本数据
- products.csv（15 个产品）
- api_response.json（5 篇 AI 文章）
- blog_post.html（RAG 博客文章）
- notes.txt（学习笔记）

### 支持模型
- OpenAI: GPT-3.5-Turbo / GPT-4-Turbo / text-embedding-3-small/large
- Google Gemini: embedding-001
- HuggingFace Embeddings（免费）

### 成本估算
- 完整课程：约 $0.50-1.00
- 1000 页自定义数据：约 $1.50-2.50
- 使用 GPT-3.5-Turbo 可节省 10 倍成本

## 核心价值

1. **零基础友好**：从 Python 环境搭建到生产级 RAG
2. **中文友好**：适合初学者入门 LangChain
3. **生产代码**：所有代码含错误处理、最佳实践
4. **成本可控**：有详细的 API 成本估算