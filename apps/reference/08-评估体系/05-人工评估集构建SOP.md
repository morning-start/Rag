# 人工评估集构建SOP

> **来源**: [Twine: Rubric vs Guidelines vs Golden Set (2026.03)](https://www.twine.net/blog/rubric-vs-guidelines-vs-golden-set/) + [DataVLab: Annotation Guidelines & Visual SOPs](https://datavlab.ai/post/how-to-design-effective-annotation-guidelines-and-visual-sops/)
>
> **整理日期**: 2026-05-25
>
> **适用场景**: RAG评估数据集构建、标注质量管理、LLM输出评审体系设计

---

## 核心要点

### 1. 三要素区分：Guidelines / Rubric / Golden Set

这是AI数据质量体系中**最常被混淆的三个概念**，它们解决完全不同的问题：

| 维度 | **Guidelines（标注指南）** | **Rubric（评分规则）** | **Golden Set（金标准）** |
|------|--------------------------|---------------------|----------------------|
| **定位** | 程序性：告诉人**怎么做** | 评价性：告诉评审者**怎么打分** | 诊断性：验证流程**是否有效** |
| **使用时机** | 标注/生产阶段 | 评审/评估阶段 | QA/校准/监控阶段 |
| **核心问题** | "这个该标什么？" | "这个质量如何？" | "我们的流程靠谱吗？" |
| **产出物** | 操作手册 | 评分卡/打分表 | 已验证的信任示例集合 |
| **类比** | IKEA组装说明书 | 考试评分标准答案 | 校准用的标准砝码 |

#### 一句话总结

- **Guidelines创造一致性（在执行层面）**
- **Rubrics创造一致性（在评判层面）**
- **Golden Sets创造一致性（在测量层面）**

---

### 2. Guidelines（标注指南）设计规范

#### 定义与作用

Guidelines是标注员的工作操作手册，定义：
- ✅ 标签类别及其含义
- ✅ 每个类别的边界
- ✅ 边缘案例的处理方式
- ✅ 模糊情况下的决策规则

#### Google Cloud推荐标准

Google Cloud关于human review instructions的最佳实践：

| 要素 | 要求 | 示例 |
|------|------|------|
| **清晰列出标签** | 扁平列表，每类1-2句定义 | `Positive: 表达满意、赞扬或推荐意图` |
| **描述含义** | 不仅说"是什么"，还要说"不是什么"` | `不是Positive: 中性陈述、提问、仅陈述事实` |
| **提供正例** | 每个类别≥2个正确示例 | ✅ "这款产品太棒了！强烈推荐！" → Positive |
| **提供负例** | 常见错误标注示例 | ❌ "产品收到了" → 不是Positive（中性） |
| **显式覆盖边缘案例** | 预判歧义场景并给出明确规则 | "反讽语句按字面意图判断，不按讽刺含义" |

#### 受众定义

Guidelines必须被以下角色理解：

| 角色 | 关注点 | 设计要求 |
|------|-------|---------|
| **标注员**（有领域知识） | 具体操作步骤 | 可快速查阅的操作细节 |
| **标注员**（无领域知识） | 足够背景解释 | 包含domain-specific术语表 |
| **QA Lead** | 质量检查标准 | 明确pass/fail边界 |
| **项目经理** | 进度和交付预期 | 清晰的范围和排除项 |
| **Client/Stakeholder** | 边缘案例决策 | 关键决策点的rationale |

> 💡 **Pro Tip**: 如果你的guideline只能通过Zoom会议讲解才能理解，那它还没写完。

#### 写作风格要求

| ✅ Do | ❌ Don't |
|--------|---------|
| 短句、主动语态 | 长段落、被动语态 |
| 一条规则一行 | 多条规则混在一段 |
| **粗体关键术语** | 模糊修饰语 |
| 具体数值和阈值 | "尽量"、"大约"、"适当" |

**反面教材**:
> ❌ "Try to capture the shape" / "If it looks like a vehicle, mark it" / "Draw carefully"

**正面教材**:
> ✅ "Use a tight polygon around the visible tire, even if partially occluded."
> ✅ "Do not label mirrors unless explicitly mentioned in the class list."
> ✅ "Draw the bounding box from edge to edge of the visible object, ignoring shadows."

---

### 3. Rubric（评分规则）设计规范

#### 定义与作用

Rubric是结构化的**质量评分框架**，用于评判已存在的输出。在LLM评估中尤为重要。

#### 核心评分维度

成熟的rubric应覆盖以下维度（根据用例裁剪）:

| 维度 | 定义 | 典型权重 |
|------|------|---------|
| **Correctness（正确性）** | 事实准确性，无幻觉 | 高 (30-40%) |
| **Completeness（完整性）** | 是否回答了问题的所有方面 | 中高 (20-30%) |
| **Instruction Following（指令遵循）** | 是否遵守了格式、长度等约束 | 中 (15-20%) |
| **Safety/Policy Compliance（安全性）** | 是否违反安全策略 | 高 (一票否决) |
| **Clarity/Style（清晰度）** | 表达是否清晰易懂 | 低 (10-15%) |

#### HealthBench案例：专家加权评分

HealthBench是医疗领域LLM评估的公开benchmark，其核心方法：

```
由执业医师编写domain-specific rubric
  ├── 每个临床问题有明确的scoring criteria
  ├── 不同错误类型赋予不同权重
  │   └── 危险性遗漏 > 措辞不当 > 格式问题
  └── 评分基于predefined checklist而非整体印象
```

**关键原则**: 不同错误的业务影响不同 → 加权评分而非简单平均

#### 二元标注推荐（大多数场景）

对于RAG评估中的faithfulness/relevancy判断：

| 方式 | 适用场景 | 效率 | 成本 |
|------|---------|------|------|
| **Thumbs Up/Down + 必填原因** | 大多数RAG eval | < 30秒/条 | 低 |
| 5级Likert scale | 需要精细区分时 | 45-60秒/条 | 中 |
| 连续打分(1-10) | 学术研究/竞赛 | 60-90秒/条 | 高 |

**二元标注模板**:
```
Question: [用户问题]
Generated Answer: [模型回答]
Context: [检索上下文]

Judgment: □ Thumbs Up  □ Thumbs Down

If Down, reason (required):
□ Hallucination (answer contains info not in context)
□ Irrelevant (answer doesn't address the question)
□ Incomplete (missing key information)
□ Other: ___________
```

---

### 4. Golden Set（金标准）构建规范

#### 定义与核心价值

Golden Set是**经过验证的信任示例集合**，用作测量基线。

**"Golden"的含义 = Trust（信任）**

这种信任来源：
- ✅ 专家审核
- ✅ 多人adjudication达成一致
- ✅ 反复验证无争议

#### Golden Set的用途

| 用途 | 说明 | 频率 |
|------|------|------|
| **标注员校准** | 新标注员是否理解guidelines | Onboarding + 月度 |
| ** reviewer一致性度量** | 不同reviewer间评分差异 | 每批次 |
| **供应商质量追踪** | 外包vendor的质量趋势 | Weekly |
| **模型版本对比** | 同一set上不同版本model的表现 | Each release |
| **回归检测** | prompt/pipeline变更后性能是否回退 | Every deploy |

#### 规模建议

| 类型 | 推荐规模 | 特点 |
|------|---------|------|
| **MVP** | 20-50条 | 快速启动，覆盖主要场景 |
| **Standard** | 100-200条 | 覆盖common + edge cases |
| **Production-grade** | 300-500条 | 分层抽样：60% common / 25% edge / 15% adversarial |

**质量 > 数量**: 小而精心的golden set远优于大但含争议的数据集。

#### 构建流程

```
① Draft: 专家初稿（基于guidelines）
   │
② Review: 第二位专家独立评审
   │
③ Adjudicate: 不一致处讨论裁决
   │
④ Validate: 在pilot标注员上测试一致性
   │    ├── Inter-annotator agreement > 0.8 (Cohen's κ)
   │    └── 否则返回②修订rubric
   │
⑤ Freeze: 版本锁定(v1.0)
   │
⑥ Maintain: 变更需走version control + changelog
```

---

### 5. Visual SOP（可视化标准作业程序）

#### 为什么需要Visual SOP

> 人类是视觉学习者：我们保留**80%看到的信息**，却只保留**10-20%阅读到的信息**。

文字指南的局限：
- "Tight bounding box" —— 每个人理解不同
- "Partial occlusion" —— 边界模糊
- "Visible helmet" —— 多少算visible？

**Visual SOP消除这些歧义**。

#### Visual SOP的七层结构

##### Layer 1: 正确标注示例（"Do This"）

提供多维度变化的正确示例：

```markdown
✔️ 正确标注示例集:

图1: 正面光照，完整可见头盔 → [示意图]
标注说明: "从边缘到边缘绘制bounding box"

图2: 侧影，头盔50%被遮挡 → [示意图]
标注说明: "即使部分遮挡，可见部分仍需标注"

图3: 远景，小目标 → [示意图]
标注说明: "尺寸不影响标注要求，只要可识别"
```

**覆盖变量**:
- 光照条件（日/夜/阴影）
- 目标尺寸（特写/远景）
- 角度/姿态（正面/侧面/俯视）
- 可见程度（完整/部分遮挡/反射）
- 环境多样性（城市/室内/杂乱背景）

##### Layer 2: 错误标注示例（"Don't Do This"）

错误是最好的老师：

```markdown
❌ 错误标注示例集:

图4: Bounding box包含阴影 → [示意图]
问题: "框选了非目标区域"
修正: "紧贴物体边缘，排除投影"

图5: 遗漏小目标 → [示意图]
原因: "因尺寸小被忽略"
提醒: "无论大小，只要符合定义就必须标注"

图6: 类别混淆（摩托车头盔 vs 安全帽）→ [示意图]
错误: "将摩托车头盔标为安全帽"
区分: "参考class definition中的shape特征"
```

##### Layer 3: Edge Cases & Corner Conditions

**这是最可能导致不一致的区域**：

| Edge Case类型 | 处理规则 | 图示 |
|--------------|---------|------|
| 部分可见（<30%） | 若可识别则标注，否则跳过 | [图] |
| 反射/屏幕显示 | 通常不标注（非实体） | [图] |
| 阴影/倒影 | 不标注 | [图] |
| 罕见类别组合 | 按优先级规则处理 | [图] |
| 运动模糊 | 若轮廓可辨则标注 | [图] |
| 镜头光晕/畸变 | 保守处理，宁可漏标 | [图] |

**具体示例**: 
> 📌 "Label the worker if at least **30% of the helmet is visible**."

##### Layer 4: 动画/交互示例（可选）

数字平台支持时使用：
- ▶️ 可播放视频片段（temporal edge cases）
- 🖼️ 可滚动图片画廊（annotations on/off切换）
- 💬 工具内tooltip和警告弹窗

##### Layer 5: Comparative Overlays

同一图像的多版本对比：

```
┌─────────────────┬─────────────────┐
│   ✅ Good       │   ❌ Bad         │
│  [tight box]    │  [loose box]     │
│  IOU=0.92       │  IOU=0.65        │
└─────────────────┴─────────────────┘

差异分析: Bad版本包含了背景噪声，
        导致训练信号不纯净。
```

##### Layer 6: Contextual Labeling

**展示全场景而非局部crop**：

> 📸 不要只展示头盔的zoom-in，而是展示整个建筑工地，解释**为什么这个worker被标了而那个没有**（如：off-duty、不在区域内）。

##### Layer 7: Poster Format（速查表）

最终产物：一页纸/一张图的快速参考：

```
╔══════════════════════════════════════════╗
║     🛡️ HELMET DETECTION - QUICK REF      ║
╠══════════════════════════════════════════╣
║ ✅ DO:                                    ║
║   • Box from edge-to-edge                 ║
║   • Include partial occlusions (>30%)     ║
║   • Label all workers in safety zone      ║
║                                           ║
║ ❌ DON'T:                                  ║
║   • Include shadows/reflections           ║
║   • Skip small-but-visible targets        ║
║   • Guess ambiguous cases → flag instead  ║
║                                           ║
║ ⚠️ EDGE CASES:                            ║
║   • Mirror reflections → SKIP             ║
║   • Through window → LABEL if clear       ║
║   • Motorcycle helmet → SEPARATE CLASS    ║
╚══════════════════════════════════════════╝
```

---

### 6. Decision Trees & Flowcharts

复杂标注逻辑应可视化为决策树：

```
Is the object a person?
  │
  ├─ No → SKIP
  │
  ├─ Yes → Is wearing a helmet?
  │        │
  │        ├─ No → Is hard hat required in zone?
  │        │        │
  │        │        ├─ Yes → FLAG as violation
  │        │        └─ No → SKIP
  │        │
  │        └─ Yes → Is ≥30% visible?
  │                 │
  │                 ├─ Yes → DRAW bounding box
  │                 └─ No → SKIP
  │
  └─ Output: labeled / skipped / flagged
```

**工具推荐**: Draw.io / Lucidchart（免费，支持导出嵌入文档）

---

### 7. 质量控制机制

| 机制 | 方法 | 频率 |
|------|------|------|
| **标注准确性** | Golden set抽检（10-20%） | 每批次 |
| **一致性度量** | Cohen's Kappa / Fleiss' Kappa | Weekly |
| **模糊案例记录** | 决策日志 + rationale | 实时 |
| **校准会议** | 团队讨论disagreed cases | Bi-weekly |
| **漂移检测** | 追踪同一标注员的score趋势 | Monthly |

**Inter-Annotator Agreement基准**:
- κ > 0.80: ✅ 优秀，可以继续
- κ 0.60-0.79: ⚠️ 可接受，需要校准
- κ < 0.60: ❌ 不可接受，需重写guidelines

---

### 8. 版本管理规范

Golden Set和Guidelines都需要严格的版本控制：

```markdown
## Changelog

### v2.1 (2026-05-20)
- **Added**: 新增"through window visibility"规则
- **Changed**: 最小可见比例从25%调整为30%
- **Fixed**: 修复motorcycle helmet vs hard hat的判定逻辑
- **Author**: QA Lead Zhang San
- **Reviewer**: Domain Expert Li Si

### v2.0 (2026-04-15)
- **Major**: 重构edge case分类体系
- **Added**: 新增reflection/exclusion规则
- **Deprecated**: 移除v1.x的"best effort"条款
```

**最佳实践**:
- 使用Git管理所有SOP文件
- 每次更新通知全体stakeholders
- 保留历史版本以支持审计
- 平台支持的话启用in-tool alerts

---

### 9. Training & Onboarding流程

即使最完美的guidelines，如果没人知道怎么用也无效：

| 步骤 | 内容 | 时长 |
|------|------|------|
| **Video walkthrough** | 录制SOP讲解视频 | 10min |
| **Edge case quiz** | 关键边缘案例测试 | 10min |
| **Practice with feedback** | 带反馈的练习任务 | 20min |
| **Mentor access** | 指定lead annotator答疑 | Ongoing |
| **Performance tracking** | 追踪初期准确率趋势 | 第1-2周 |

> 🧩 **Pro tip**: 一次20分钟的onboarding session可以防止后续数百条误标。

---

## 适用场景说明

✅ **推荐使用本SOP的场景**:

1. **搭建RAG人工评估体系**: 从零建立可信的eval dataset
2. **外包标注团队管理**: 为vendor提供清晰可执行的标准
3. **LLM output评审**: 构建faithfulness/relevancy人工验证流程
4. **质量控制体系建设**: 建立可持续的QA metrics和calibration机制

⚠️ **核心原则**:
- Guidelines → Rubric → Golden Set **三者缺一不可**
- 先写guidelines让工作可执行，再建rubric让评判一致，最后用golden set验证流程有效
- 这是一个**迭代反馈循环**，不是一次性文档

---

## 参考资源

- Twine原文: https://www.twine.net/blog/rubric-vs-guidelines-vs-golden-set/
- DataVLab Visual SOPs: https://datavlab.ai/post/how-to-design-effective-annotation-guidelines-and-visual-sops/
- Google Cloud Data Labeling Guide: https://cloud.google.com/use-cases/data-labeling
- OpenAI Evaluation Best Practices: platform.openai.com/docs/guides/evaluation
- HealthBench Benchmark: github.com/healthbench/healthbench
- Labelbox Benchmark Workflows: labelbox.com/docs/quality-benchmarking
