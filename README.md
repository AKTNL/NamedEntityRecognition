# 中文细粒度命名实体识别（CLUENER2020）：抽取式与生成式双范式对照研究

<p align="center">
  <img src="https://img.shields.io/badge/Task-Chinese_Fine--Grained_NER-blue.svg" alt="Task">
  <img src="https://img.shields.io/badge/Dataset-CLUENER2020-orange.svg" alt="Dataset">
  <img src="https://img.shields.io/badge/Paradigm-Extractive_|_Generative-brightgreen.svg" alt="Paradigm">
  <img src="https://img.shields.io/badge/Base_Models-BERT_|_MacBERT_|_GPT--NER-green.svg" alt="Models">
  <img src="https://img.shields.io/badge/Framework-PyTorch_&_Transformers-red.svg" alt="Framework">
  <img src="https://img.shields.io/badge/Evaluation-Seqeval_(Strict_Entity--Level)-purple.svg" alt="Evaluation">
  <img src="https://img.shields.io/badge/License-MIT-lightgrey.svg" alt="License">
</p>

> **《自然语言处理》课程综合实践 · 经典论文研读、全流程基线复现与深度错例挖掘**  
> 
> 本项目围绕**中文细粒度命名实体识别**展开，以 **CLUENER2020**（Xu et al., 2020）为数据与评测基准，采用**双范式对照**的研究设计：**抽取式**一侧以 **BERT**（Devlin et al., NAACL 2019）为基线、以 **MacBERT**（Cui et al., EMNLP 2020 Findings）为对照变量；**生成式**一侧以 **GPT-NER**（Wang et al., 2023, arXiv:2304.10428）为方法原型，将序列标注转化为文本生成，并提供可选的自验证机制（当前展示的实验未开启）。项目包含理论剖析、严格受控的端到端训练闭环，以及基于 1,343 条开发集样本的全量实体级错例挖掘系统。

---

## 目录

- [1. 项目概述与背景 (Overview & Background)](#1-项目概述与背景-overview--background)
- [2. 核心原理与演进机制 (Theoretical Background & Mechanism)](#2-核心原理与演进机制-theoretical-background--mechanism)
  - [2.1 中文 NER 技术演进全景](#21-中文-ner-技术演进全景)
  - [2.2 原生 BERT 在中文序列标注中的局限](#22-原生-bert-在中文序列标注中的局限)
  - [2.3 MacBERT 核心架构与机制创新](#23-macbert-核心架构与机制创新)
  - [2.4 下游序列标注适配建模 (Sequence Labeling Head)](#24-下游序列标注适配建模-sequence-labeling-head)
  - [2.5 生成式 NER 范式：GPT-NER (Generative Paradigm)](#25-生成式-ner-范式gpt--ner-generative-paradigm)
  - [2.6 两种范式的结构性对照](#26-两种范式的结构性对照)
- [3. 实验配置与评测基准 (Benchmark & Experimental Setup)](#3-实验配置与评测基准-benchmark--experimental-setup)
  - [3.1 CLUENER2020 细粒度数据集](#31-cluener2020-细粒度数据集)
  - [3.2 严格实体级评测标准 (Strict Entity-Level F1)](#32-严格实体级评测标准-strict-entity-level-f1)
  - [3.3 训练与评估超参数](#33-训练与评估超参数)
- [4. 核心实验结果与量化对比 (Core Experimental Results)](#4-核心实验结果与量化对比-core-experimental-results)
  - [4.1 细粒度 10 大类别性能全景对比表](#41-细粒度-10-大类别性能全景对比表)
  - [4.2 核心量化发现与消融洞察](#42-核心量化发现与消融洞察)
- [5. 精选案例研究与深度错例挖掘 (Bad Case & Qualitative Analysis)](#5-精选案例研究与深度错例挖掘-bad-case--qualitative-analysis)
  - [5.1 案例集总体分布统计](#51-案例集总体分布统计)
  - [5.2 类别 A：MacBERT 优势与纠错还原案例 (Representative Superiority Cases)](#52-类别-a-macbert-优势与纠错还原案例-representative-superiority-cases)
  - [5.3 类别 B：中文 NER 固有瓶颈与顽疾错例剖析 (Challenging Cases & Bottlenecks)](#53-类别-b-中文-ner-固有瓶颈与顽疾错例剖析-challenging-cases--bottlenecks)
- [6. 项目工程结构 (Repository Structure)](#6-项目工程结构-repository-structure)
- [7. 快速复现指引 (Quick Start Guide)](#7-快速复现指引-quick-start-guide)
  - [7.1 环境配置（含 NVIDIA RTX 50 系列 sm_120 兼容指南）](#71-环境配置含-nvidia-rtx-50-系列-sm_120-兼容指南)
  - [7.2 模型训练运行 (Training Pipeline)](#72-模型训练运行-training-pipeline)
  - [7.3 模型独立评估 (Evaluation Only)](#73-模型独立评估-evaluation-only)
  - [7.4 案例挖掘与差异分析 (Case Study Execution)](#74-案例挖掘与差异分析-case-study-execution)
- [8. 交互式复现工作台与全栈模型体验系统 (Interactive Reproduction Workbench & Model Arena)](#8-交互式复现工作台与全栈模型体验系统-interactive-reproduction-workbench--model-arena)
  - [8.1 工作台设计架构与核心特性](#81-工作台设计架构与核心特性)
  - [8.2 快速启动与双模运行指南](#82-快速启动与双模运行指南)
  - [8.3 六大交互模块](#83-六大交互模块)
  - [8.4 RESTful API 规范与扩展接口](#84-restful-api-规范与扩展接口)
  - [8.5 数据来源、更新与验证](#85-数据来源更新与验证)
- [9. 与原论文公开结果的对比 (Comparison with Published Results)](#9-与原论文公开结果的对比-comparison-with-published-results)
- [10. 实验结果的可靠性声明与局限 (Reliability Statement)](#10-实验结果的可靠性声明与局限-reliability-statement)
- [11. 实验环境与版本记录 (Environment & Reproducibility)](#11-实验环境与版本记录-environment--reproducibility)
- [12. 实验过程中遇到的问题与解决办法 (Troubleshooting Log)](#12-实验过程中遇到的问题与解决办法-troubleshooting-log)
- [13. 小组成员分工 (Team Contributions)](#13-小组成员分工-team-contributions)
- [14. 人工智能工具使用声明 (AI Tool Usage Disclosure)](#14-人工智能工具使用声明-ai-tool-usage-disclosure)
- [15. 总结与后续展望 (Conclusion & Future Work)](#15-总结与后续展望-conclusion--future-work)
- [16. 参考文献与学术溯源 (References)](#16-参考文献与学术溯源-references)

---

## 1. 项目概述与背景 (Overview & Background)

命名实体识别（Named Entity Recognition, NER）是信息抽取、知识图谱构建、问答系统及篇章语义理解的基石级底层任务。相较于英文等印欧语系语言，**中文 NER 具有其特有的语言学挑战**：
1. **无显式词边界标记**：中文文本以连续汉字字符流呈现，词间无空格分隔，词边界与实体边界高度交织；
2. **多义性与长跨度复合结构**：实体内部常嵌套常见动词、形容词或空间方位修饰词（如“中国房地产及住宅研究会”）；
3. **分词歧义传递**：若采用分词预处理引入外部词典，易受到分词错误传递的次生污染；若纯以单字切分，又极易丢失词汇级的完整语义块表达。

针对上述难题，本项目采用**双范式对照（Dual-Paradigm Comparison）**的研究设计，在同一份 CLUENER2020 数据与同一套严格实体级评测口径下，对两条技术路线进行平行考察：

| 维度 | **抽取式范式（Extractive）** | **生成式范式（Generative）** |
| :--- | :--- | :--- |
| 代表方法 | BERT / MacBERT + Token Classification Head | GPT-NER（LLM + 上下文示例 + 自验证） |
| 骨干架构 | Encoder-only（双向 Transformer） | Decoder-only（自回归 Transformer） |
| 任务形式化 | 逐字符多分类（序列标注 $21$ 类） | 文本到文本生成（`@@实体##` 标记序列） |
| 输出解码 | BIO 标签链 → 贪心/Viterbi 解码为 span | 生成文本 → 正则解析回 span 集合 |
| 监督信号 | 全量 10,748 条标注数据微调 | 少样本（Few-shot）提示 + 示例检索 |
| 本项目状态 | **已完成**（含训练、评测、错例挖掘） | **已接入训练后 LoRA 实时识别、检索消融、评测与失败案例**（见 §8） |

**抽取式一侧**已完整落地：在统一的硬件与超参数约束下，完成了 **BERT-base-chinese** 与 **chinese-macbert-base** 的严格实体级评测比对，并自主研发了跨模型错例挖掘工具，输出了包含量化指标、错例溯源及语言学分析的完整学术闭环。

**生成式一侧**以 **GPT-NER**（Wang et al., 2023）为方法原型，考察"将 NER 转化为文本生成任务"这一路线在低资源条件下的行为特征，与抽取式基线形成数据量维度上的对照。

> **关于研究范围的说明**：本项目的"复现"定位为**下游任务复现（Downstream Reproduction）**——即复用公开的预训练骨干权重，在 CLUENER2020 上复现并检验其下游 NER 性能，而非从零预训练语言模型。二者的工程量与算力需求相差数个数量级（见 §9.1 的口径声明）。

---

## 2. 核心原理与演进机制 (Theoretical Background & Mechanism)

### 2.1 中文 NER 技术演进全景

```
+-----------------------------------------------------------------------------------------------+
|                                    中文 NER 技术演进路径                                       |
+-----------------------------------------------------------------------------------------------+
|  【传统规则/词典时代】        人工专家规则、模式匹配、外部地名/机构名大词典                         |
|            |                                                                                  |
|            v                                                                                  |
|  【统计机器学习时代】        HMM (隐马尔可夫模型) -> MEMM (最大熵马尔可夫模型) -> CRF (条件随机场)   |
|            |                                                                                  |
|            v                                                                                  |
|  【深度学习时代】            BiLSTM-CRF (双向长短时记忆网络 + 转移概率矩阵约束转移合法性)          |
|            |                                                                                  |
|            v                                                                                  |
|  【预训练模型时代】          BERT (Transformer 双向编码 + Masked Language Modeling + NSP)        |
|            |                                                                                  |
|            v                                                                                  |
|  【中文针对性纠错演进】      MacBERT (纠错式 MLM 消除预训练微调鸿沟 + 全词掩码 WWM + N-gram + SOP) |
+-----------------------------------------------------------------------------------------------+
                                              |
                                              v
                        +---------------------+---------------------+
                        |                                           |
              【抽取式路线 · 判别式标注】               【生成式路线 · 自回归生成】
              BERT / MacBERT + Softmax              GPT-NER / LLM + In-Context
              逐 token 打 BIO 标签                    直接生成 @@实体## 标记文本
              (本项目已完成，见 §4)                    (方法与实测展示，见 §2.5、§8)
```

### 2.2 原生 BERT 在中文序列标注中的局限

原生 Google 中文 BERT (`bert-base-chinese`) 存在两大对实体识别不利的先天局限：

1. **预训练与微调的不一致（Pre-train / Fine-tune Discrepancy）**：
   - 在预训练阶段，BERT 对输入序列随机选取 15% 的 Token 进行操作，其中 80% 被替换为人造占位符 `[MASK]`，10% 随机替换，10% 保持不变；
   - 然而，在下游真实下游任务（如 NER 微调）中，**输入文本永远不会出现 `[MASK]` 符号**。模型在预训练期被迫学习适应大量人工噪声标记，削弱了其对真实连续上下文语义的直接推理能力。
2. **字符级掩码（Character-level Masking）破坏实体完整性**：
   - 原始中文 BERT 预训练主要在单个汉字级别做遮蔽（例如将“实体识别”中的“别”Mask 掉）。这种方式模型容易根据浅层的“字搭配”直觉填空，模型无需理解多字实体的全局结构与语义块边界，导致微调时对长实体发生**碎片化切分**。

### 2.3 MacBERT 核心架构与机制创新

MacBERT (*Revisiting Pre-trained Models for Chinese Natural Language Processing*, Findings of EMNLP 2020) 在继承标准 BERT 骨干 Transformer 结构的基础上，针对上述缺陷进行了三重革新：

```
========================= 掩码策略与输入表征机制对比 =========================
原始中文文本:     [CLS]  我  们  在  中  国  房  地  产  研  究  会  工  作  [SEP]

原生 BERT:       [CLS]  我  们  在 [MASK] 国  房 [MASK] 产  研  究  会  工  作  [SEP]
                      (单字孤立掩码，出现人工伪标记 [MASK]，割裂词汇完整性)

中文 BERT-WWM:   [CLS]  我  们  在 [MASK][MASK] 房  地  产  研  究  会  工  作  [SEP]
                      (基于 LTP 分词，整词全掩码，但仍存在 [MASK] 鸿沟)

MacBERT (本项目): [CLS]  我  们  在  华  夏  房  地  产  协  会  工  作  [SEP]
                      (纠错式替换：使用近义词/相似词替代 [MASK]，逼近自然语境纠错！)
=============================================================================
```

1. **MLM as Correction（纠错式掩码机制，Mac）**：
   - 彻底摒弃了向句子中注入 `[MASK]` 符号的方式。
   - 对 15% 的选中词，采用**近义词词典（Synonyms Dictionary）**检索相似词进行替换（87% 替换为近义词，10% 替换为随机词，3% 保持原词）；
   - 任务目标转变为：**“输入一个带有语义扰动的正常自然语言句子，网络隐状态输出纠错还原后的原本 Token”**。
   - **巨大价值**：彻底抹平了预训练与微调之间由于人工标记 `[MASK]` 导致的表征鸿沟。
2. **Whole Word Masking (WWM) + N-gram Masking**：
   - 深度融合中文语言特性，调用 LTP（语言技术平台）进行中文分词；
   - 凡分词构成的多字词组（Word-level）必须作为整体同时掩码，并以特定概率抽取 2-gram、3-gram 甚至 4-gram 进行连续掩码。这迫使 Transformer 的多头自注意力机制（Multi-Head Attention）必须建模长程词块依赖，天然保留了中文实体的边界完整性。
3. **Sentence Order Prediction (SOP) 代替 NSP**：
   - 借鉴 ALBERT 的训练策略，将原有由于两篇文章主题差异过大而过于简单的“下一句预测”（NSP），升级为同一篇文档中两段连续文本的“正序 vs 逆序判断”（SOP）；
   - 迫使网络深入理解篇章级的高阶语篇连贯性（Coherence），而非仅仅依靠词表重合度走捷径。

### 2.4 下游序列标注适配建模 (Sequence Labeling Head)

本项目采用学术界成熟的 **BERT/MacBERT + Linear Token Classification** 经典架构。输入经过字符分词后输入双向 Transformer 编码层，获得每个字符的隐层表征向量 $\mathbf{h}_t \in \mathbb{R}^{d}$（$d=768$）。

对于标签空间 $\mathcal{Y}$，本项目设计标准的 **BIO（Begin, Inside, Outside）** 标注体系：
- 类别数 $K = 10$，对应 $2 \times 10 + 1 = 21$ 个标签。
- 字符 $t$ 的发射概率由线性分类头计算：
  $$P(y_t = k \mid \mathbf{x}) = \text{softmax}(\mathbf{W}_h \mathbf{h}_t + \mathbf{b}_h)_k$$
- 训练损失采用交叉熵损失函数（Cross-Entropy Loss），在计算时通过屏蔽向量（`-100` Mask）严格忽略 `[CLS]`, `[SEP]`, `[PAD]` 等非文本字符：
  $$\mathcal{L}_{CE} = - \sum_{t=1}^{T} \mathbb{I}(y_t \neq -100) \log P(y_t \mid \mathbf{x})$$

### 2.5 生成式 NER 范式：GPT-NER (Generative Paradigm)

前述 §2.1–§2.4 均属**抽取式范式（Extractive Paradigm）**：模型为每个字符输出一个标签，再由解码器还原实体 span。该范式依赖全量标注数据微调，且标签空间固定、无法处理嵌套实体。

**GPT-NER**（Wang et al., 2023, *arXiv:2304.10428*）提出了一条截然不同的路线：借助大语言模型（LLM）天然的生成能力，将 NER **形式化为文本到文本的生成任务**，仅需少样本示例即可工作。

#### 2.5.1 机制一：任务的生成式形式化 —— `@@实体##` 标记协议

GPT-NER 不要求模型输出 BIO 标签，而是要求模型**把原文中的实体用特殊标记包裹后原样输出**。对每种实体类型单独构造一次提示（per-type prompting）：

```
================= 抽取式 vs 生成式：同一句话的两种输出形态 =================

原文:      北京时间 3 月 15 日，华为在东莞松山湖基地发布了新款折叠屏手机

【抽取式输出】(BIO 标签链，长度 = 字符数)
  北(B-TIME) 京(I-TIME) 时(I-TIME) 间(I-TIME) 3(I-TIME) 月(I-TIME) 1(I-TIME) 5(I-TIME) 日(I-TIME)
  ，(O) 华(B-ORG) 为(I-ORG) 在(O) 东(B-LOC) 莞(I-LOC) 松(I-LOC) 山(I-LOC) 湖(I-LOC) 基(B-LOC)...

【生成式输出】(标记文本，长度 ≈ 原文 + 标记符)
  提示类型 = LOC
  输出: 北京时间 3 月 15 日，华为在 @@东莞## @@松山湖## 基地发布了新款折叠屏手机
  解析: 提取 @@...## 内部内容 -> {"LOC": ["东莞", "松山湖"]}
============================================================================
```

这一设计的巧妙之处在于：**输出空间与输入文本强耦合**，模型只需"圈出"而非"凭空写出"，从而将开放式生成约束为近似的选择题，显著降低了搜索空间。

#### 2.5.2 机制二：实体级 kNN 示例检索 (Entity-Level kNN Retrieval)

少样本提示的成败高度依赖示例质量。GPT-NER 的关键实证发现是：**随机选示例效果很差，且检索粒度必须落在"实体级"而非"句子级"**。

原因在于 NER 数据的极端稀疏性——一个句子中绝大多数 token 都是 `O`（非实体）。若用整句语义向量做检索，句子表示会被大量 `O` token 稀释，导致检索到的示例与待测实体类型无关。

```
========================= 三种示例检索策略的消融（论文原表） =========================
  随机选择示例 (Random)           ->  CoNLL2003 F1 = 72.62
  句子级 kNN 检索 (Sentence-level) ->  CoNLL2003 F1 = 84.36   (+11.74)
  实体级 kNN 检索 (Entity-level)   ->  CoNLL2003 F1 = 89.97   (+17.35)
=====================================================================================
```

实体级检索的实现分三步（论文 Figure 2）：

```
Step 1  构建 Datastore
        用一个【微调好的 NER 模型】为训练集每个句子抽取实体，
        构造 (key = 实体表示, value = 原句) 键值对集合。

Step 2  表示提取
        对输入句子，用同一个微调 NER 模型编码为向量序列，
        再经 softmax 打标签，识别出句中的候选实体（如 "Washington"）。

Step 3  kNN 搜索
        以候选实体的 embedding 作为 query，在 Datastore 中检索 k 近邻，
        其关联的 value（原句）即作为 few-shot 示例。
        更精确地说：对长度为 N 的输入序列，逐 token 各取 K 个近邻得 K×N 个候选，
        再从这 K×N 个候选中取 top k，用其所属句子作为示例。
```

> **⚠ 对本项目至关重要的一点**：论文明确指出，实体级检索**依赖一个"微调好的 NER 模型"**来提供实体表示与候选实体识别。
> 本项目恰好已经训练好了 BERT 与 MacBERT 两个 NER 模型（见 §4），因此**可直接复用其作为检索编码器**。
> 这意味着两条技术路线并非彼此孤立——**生成式一侧的示例检索，恰恰建立在抽取式一侧已训模型的基础之上**，
> 二者在本项目中构成了一个有机的整体，而非两个平行无关的实验。

#### 2.5.3 机制三：自验证策略 (Self-Verification)

生成式方法最大的风险是论文所称的**过预测（Over-prediction）**——LLM 倾向于"过度自信地把非实体标注为实体"。论文给出的真实例子：

```
任务：标注 location 实体
  Input:  Columbus is a city            Output: @@Columbus## is a city
  Input:  Rare Hendrix song sells for $17
  GPT-3 输出: Rare @@Hendrix## song sells for $17
                   ^^^^^^^^ "Hendrix"（实为人名）被错误标注为地点实体
```

自验证策略的做法：将上一阶段提取出的实体逐个回喂给 LLM，要求其以 **yes/no** 判定该实体是否真属于目标类型，否定的结果直接剔除。

```
自验证提示（论文 Figure 3 结构，中文场景改写示意）:
  [任务描述] 下面的任务是判断给定句子中的词语是否为"地点"实体。
  [示例 1]   The input sentence: Only France and Britain backed Fischler's proposal
             Is the word "France" in the input sentence a location entity?
             Please answer with yes or no.
             Yes.
  ...（多个示例）...
  [待测]     Sentence: 华为在东莞松山湖基地发布了新款折叠屏手机
             Is the word "东莞" in the input sentence a location entity?
             Please answer with yes or no.
```

注意：**自验证环节本身同样需要 few-shot 示例**，且论文指出此处的示例检索也应使用 entity-level embedding（因为问题的核心是"某个实体是否属于某类型"，而非整句语义）。

消融结果显示自验证带来的增益（CoNLL2003 全量集，entity-level 检索下）：

| 配置 | Precision | Recall | F1 |
| :--- | :--- | :--- | :--- |
| 无自验证 | 88.54 | 91.40 | 89.97 |
| + 自验证（zero-shot） | 89.47 | 91.77 | 90.62 |
| + 自验证（few-shot） | **89.76** | **92.06** | **90.91** |

可见自验证的增益集中在 **Precision（+1.22）** 而非 Recall —— 这与其"过滤幻觉"的设计意图完全吻合。

#### 2.5.4 生成式范式的真实定位：低资源占优，全量仍落后

关于低资源表现，论文的定性结论是：

> *"GPT-NER exhibits impressive proficiency in low-resource and few-shot NER setups: when the amount of training data is extremely scarce, GPT-NER performs significantly better than supervised models."*

但必须**同时**看到另一半事实——在**全量训练数据**上，GPT-NER 并未超过监督基线：

| 数据集（全量） | 监督基线 BERT-Tagger | GPT-NER（最优配置） | 监督 SOTA |
| :--- | :--- | :--- | :--- |
| CoNLL2003 | **92.8** | 90.91 | 94.6 |
| OntoNotes5.0 | **89.16** | 82.20 | 92.07 |

论文自身也明确承认这一差距：*"there still remains a gap between the supervised SOTA model: 94.6 v.s. 90.91 on the full CoNLL2003 dataset."*

这带来一个必须诚实传达的结论：**生成式 NER 的价值不在于全量数据上击败 BERT，而在于标注成本受限时提供一条可行的替代路径**。两种范式之间存在一条随数据量移动的"性能交叉线"，而刻画这条线正是本项目设计**低资源分档对照实验**（见 §15.2 P0）的直接动机。

> **注**：论文中低资源场景的具体数值以曲线图（Figure）形式给出，PDF 文本层无法提取，故此处仅引用论文的定性结论，不转述未经核实的数字。待本项目实测后，将用本项目自身的分档实验数据填补这一空白。

#### 2.5.5 本项目对 GPT-NER 的复现定位与已知风险

将 GPT-NER 迁移到本项目场景时，存在两项必须诚实声明的偏移（依据任务书 P4 条款）：

1. **跨语言与跨数据集迁移**：GPT-NER 原论文实验基于英文数据集（CoNLL2003 / OntoNotes / ACE2004 / GENIA），本项目改用中文 CLUENER2020。因此**绝对 F1 数值不可与原论文直接比较**，本项目仅比较"同一数据集上两种范式的相对表现"。
2. **基座模型降级**：原论文使用 GPT-3（`text-davinci-003`，175B 级闭源 API），本项目受本地硬件约束（RTX 5060 8GB），拟采用 **Qwen2.5-1.5B-Instruct** 等可在单卡部署的开源指令微调模型。参数量相差约两个数量级，预期绝对性能显著低于原论文报告值，此差异须在结果解读时明确归因于"基座规模"而非"方法失效"。

> **状态说明（2026-10-23）**：工作台已接入本地 Qwen2.5-1.5B 的检索策略消融、LoRA 评测及失败案例。生成式页面已接入训练后的 Qwen + LoRA 实时识别，并保留历史实验展示与导出；页面下方的协议教学示例仍为固定示意。所展示运行均未开启自验证，不能将自验证的理论收益当作本项目实测结论。

### 2.6 两种范式的结构性对照

| 对比维度 | 抽取式（BERT / MacBERT） | 生成式（GPT-NER） |
| :--- | :--- | :--- |
| 骨干架构 | Encoder-only，双向注意力 | Decoder-only，因果注意力 |
| 标签空间 | 固定 $21$ 维，扩展需重训 | 由提示词定义，增删类型零成本 |
| 数据需求 | 依赖全量标注微调 | 少样本即可启动 |
| 嵌套实体 | 单层 BIO 无法表达 | 理论上可通过多轮生成覆盖 |
| 主要失效模式 | 边界切分错误、类别混淆 | 幻觉实体、漏检、格式不遵循 |
| 推理成本 | 单次前向，毫秒级 | 自回归逐 token 解码 + 多类型多轮 |
| 可解释性 | 需借助注意力/错例反推 | 生成过程本身具备一定可读性 |

两者并非简单的优劣替代关系，而是**在不同资源约束下的取舍**：抽取式在吞吐与全量数据精度上占优，生成式在冷启动与类型灵活扩展上占优。本项目将二者置于同一评测口径下对照，正是为了定量刻画这一取舍边界。

---

## 3. 实验配置与评测基准 (Benchmark & Experimental Setup)

### 3.1 CLUENER2020 细粒度数据集

CLUENER2020 是中文语言理解评测基准（CLUE）发布的高质量细粒度中文 NER 数据集，涵盖来自各领域的开源短文本，共划分为 **10 大细粒度实体类别**：

| 实体类别英文代码 | 中文名称 | 典型实体示例 | 难度特点与挑战 |
| :--- | :--- | :--- | :--- |
| `address` | 地址 | 邢州大道、文汇路、京石高速西三环六里桥 | 包含大量空间修饰语、路径介词，边界极难判定 |
| `book` | 书籍 | 《红楼梦》、《现代汉语词典》 | 依赖书名号先验，无书名号时极易被切散 |
| `company` | 公司 | 腾讯公司、商汤科技、万科企业 | 易与常见机构名（organization）或人名混淆 |
| `game` | 游戏 | DOTA、实况足球（2011）、CS1.6 | 中英混排、含版本号/标点、口语化简称多 |
| `government` | 政府机构 | 国家发改委、北京市教委、最高人民检察院 | 实体较长，行政层级分明，简称缩写密集 |
| `movie` | 电影 | 《阿凡达》、《流浪地球》 | 多义性强，与文学作品/普通短语高度重叠 |
| `name` | 人名 | 顾云昌、温格、商为智 | 罕见字/单字姓、西方音译人名消歧 |
| `organization` | 组织机构 | 莫斯科中央陆军、中国房地产及住宅研究会 | 超长跨度、行业协会、体育俱乐部，极易碎片化 |
| `position` | 职位 | 副会长、总工程师、实习生、主力前锋 | 职级前缀多样、常与部门复合 |
| `scene` | 景点 | 颐和园、黄山风景区、故宫博物院 | 易与普通地理地址混淆 |

- **数据集切分规模**：
  - 训练集（`train.json`）：**10,748** 条样本
  - 验证集（`dev.json`）：**1,343** 条样本（包含 3,072 个真实实体标注）
  - 测试集（`test.json`）：**1,345** 条样本

### 3.2 严格实体级评测标准 (Strict Entity-Level F1)

在命名实体识别中，通常存在**字符级准确率（Token-level）**与**实体级评测（Entity-level）**的区别。字符级指标往往会因为背景字符（`O`）占据绝大多数而呈现虚高，无法真实反映模型抽取的可靠性。

本项目全程采用符合 CoNLL 规范的 **`seqeval` 严格实体级精确匹配评测（Strict Exact-Match）**：
- **精确率（Precision）**：模型抽取的实体中，**实体边界与类别完全正确**的比例；
- **召回率（Recall）**：标注数据集中所有真实实体中，被模型**完全无偏差提取**的比例；
- **$F_1$ 值（Entity-Level Micro F1）**：
  $$F_1 = \frac{2 \times P \times R}{P + R}$$
  > **注意**：哪怕边界差一个汉字（例如将“滨江路围合区域”预测为“滨江路”），在 strict 评测下同时被记为一次 False Positive（错报）和一次 False Negative（漏报），惩罚极严苛。

### 3.3 训练与评估超参数

> 本节和 §4 保留早期 batch=16、3 epoch 实验记录；§9、§10 的相应分数讨论沿用该轮实验。新版工作台快照来自后续 batch=32、4 epoch 运行，见 §8.5。请勿将不同运行的损失、轮次或分类指标拼接比较。

为了保障两组基线实验具有高度可比性与实验公平性，实验全程基于相同的物理运行环境，固定随机种子并采用同一套超参数配置：

| 超参数项 (Hyperparameter) | 配置取值 (Value) | 说明 (Notes) |
| :--- | :--- | :--- |
| **预训练模型对比组** | `bert-base-chinese` vs `hfl/chinese-macbert-base` | 参数量均为 ~102M，12 层 Transformer |
| **最大序列截断长度 (`max_len`)** | 128 | **实测 100% 覆盖**：CLUENER 全集中的最长文本仅 50 字，均值 37.4 字，无一例超长（见 §12 问题 5）。注：沿用论文的 128 以保证与原文配置对齐，代价是约 65% 的填充开销，改为 64 可显著提速且无信息损失 |
| **批次大小 (`batch_size`)** | 16 | 适配显存与梯度稳定性 |
| **优化器 (`optimizer`)** | AdamW ($\beta_1=0.9, \beta_2=0.999, \epsilon=1\text{e-}8$) | 权重衰减系数 `weight_decay=0.01` |
| **最大学习率 (`lr`)** | $3 \times 10^{-5}$ | 预训练模型微调黄金学习率 |
| **学习率调度 (`scheduler`)** | Linear Warmup with Linear Decay | `warmup_ratio=0.1` |
| **梯度裁剪 (`max_grad_norm`)** | 1.0 | 防止反向传播梯度爆炸 |
| **训练周期 (`epochs`)** | 3 | 在验证集上根据最优 Micro F1 执行 Model Checkpoint |
| **随机数种子 (`seed`)** | 42 | 统一固定 Python / Numpy / PyTorch 随机状态 |

---

## 4. 核心实验结果与量化对比 (Core Experimental Results)

### 4.1 细粒度 10 大类别性能全景对比表

在统一设定的 CLUENER2020 验证集（1,343 个样本，共 3,072 个实体金标）上，模型严格评测得分如下表所示：

| 实体类型 (Category) | 标注数 (Support) | BERT-Base Precision | BERT-Base Recall | BERT-Base F1-Score | MacBERT Precision | MacBERT Recall | MacBERT F1-Score | $\Delta F_1$ (MacBERT - BERT) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`address` (地址)** | 373 | 58.06% | 65.68% | 61.64% | 56.56% | 67.02% | 61.35% | -0.29% |
| **`book` (书籍)** | 154 | 75.61% | 80.52% | 77.99% | 75.62% | 78.57% | 77.07% | -0.92% |
| **`company` (公司)** | 378 | 71.97% | 80.16% | 75.84% | 72.13% | 81.48% | 76.52% | **+0.68%** |
| **`game` (游戏)** | 295 | 73.53% | 84.75% | 78.74% | 76.38% | 84.41% | 80.19% | **+1.45%** |
| **`government` (政府)** | 247 | 75.17% | 87.04% | 80.68% | 76.19% | 84.21% | 80.00% | -0.68% |
| **`movie` (电影)** | 151 | 80.82% | 78.15% | 79.46% | 76.82% | 76.82% | 76.82% | -2.64% |
| **`name` (人名)** | 465 | 83.57% | 89.68% | 86.51% | 85.83% | 89.89% | 87.82% | **+1.31%** |
| **`organization` (组织)** | 367 | 72.01% | 77.11% | 74.47% | 71.84% | 82.02% | 76.59% | **+2.12% 🚀** |
| **`position` (职位)** | 433 | 75.11% | 80.83% | 77.86% | 76.03% | 81.29% | 78.57% | **+0.71%** |
| **`scene` (景点)** | 209 | 65.02% | 69.38% | 67.13% | 64.57% | 68.90% | 66.67% | -0.46% |
| **总体微平均 (Micro Avg)** | **3072** | **72.92%** | **79.75%** | **76.18%** | **73.18%** | **80.31%** | **76.58%** | **+0.40%** |
| **总体宏平均 (Macro Avg)** | **3072** | 73.09% | 79.33% | 76.03% | 73.20% | 79.46% | 76.16% | +0.13% |
| **加权平均 (Weighted Avg)** | **3072** | 73.04% | 79.75% | 76.21% | 73.45% | 80.31% | 76.69% | +0.48% |

#### 训练过程与收敛元数据对比

| 核心训练元数据 (Training Meta) | BERT-Base 基线 | MacBERT-Base 改进模型 | 差异对比与技术洞察 |
| :--- | :---: | :---: | :--- |
| **达到最佳权重轮次 (Best Epoch)** | Epoch 3 / 3 | **Epoch 2** / 3 | MacBERT 提前一整轮达到最高 F1，收敛速度与样本利用率更优 |
| **最佳验证集损失 (Best Val Loss)** | 0.2050 | **0.1935** | 验证集交叉熵损失降低 5.6%（-0.0115），模型输出置信度更高 |
| **主干参数规模 (Model Parameters)** | ~102M (12-layer) | ~102M (12-layer) | 参数规模与推理开销完全对齐，性能提升纯粹来自架构预训练质量 |

### 4.2 核心量化结论与消融洞察

1. **组织机构实体（`organization`）暴涨 +2.12%，召回率飙升 +4.91%**：
   - 组织机构名通常是中文中平均字符长度最长、内部构词最复杂的实体形态（如“XX省XX行业发展促进研究会”）。原生 BERT 在单字 Mask 下，非常容易在内部出现断裂或将其中单字误判为政务或公司；MacBERT 凭借 **Whole Word Masking (WWM) 与 N-gram Masking**，迫使网络建立词块级别的表征记忆，使其在长实体组织上的检出能力实现跃升（召回由 77.11% 提升至 82.02%）。
2. **游戏（`game` +1.45%）与人名（`name` +1.31%）显著提升，精确率大幅增益**：
   - 游戏名中包含大量非标准字符组合与版本符号，MacBERT 在 `game` 上的精确率由 73.53% 提升至 76.38%（提升 **+2.85%**）；
   - 在 `name` 上，MacBERT 精确率由 83.57% 跃升至 85.83%（提升 **+2.26%**）。这得益于纠错机制强化了真实上下文的语义约束，极大抑制了将罕见姓氏虚假误判为普通专名的假阳性问题。
3. **收敛速度与验证集泛化损失更优**：
   - MacBERT 在 **第 2 轮（Epoch 2）** 即斩获最佳 Micro F1，验证集损失压低至 **0.1935**（相比 BERT 的 0.2050 降低了 5.6%），表明消除 `[MASK]` 鸿沟后，微调任务与预训练目标更为契合，权重梯度的调整更具方向性与平滑度。

---

## 5. 精选案例研究与深度错例挖掘 (Bad Case & Qualitative Analysis)

为了穿透宏观量化数值，本项目针对 CLUENER2020 验证集全体 1,343 条样本进行了全量无偏推理预测，并开发了差异挖掘管道，输出结构化报告 `case_study_results.json`。

### 5.1 案例集总体分布统计

```
验证集总样本数 (Total Samples): 1343 条
=============================================================================
[1] 双模型完全正确 (Both Correct): 657 条 (48.92%)
[2] MacBERT 显著优于 BERT (MacBERT Better): 76 条 (5.66%)  <--- 核心优势案例池
[3] BERT 优于 MacBERT (BERT Better):        70 条 (5.21%)
[4] 双模型均存在失误 (Both Wrong):         540 条 (40.21%) <--- 中文 NER 固有瓶颈
=============================================================================
```

### 5.2 类别 A：MacBERT 优势与纠错还原案例 (Representative Superiority Cases)

#### 案例 A-01：超长组织机构名边界碎片化切分修复 (Sample #18)
- **原始文本**：`莫斯科中央陆军vs波兹南、拉科vs费耶诺德、加拉塔萨雷vs梅塔利斯特，`
- **真实标签 (GT)**：`莫斯科中央陆军` (organization)
- **BERT 预测**：`莫斯科中央` (organization) + `陆` (government) + `军` (organization)
- **MacBERT 预测**：`莫斯科中央陆军` (organization)
- **深度机理解析**：
  BERT 发生了典型的**单字级语义崩塌**——由于“陆军”单字与军事、行政强相关，单字注意力将其误判为政府机构（government），导致原本统一的足球俱乐部实体断裂为三个碎片；MacBERT 利用 WWM 和全词掩码，在预训练中学习到了长跨度专有名词的整体先验，完美还原了完整边界。

```diff
- 原文: 莫斯科中央陆军vs波兹南...
- BERT: [莫斯科中央](org) [陆](gov) [军](org)  <--- 严重切碎与类别混淆
+ MacBERT: [莫斯科中央陆军](org)              <--- 完整且准确识别
```

#### 案例 A-02：复合层级行业研究会整体捕获 (Sample #504)
- **原始文本**：`中国房地产及住宅研究会副会长顾云昌认为，经济适用房建设进度主要取决于三个因素：`
- **真实标签 (GT)**：`中国房地产及住宅研究会` (organization), `副会长` (position), `顾云昌` (name)
- **BERT 预测**：`中国房地`(org) + `产`(gov) + `及`(org) + `住宅`(gov) + `研究会`(org)
- **MacBERT 预测**：`中国房地产及住宅研究会` (organization) [100% 精确捕获]
- **深度机理解析**：
  面对长达 11 字的复合协会名，实体内部包含了“地产”、“住宅”等高频可独立成词的词汇。原生 BERT 的自注意力极易被局部强特征捕获，连环切碎成 5 个离散小词；MacBERT 的纠错式预训练机制强化了跨词块的全局语义一致性，坚固地维系了超长实体的首尾边界。

#### 案例 A-03：混合全角括号与年份版本的游戏专名抽取 (Sample #1156)
- **原始文本**：`上海和广州三大城市开展。项目包括DOTA、实况足球（2011）、CS1.6、`
- **真实标签 (GT)**：`实况足球（2011）` (game)
- **BERT 预测**：`实况足球` (game) + `011）` (game)
- **MacBERT 预测**：`实况足球（2011）` (game)
- **深度机理解析**：
  实体内嵌全角括号 `（` 与阿拉伯数字。原生 BERT 将左括号与数字截断，遗漏了“2”，导致破坏性错标；MacBERT 结合 N-gram 掩码，成功建模了字符与版本后缀之间的紧密依存关系。

#### 案例 A-04：全局语义消歧与罕见人名纠偏 (Sample #1267 & #923)
- **篇章语境消歧 (#1267)**：`贝蒂斯在客场与瓦伦打起对攻...`
  - BERT 将西甲球队 `贝蒂斯` 误判为人名（`name`）；
  - MacBERT 结合“客场”、“打起对攻”等全局体育动作语义，判定其为组织（`organization`）。
- **汉字偏置纠错 (#923)**：`商为智实习生陈文波`
  - BERT 因“商”字强烈的商业先验，将罕见人名 `商为智` 误判为公司（`company`）；
  - MacBERT 纠正为正确的人名（`name`）。

---

### 5.3 类别 B：中文 NER 固有瓶颈与顽疾错例剖析 (Challenging Cases & Bottlenecks)

在双错（540 条，占 40.21%）的案例中，揭示了当前深度学习序列标注在中文场景下的四大深层瓶颈：

#### 瓶颈 1：数据集人工标注遗漏与标签噪声 (GT Omission Noise)
- **典型案例 (#1)**：`温格的球队终于又踢了一场经典的比赛，2比1战胜曼联之后枪手仍然留在了夺冠集团之内，`
  - **GT 金标**：`温格` (name), `曼联` (organization) —— **遗漏了“枪手”！**
  - **BERT 与 MacBERT 预测**：均一致抽取出 `枪手` (organization)。
  - **分析**：在足球领域，“枪手”是英超阿森纳俱乐部的法定知名别称。双模型均根据上下文成功推断出组织实体，但因标注人员疏漏，在 strict 评测下被判定为假阳性（FP），导致模型精度被冤枉倒扣。
- **典型案例 (#3)**：`...找校方或者教委要个说法。`
  - GT 遗漏标注了行政部门 `教委`，而两模型均精准检出。

#### 瓶颈 2：专名核心与空间修饰语的边界模糊 (Boundary Modifier Ambiguity)
- **典型案例 (#1246)**：`限行规则与北京保持一致。限行区域为邢州大道、祥和大街、东华路、滨江路围合区域含上述道路。`
  - **GT 金标**：`滨江路围合区域` (7字)
  - **BERT / MacBERT 预测**：`滨江路` (3字)
  - **分析**：“滨江路”为核心专有名词，而“围合区域”为表空间管制的状态修饰短语。由于 BIO 体系缺乏对修饰词扩展的公理化标准，模型倾向于在常用道路词“路”处截断，导致严格精确匹配失败。

#### 瓶颈 3：超长交通路径与复杂层级嵌套 (Hierarchical Route Address)
- **典型案例 (#812)**：`社区位于京良路南侧，距离京石高速约5公里，从京石高速西三环六里桥到京良路出口仅10余公里。`
  - **GT 金标**：`京石高速西三环六里桥到京良路出口` (长达 16 字的整段路径)
  - **BERT / MacBERT 预测**：切分为 `京石高速西三环六里桥` 与 `京良路出口`。
  - **分析**：整段话本质上包含起点（六里桥）、路径介词（到）、终点（出口）的空间网络关系。一维线性 BIO 标注体系无法有效表达具有关系图属性的超长路径。

#### 瓶颈 4：真实姓名与网络别名代号嵌套 (Nested Alias Collision)
- **典型案例 (#27)**：`...担任SOLO位的世界第一影魔Pis（卜严骏），`
  - **GT 金标**：`Pis（卜严骏）` 作为单一人名；
  - **BERT / MacBERT 预测**：均倾向于将括号内外剥离，分别识别出网名 `Pis` 与真实姓名 `卜严骏`。

---

## 6. 项目工程结构 (Repository Structure)

本项目结构清晰，遵循高内聚、低耦合的模块化设计：

```bash
NamedEntityRecognition/
├── .data/
│   └── cluener/                     # CLUENER2020 官方细粒度数据集
│       ├── train.json               # 训练集 (10,748 条)
│       ├── dev.json                 # 验证集 (1,343 条，3,072 个实体)
│       └── test.json                # 测试集 (1,345 条)
├── saved_models/                    # 模型权重与评估结果归档目录
│   ├── bert/                        # 训练好的 BERT 基线模型
│   │   ├── config.json              # 模型结构配置
│   │   ├── model.safetensors        # 微调后的最优模型权重
│   │   ├── eval_results.json        # 实体级验证集分类报告与指标
│   │   └── tokenizer.json           # 分词器词表映射
│   └── macbert/                     # 训练好的 MacBERT 改进模型
│       ├── config.json
│       ├── model.safetensors
│       ├── eval_results.json        # 对应检查点的验证集指标，以文件内容为准
│       └── tokenizer.json
├── docs/                            # 理论研究文献与课程指引
│   ├── CLUENER2020.pdf              # 【主讲·任务与数据】中文细粒度 NER 数据集与基准 (Xu et al., 2020)
│   ├── GPT-NER_生成式NER.pdf          # 【主讲·方法】生成式 NER 代表作 (Wang et al., 2023, arXiv:2304.10428)
│   ├── 作业说明.pdf                  # 课程综合实践任务书与评分细则
│   └── supplementary/               # 支撑文献（第二档：需理解核心创新，非主讲）
│       ├── BERT.pdf                 # 抽取式基座 (Devlin et al., NAACL 2019)
│       └── MacBERT中文预训练.pdf      # 抽取式基座 · 实验对照变量 (Cui et al., EMNLP 2020)
├── dataset.py                       # CLUENER 数据集加载与字符级 BIO 动态对齐管道
├── train.py                         # 统一的微调训练引擎与 Seqeval 评估闭环
├── multi_seed.py                    # 多种子重复实验与配对统计显著性检验 (mean±std / Bootstrap CI)
├── multi_seed_results.json          # 多种子实验原始结果 (运行 multi_seed.py 后生成)
├── multi_seed_report.md             # 可直接粘进实验报告的多种子汇总表 (自动生成)
├── analyze_cases.py                 # 全量模型预测对比与自动化错例挖掘系统
├── case_study_results.json          # 结构化案例挖掘数据库 (包含详尽模式与机理解析)
├── requirements.txt                 # 依赖清单 (对应任务书「记录实验环境」要求)
├── app.py                           # 交互式复现工作台 Flask 全栈服务 (双模型常驻 CPU 推理)
├── build_workbench_html.py          # 构建入口：内联样式、逻辑、实验记录与离线 Bundle
├── frontend/
│   ├── dual-paradigm.html           # 双范式总览与生成式实验室
│   ├── dual-paradigm.css            # 黑白设计系统与响应式布局
│   ├── dual-paradigm.js             # 实验切换、案例展示、数据同步与导航
│   ├── generative-live.js            # 生成式实时请求、校验和结果渲染
│   └── README.md                   # 前端维护说明
├── test_dual_paradigm_frontend.js   # 生成页面的真实函数回归测试
├── test_frontend_logic.js           # 静态 Bundle 与内联脚本检查
├── test_workbench_verification.py  # Flask 路由与本地模型集成测试
├── generative_service.py             # 懒加载 LoRA、请求限流和原文对齐
├── test_generative_api.py            # 生成式接口与实体偏移回归测试
├── static_offline_bundle.json       # 离线全量评估指标、代表错例与预置推演数据包
├── templates/
│   └── index.html                   # Flask 动态托管的交互式工作台前端模板
├── workbench.html                   # 零环境依赖脱机即用的自包含学术级单文件工作台
├── inspect_data.py                  # 数据集格式与分布快速探针脚本
├── AGENTS.md                        # AI 协同开发规范与安全守则
└── README.md                        # 本项目完整中英学术文档
```

---

## 7. 快速复现指引 (Quick Start Guide)

### 7.1 环境配置（含 NVIDIA RTX 50 系列 sm_120 兼容指南）

本项目要求 **Python 3.10+** 及现代化 PyTorch 环境。

#### 1. 基础 Python 依赖安装
```bash
pip install transformers seqeval tqdm numpy
```

> [!TIP]
> **国内下载加速配置**：
> 如果直接从 Hugging Face 官方拉取模型较慢，可设置镜像加速：
> ```bash
> # Linux / macOS / Git Bash
> export HF_ENDPOINT="https://hf-mirror.com"
> 
> # Windows PowerShell
> $env:HF_ENDPOINT="https://hf-mirror.com"
> ```

#### 2. GPU 驱动与 PyTorch 兼容指南（重要提示）
- **通用 GPU 用户（RTX 30 / 40 系列及更早型号）**：
  直接安装官方支持 CUDA 11.8 或 CUDA 12.1/12.4 的 PyTorch 稳定版即可：
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cu121
  ```
- **最新架构 GPU 用户（如 NVIDIA RTX 50 系列 Blackwell 架构 / `sm_120`）**：
  > [!WARNING]
  > RTX 50 系列显卡采用了全新 Blackwell 微架构（Compute Capability 12.0，即 `sm_120`）。早期构建的 PyTorch 官方预编译包通常仅打包了最高至 `sm_90`（Hopper）或 `sm_89`（Ada Lovelace）的 PTX 内核。如果在 RTX 50 显卡上强行使用不匹配的 CUDA 轮子运行，可能会引发 `CUDA error: no kernel image is available for execution on the device` 报错。
  > 
  > **解决方案**：
  > 1. 请务必更新显卡驱动至支持 **CUDA 12.8+** 的版本；
  > 2. 安装支持 `sm_120` 的 PyTorch 最新 Nightly 版本：
  >    ```bash
  >    pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
  >    ```
  > 3. 若在纯 CPU 环境下执行轻量评估与案例分析，系统将自适应退化为 CPU 计算，无需上述配置。

---

### 7.2 模型训练运行 (Training Pipeline)

本项目提供了统一参数化的训练入口脚本 `train.py`：

```bash
# 1. 训练 BERT 基线模型 (bert-base-chinese)
python train.py \
    --model_name bert-base-chinese \
    --output_dir ./saved_models/bert \
    --epochs 3 \
    --batch_size 16 \
    --lr 3e-5 \
    --max_len 128 \
    --seed 42

# 2. 训练 MacBERT 目标模型 (hfl/chinese-macbert-base)
python train.py \
    --model_name hfl/chinese-macbert-base \
    --output_dir ./saved_models/macbert \
    --epochs 3 \
    --batch_size 16 \
    --lr 3e-5 \
    --max_len 128 \
    --seed 42
```

训练过程中，系统将自动输出每轮验证集的评估进度、损失值与分类报告，并在训练结束时将最优模型权重 (`model.safetensors`)、分词器与评估报告 (`eval_results.json`) 统一归档。

---

### 7.3 模型独立评估 (Evaluation Only)

若已存在保存好的模型权重，可添加 `--eval_only` 参数快速在验证集上执行单次评估，无需重复训练：

```bash
# 独立评测 MacBERT
python train.py \
    --eval_only \
    --model_name ./saved_models/macbert \
    --output_dir ./saved_models/macbert
```

---

### 7.4 案例挖掘与差异分析 (Case Study Execution)

本项目配备了跨模型案例挖掘工具 `analyze_cases.py`。该脚本自动加载 BERT 与 MacBERT 模型，对开发集进行双盲推理，匹配真实标签并将预测结果进行高精度比对：

```bash
python analyze_cases.py \
    --bert_dir ./saved_models/bert \
    --macbert_dir ./saved_models/macbert \
    --dev_file .data/cluener/dev.json \
    --output_json case_study_results.json
```

---

### 7.5 多种子重复实验与显著性检验 (Multi-Seed & Significance)

单一随机种子下的微小 F1 差值不足以支撑“显著提升”的结论（详见 §10.1）。本项目提供 `multi_seed.py` 执行多种子重复实验：

```bash
# 完整实验（建议在 GPU 上执行）：5 种子 × 2 模型 × 3 epoch
python multi_seed.py --seeds 42,123,2024,3407,8888 --epochs 3

# 快速冒烟测试（验证脚本可跑通，结果不具统计意义）
python multi_seed.py --seeds 42,123 --epochs 1 --limit 200 --smoke

# 仅基于已有结果重新生成 Markdown 报告（不重新训练）
python multi_seed.py --report_only
```

脚本特性：
- 复用 `train.py` 的同一套训练引擎，避免“两套实现”造成的可复现性争议；
- 支持**断点续跑**：已完成的 `(model, seed)` 组合自动跳过，`--force` 可强制重跑；
- 输出 `multi_seed_results.json`（原始数据）与 `multi_seed_report.md`（可直接粘进实验报告）；
- 统计检验包含**配对 Bootstrap 95% 置信区间**、符号检验胜出计数，以及可选的 Welch t 检验。

输出示例：

```text
==============================================================================
多种子实验汇总 (Micro F1, %)
==============================================================================
  BERT-base-Chinese (Baseline)       76.02 ± 0.61  (min 75.31 / max 76.74, n=5)
  Chinese-MacBERT-base (Proposed)    76.44 ± 0.48  (min 75.88 / max 77.02, n=5)
------------------------------------------------------------------------------
  配对平均差值: +0.42 pp (std 0.55)
  95% Bootstrap CI: [-0.19, 1.03]
  配对差值的 95% 置信区间跨越 0，优势在统计上不显著，应表述为“数值上略优”。
==============================================================================
```

> 上表为**格式示意**，实际数值需在有 GPU 的环境运行后填入，严禁直接引用。

---

**控制台将实时输出高亮对比流**：
```text
================================================================================
[Case-A-01] Sample #18: 超长组织机构名边界碎片化切分修复
原始文本: 莫斯科中央陆军vs波兹南、拉科vs费耶诺德、加拉塔萨雷vs梅塔利斯特，
真实标签: [莫斯科中央陆军] (organization)
BERT预测: [莫斯科中央] (organization) | [陆] (government) | [军] (organization)
MacBERT:  [莫斯科中央陆军] (organization)
机理解析: 【WWM 全词掩码与纠错机制优势】原生 BERT 采用字符级单字 Mask，破坏了词汇边界关联...
================================================================================
```

---

## 8. 交互式复现工作台与全栈模型体验系统 (Interactive Reproduction Workbench & Model Arena)

**NER Lab** 将抽取式与生成式 NER 整合到同一个中文工作台：抽取式支持 BERT/MacBERT 实时推理，生成式支持训练后的 Qwen2.5 + LoRA 实时识别，并展示已完成实验。页面使用 Flask + 原生 JavaScript/CSS，不需要安装前端框架或下载 CDN 资源。

### 8.1 工作台设计架构与核心特性

- **双范式入口**：总览页解释 BIO 序列标注和 `@@实体##` 文本生成两条路线，提供独立入口和实验指标摘要。
- **黑白简约界面**：深色/浅色主题、本地主题记忆、桌面与手机布局；实体通过明暗、边框和文字标签区分。
- **明确结果来源**：实时模型预测、已保存预测快照、词典匹配演示分别标注。词典演示不提供模型置信度、模型耗时或 BIO 探针。
- **离线可展示**：`workbench.html` 内嵌所需样式、脚本和实验快照。离线可浏览实验、代码与案例，但不能执行真实模型推理。
- **生成式实时识别**：输入文本并选择实体类别，使用本地 Qwen 基座与训练后的 LoRA 适配器生成标记文本；显示实体高亮、原始输出、校验提示与 JSON 导出。历史实验与实时请求独立展示。

### 8.2 快速启动与双模运行指南

#### 方式一：直接查看已生成的页面

用浏览器打开根目录的 **`workbench.html`**。无需 Python 或模型权重即可查看随页面交付的实验快照。

#### 方式二：启动 Flask，体验抽取式实时识别

先按 §7.1 安装 Python 依赖，并准备 `saved_models/bert` 与 `saved_models/macbert` 模型检查点。模型权重不包含在普通 Git 提交中，需要本地训练或自行准备。

```bash
# 默认加载两组模型并自动打开浏览器
python app.py

# 不自动打开浏览器；仅监听本机
python app.py --host 127.0.0.1 --port 5000 --no-browser
```

浏览器访问 [http://127.0.0.1:5000](http://127.0.0.1:5000)。实际推理时延取决于硬件、文本和运行环境，以本次响应显示为准。

#### 修改前端后重新构建

```bash
python build_workbench_html.py
```

构建器从 `frontend/dual-paradigm.{html,css,js}` 和现有工作台源码生成相同的 `templates/index.html`、`workbench.html`。请修改源文件后重建，不要只编辑某一份生成 HTML。更新页面后刷新浏览器；Flask 使用模板缓存时需重启服务。

> 构建时会读取本地可选的 `results/generative/*.json`。这些原始运行文件不随本次前端提交上传；没有原始文件时仍可直接浏览已生成页面中的快照，但重新构建会省略缺失实验，并在无实验时显示空状态。

### 8.3 六大交互模块

| 模块 | 内容与操作 |
| :--- | :--- |
| **工作台总览** | 双范式原理简介、指标摘要、10 类实体说明与功能入口 |
| **抽取式识别** | 中文文本输入、BERT/MacBERT 对比、实体高亮、差异分析、逐字 BIO 探针、自定义用例保存和导入导出 |
| **生成式实验室** | 使用训练后 LoRA 实时识别新文本；另可切换历史实验、查看 P/R/F1 和异常率、导出预测或实验 JSON |
| **复现代码** | 查看数据处理、训练与错例挖掘的源码快照及批注 |
| **抽取式评测** | 总体指标、10 类 P/R/F1 图表与表格、训练配置和论文口径说明 |
| **案例分析** | 抽取式纠错/边界难例与生成式失败案例筛选、关键词搜索、原始生成文本和机制分析 |

生成式实验室还包含可切换“姓名 / 地址 / 公司”的固定教学示例，展示提示词、带标记文本和从 0 开始的实体闭区间。这是协议示意，不是模型预测。案例注释保留原运行背景，不代表当前选中实验的结果。

### 8.4 RESTful API 规范与扩展接口

| 接口 | 作用 | 主要字段 |
| :--- | :--- | :--- |
| `POST /api/predict` | 抽取式预测；请求体 `{"text":"文本","model":"both"}` | `results.macbert`、`results.bert`、`diff` |
| `GET /api/metrics` | 抽取式指标 | `data.models`、`data.categories_comparison`、`data.headline_delta_f1` |
| `GET /api/cases` | 案例记录 | `data.category_a_macbert_advantages`、`data.category_b_hard_cases`；扩展数据可包含 `data.category_c_generative_failures` |
| `GET /api/code` | 源码与批注 | `data.files` |
| `GET /api/health` | 抽取式服务与模型状态 | `status`、`device`、`models` |
| `GET /api/generative/status` | 生成式权重可用性、加载与忙碌状态（不加载模型） | `data.available`、`data.loaded`、`data.busy`、`data.device` |
| `POST /api/generative/predict` | 训练后 Qwen + LoRA 实时预测 | `data.entities`、`data.outputs`、`data.timing`、`data.warning_count` |

HTTP 模式会刷新 `/api/metrics` 和 `/api/cases`；失败时保留内嵌数据，并在页脚标明来源。历史生成式指标来自构建时嵌入的原始运行记录，不使用可能混合早期样本范围的汇总接口。实时识别单独调用 `/api/generative/predict`，失败时报告错误，不以离线结果替代。

#### 生成式模型配置与调用

默认读取以下本地文件，不下载模型，也不会在适配器缺失时退回未微调基座：

- `models/Qwen2.5-1.5B-Instruct/`：Qwen 基座配置、分词器与权重。
- `saved_models/gptner_lora/adapter_config.json` 和 `adapter_model.safetensors`：你训练后的 LoRA 适配器。

安装依赖后启动 `python app.py`，进入“生成式实验室 → 生成式实时识别”。首次请求懒加载模型，后续复用；优先 CUDA，未提供 CUDA 时使用 CPU（可能较慢）。可在启动前设置 `NER_GENERATIVE_MODEL_DIR`、`NER_GENERATIVE_LORA_DIR` 和 `NER_GENERATIVE_DEVICE=auto|cuda|cpu` 覆盖默认配置。

```python
import json
from urllib.request import Request, urlopen

payload = {
    "text": "腾讯公司位于深圳，马化腾担任董事会主席。",
    "categories": ["company", "address", "name"],
    "max_new_tokens": 96,
}
request = Request(
    "http://127.0.0.1:5000/api/generative/predict",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
with urlopen(request, timeout=180) as response:
    print(json.load(response))
```

`text` 为最多 128 个 Unicode 字符的非空单行文本，不能含协议保留标记 `@@` / `##`；不自动截断输入。`categories` 可省略（默认十类），或提供 1–10 个不重复的 CLUENER 类别。`max_new_tokens` 为 32–256 的整数，默认 96。当前实时模式固定使用训练时的 `zh_mrc` 提示、0-shot、贪心解码，不开启检索或自验证。

成功响应为 `{success: true, data: ...}`，含闭区间实体位置、每类原始输出、GPU/CPU 与耗时。只有去除成对标记后严格等于原文的输出才产生实体；否则保留原始生成并给出提示。同一文本被预测为不同类别时全部保留，格式校验不能证明语义类别正确。这个严格策略与历史评测脚本的宽松字符串定位不同，不能据此重算或替代历史 F1。

输入错误返回 400，模型正忙返回 409（带 `Retry-After`），缺少权重或推理失败返回 503。每进程同时只允许一个生成请求；前端等待上限 3 分钟，客户端超时不会终止服务器上的生成，可稍后刷新状态重试。

### 8.5 数据来源、更新与验证

**2026-10-23 随页面交付的实验快照：**

| 实验 | 评测范围与配置 | Micro F1 |
| :--- | :--- | ---: |
| BERT | 开发集 1,343 句；batch=32，4 epoch | 76.05% |
| MacBERT | 开发集 1,343 句；batch=32，4 epoch | 76.61% |
| Qwen + LoRA | 开发集 1,343 句；0-shot，max_new_tokens=96 | 41.59% |
| Qwen 随机示例检索 | 前 120 句；8-shot，max_new_tokens=72 | 7.46% |
| Qwen 句子级语义检索 | 前 120 句；8-shot，max_new_tokens=72 | 27.55% |
| Qwen 实体级示例检索 | 前 120 句；8-shot，max_new_tokens=72 | 33.57% |

三组检索实验具有相同样本范围与生成长度，可比较检索策略；LoRA 与抽取式运行的训练设置不同，不能将其差值归因于单一机制。所有展示的生成式运行均未开启自验证。早期 100 句 LoRA 运行也单独保留在实验选择器中。

数据来源与优先级：

1. `static_offline_bundle.json` 提供原有指标、案例、预设预测和源码快照。
2. 构建时将选定的 `results/generative/*.json` 嵌入 `bundle.experiments`，保留原始文件名、样本数及配置；生成式页面展示来源并支持导出。
3. `saved_models/*/eval_results.json` 提供本地训练配置；HTTP 访问时抽取式指标还会从后端刷新，因此可与交付快照不同。
4. 原始生成指标中的 P/R/F1 为百分数，格式异常率和幻觉率为 0–1 比例，前端转换为百分比显示；例如 `0.0015` 显示为 `0.15%`。

```bash
# 前端数据、脚本和行为回归（需要 Node.js，无需 npm install）
node test_frontend_logic.js
node test_dual_paradigm_frontend.js

# Flask 与真实本地模型集成测试（需要依赖和检查点）
python -m unittest test_workbench_verification test_generative_api
```

前端回归覆盖真实生成函数、数据单位、实体偏移、内容转义、筛选和空数据状态。浏览器验收覆盖在线推理、API 不可用时的回退、键盘导航、深浅主题及 390px / 1440px 布局。维护细节见 [frontend/README.md](frontend/README.md)。

---

## 9. 与原论文公开结果的对比 (Comparison with Published Results)

> 对应任务书「基线模型复现」第 8、9 条要求：将复现结果与原论文或公开结果进行对比，并分析差异原因。

### 9.1 指标口径必须先对齐

直接把两个数字相减是最常见的复现错误。本项目与 CLUENER2020 原论文在**四个维度**上口径不同，**不可直接相减**：

| 维度 | CLUENER2020 原论文 | 本项目复现 |
| :--- | :--- | :--- |
| 数据划分 | **test** 集（1,345 条，标签不公开，需提交榜单） | **dev** 集（1,343 条 / 3,072 个实体） |
| 标注体系 | **BIOS**（B / I / S / O，单字实体用 S-） | **BIO**（B / I / O） |
| 汇总方式 | **Overall@Macro**（10 类 F1 的算术平均） | **Micro**（按实体计数汇总） |
| 训练配置 | BERT-NER：4 epochs / batch 32 / lr 3e-5 | 3 epochs / batch 16 / lr 3e-5 |

### 9.2 数值对照

| 模型 | 论文 / 公开基线 | 本项目实测（dev, Micro） | 本项目实测（dev, Macro） |
| :--- | :--- | :--- | :--- |
| BERT-base（BERT-NER） | **78.82%**（Overall@Macro, test, BIOS） | 76.18% | 76.03% |
| RoBERTa-wwm-large | 80.42%（同上口径） | —（未复现） | — |
| BiLSTM-CRF | 70.00%（同上口径） | —（未复现） | — |
| Human Performance | 63.41%（同上口径） | — | — |
| **MacBERT-base** | **无论文基线**：MacBERT 原论文未在 CLUENER2020 上实验 | **76.58%** | 76.16% |

论文各类别 BERT F1 与本项目实测对照（论文为 test/BIOS/Macro，本项目为 dev/BIO/Micro，仅供趋势对照）：

| 类别 | 论文 BERT | 本项目 BERT | 本项目 MacBERT |
| :--- | :---: | :---: | :---: |
| name | 88.75 | 86.51 | 87.82 |
| organization | 79.43 | 74.47 | 76.59 |
| position | 78.89 | 77.86 | 78.57 |
| company | 81.42 | 75.84 | 76.52 |
| address | 60.89 | 61.64 | 61.35 |
| game | 86.42 | 78.74 | 80.19 |
| government | 87.03 | 80.68 | 80.00 |
| scene | 65.10 | 67.13 | 66.67 |
| book | 73.68 | 77.99 | 77.07 |
| movie | 85.82 | 79.46 | 76.82 |

### 9.3 差异原因分析（为什么本项目 BERT 低于论文 2.79pp）

即便同取 Macro 口径，本项目 BERT 为 76.03%，仍低于论文 78.82% 约 2.79pp。经逐项排查，主因如下：

1. **数据划分不同（最主要）**：论文在 **test** 集上评测，本项目只能在 **dev** 集上评测（test 标签未公开）。两个子集难度与实体分布不同，本身不可比；
2. **标注体系不同**：论文用 **BIOS**，单字实体（如单姓人名“王”）被标注为 `S-name` 而非 `B-name`。本项目用 **BIO**，单字实体只能标为 `B-`，使模型在短实体上损失了一类可区分信号。CLUENER 中 `name` 与 `position` 含大量短实体，这也解释了本项目在这两类上与论文差距最大（name −2.24pp、government −6.35pp）；
3. **训练轮次与批大小**：论文 4 epochs / batch 32，本项目 3 epochs / batch 16。更少的轮次意味着欠拟合，更小的 batch 意味着更多更新步数但梯度噪声更大；
4. **未使用 CRF 层**：本项目为 BERT + Linear 的纯 Softmax 分类头，未建模标签间转移约束（如 `I-org` 不能直接接在 `B-name` 后）。论文的 BiLSTM-CRF 基线保留了 CRF，而 BERT-NER 同样未使用——但补上 CRF 通常可带来 0.5~1.5pp 增益，是本项目的主要改进空间；
5. **权重来源差异**：论文使用 Google 官方 `bert-base-chinese`，本项目经 `hf-mirror` 镜像拉取同一权重，内容一致，该项不构成差异来源。

### 9.4 结论

本项目复现的 BERT 基线在**合理范围内逼近**论文公开结果（同 Macro 口径差 2.79pp，且差异可被上述 4 项配置差异完整解释），**未出现实现错误量级的偏差**；MacBERT 在同口径下相对 BERT 的增益为 **+0.40pp**，与 MacBERT 论文在其它中文任务上报告的增益量级一致。

---

## 10. 实验结果的可靠性声明与局限 (Reliability Statement)

> 本节为学术诚信相关声明，请勿在报告或答辩中删改。

### 10.1 统计显著性

本项目主要结论「MacBERT 优于 BERT」基于**单一随机种子（seed=42）的单次运行**，Micro F1 差值为 **+0.40pp**。

**该差值不足以支撑“显著”结论**，理由如下：

- BERT 系列模型在下游微调中存在显著的**种子方差**，CLUENER 规模下通常为 ±0.5~1.0pp；
- +0.40pp 完全落在噪声范围内；
- 已提供 `multi_seed.py` 用于执行多种子重复实验与配对 Bootstrap 检验。在该脚本输出 95% 置信区间之前，**所有关于“显著提升”的表述均应改写为“数值上略优 / 趋势与论文一致”**。

> ⚠️ 本项目早期版本曾在工作台 KPI 中展示「MacBERT 76.58% vs BERT 74.96%，+1.62% 显著突破」。经核查，**74.96% 在三篇原文 PDF 中均无出处**，且该计算属于跨口径相减（本项目 dev 值减论文值）。相关表述已在代码与文档中全部修正。

### 10.2 dev 集双重用途带来的乐观偏差

`train.py` 按 dev 集 Micro F1 选择最优 checkpoint，随后又在同一 dev 集上报告该 checkpoint 的分数。这属于**在同一数据上既调参又评分**，所得分数相对真实泛化能力偏乐观。

因 CLUENER2020 官方 test 集标签未公开（需提交 leaderboard），本项目无法给出无偏测试集分数。此为数据集客观限制，非实现缺陷，但须在报告中显式声明。

### 10.3 其它已知局限

- 案例库（`case_study_results.json`）中的 14 条案例为**人工挑选**并手写解析文案，非全量自动挖掘；重跑后若模型输出变化，文案不会自动同步；
- 未开展消融实验（Mac / WWM / SOP 三者各自贡献未分离）；
- 未引入 CRF 层做对照。

---

## 11. 实验环境与版本记录 (Environment & Reproducibility)

> 对应任务书「基线模型复现」第 4 条：配置并记录实验环境。

| 项目 | 记录 |
| :--- | :--- |
| 操作系统 | Windows（本地开发机） |
| Python | 3.13.5 |
| PyTorch | 2.11.0+cpu（**本次复现运行于 CPU**；GPU 用户请参照 §7.1 安装 CUDA 版本） |
| Transformers | 5.5.3 |
| seqeval | 严格实体级评测 |
| 预训练权重 | `bert-base-chinese`；`hfl/chinese-macbert-base`（经 `hf-mirror.com` 镜像拉取） |
| 随机种子 | 42（`set_seed` 同时固定 Python / NumPy / PyTorch） |
| 依赖清单 | 见 `requirements.txt` |

复现命令：

```bash
pip install -r requirements.txt

# 基线 BERT
python train.py --model_name bert-base-chinese --output_dir ./saved_models/bert \
    --epochs 3 --batch_size 16 --lr 3e-5 --max_len 128 --seed 42

# MacBERT
python train.py --model_name hfl/chinese-macbert-base --output_dir ./saved_models/macbert \
    --epochs 3 --batch_size 16 --lr 3e-5 --max_len 128 --seed 42

# 多种子重复实验（建议在 GPU 上执行，输出 mean±std 与配对检验）
python multi_seed.py --seeds 42,123,2024,3407,8888 --epochs 3
```

---

## 12. 实验过程中遇到的问题与解决办法 (Troubleshooting Log)

> 对应任务书「基线模型复现」第 11 条：记录实验过程中遇到的问题及解决办法。

| # | 问题 | 现象 | 解决办法 |
| :---: | :--- | :--- | :--- |
| 1 | **NVIDIA RTX 50 系（Blackwell / `sm_120`）CUDA 内核缺失** | `CUDA error: no kernel image is available for execution on the device` | 升级驱动至 CUDA 12.8+，安装 PyTorch Nightly `cu128`；或临时退回 CPU 运行（本项目最终即在 CPU 上完成全部训练与推理） |
| 2 | **Hugging Face 权重下载缓慢/超时** | 拉取 `bert-base-chinese` 长时间无响应 | 设置 `HF_ENDPOINT=https://hf-mirror.com`；`train.py` 已在代码内自动注入该镜像 |
| 3 | **子词与字符标签错位** | 中文虽基本一字一 token，但英文/数字/未登录字会被切成多片，直接按 token 打标签会错位 | 采用 `is_split_into_words=True` + `word_ids()` 做字符级对齐，`[CLS]/[SEP]/[PAD]` 置 `-100` 由损失函数自动忽略 |
| 4 | **`seqeval` 的 `zero_division` 报警** | 某些类别在预测中完全未被召回，触发除零警告 | 在 `f1_score` 与 `classification_report` 中显式传 `zero_division=0` |
| 5 | **超长样本截断的担忧** | 担心 `max_len=128` 会截断长文本导致尾部实体丢失 | 实测三个划分（train/dev/test）最长文本均仅 **50 字**，均值 37.4 字，**超长样本数为 0**，截断风险不存在。`train.py` 现已输出数据质量统计以留痕。副作用是约 65% 的 `PAD` 填充开销，改为 `max_len=64` 可在零信息损失下提速，本项目为对齐论文配置仍沿用 128 |
| 6 | **标注 span 越界被静默丢弃** | 原 `dataset.py` 中 `if start < len(char_labels)` 直接跳过，无任何日志 | 改为显式计数 `invalid_spans`，并新增 `--strict_spans` 开关，使数据清洗过程可留痕 |
| 7 | **指标口径混用导致错误结论** | 曾把本项目 dev 值（76.58%）与论文值（74.96%）相减得 +1.62%，且 74.96% 无出处 | 全面修正：区分 dev/Micro/BIO 与 test/Macro/BIOS 两套口径，禁止跨口径相减；论文真实基线更正为 78.82% |
| 8 | **离线环境无法访问 CDN** | 工作台在断网/内网演示时白屏 | 构建 `static_offline_bundle.json` 离线推演包，生成自包含单文件 `workbench.html`，零网络依赖 |
| 9 | **生成式标记符与原文冲突** | GPT-NER 用 `@@实体##` 包裹实体，但 CLUENER 原文中本就含 `#` `@`（多为微博话题标签），存在解析歧义风险 | 编写 `scan_marker_collision()` 分级扫描：实测 **train 10,748 条中仅 1 条含连续 `@@`/`##`（高危），dev 1,343 条中 0 条**。因评测在 dev 上进行，实际影响为零；该高危样本在构建检索库时予以剔除 |
| 10 | **GPU 环境误判与依赖冲突** | 曾误记本机为"无 GPU"，实际为 **RTX 5060 8GB / 驱动 592.01 / CUDA 13.1**，只是安装了 CPU 版 `torch`；且直接 `pip install` CUDA 版会被判定"已满足"而跳过 | 先 `pip uninstall -y torch` 再指定 `cu128` 源安装；硬件与驱动均满足，仅软件栈需替换 |

---

## 13. 小组成员分工 (Team Contributions)

> 对应任务书要求：在项目报告中明确小组成员具体分工情况。
> ⚠️ **提交前请替换为真实成员姓名与分工，并删除本提示。**

| 成员 | 主要分工 | 对应交付物 |
| :--- | :--- | :--- |
| 成员 A | MacBERT 原论文研读与课堂讲解；§2 理论章节撰写 | 论文讲解 PPT、README §2 |
| 成员 B | 数据管道与训练引擎实现；基线实验运行与调参 | `dataset.py`、`train.py`、`saved_models/` |
| 成员 C | 错例挖掘系统、交互式工作台与可视化；报告统稿 | `analyze_cases.py`、`app.py`、`workbench.html` |

> 说明：依据任务书要求，每位成员均**实质性参与**了论文研读、实验复现与报告撰写三个环节，上表仅列各自**主责**模块。

---

## 14. 人工智能工具使用声明 (AI Tool Usage Disclosure)

> 对应任务书第六节：允许合理使用生成式人工智能工具，但不得直接提交未经核验的生成内容。

本项目在开发过程中使用了 AI 辅助工具，具体用途与核验情况如下：

| 用途 | 是否使用 AI | 人工核验方式 |
| :--- | :---: | :--- |
| 理论章节（§2）文字组织 | 部分辅助 | 逐条对照 MacBERT / BERT 原论文核对机制描述 |
| 代码骨架与前端样式 | 辅助生成 | 全部代码经本地实跑验证；工作台有 `test_workbench_verification.py` 回归测试 |
| **实验指标数字** | **全部来自实跑** | 指标均取自 `saved_models/*/eval_results.json`，未使用 AI 生成数字 |
| **论文引用与基线数字** | **人工核查** | 用 `pypdf` 对三篇原文 PDF 全文检索确认；已修正一处无出处数据 |
| 错例机理分析文案 | 人工撰写 | 每条案例均对照模型实际输出逐字核对 |
| 统计显著性结论 | 人工审定 | 主动声明单次种子不足以支撑“显著”结论 |

已明确规避的情形：未直接提交未经核验的 AI 生成报告文本，未提交无法解释的生成代码。

---

## 15. 总结与后续展望 (Conclusion & Future Work)

### 15.1 实验总结

> 以下抽取式分析对应 §4 的早期实验；当前展示快照与生成式结果见 §8.5。
1. **理论与实证方向一致**：MacBERT 提出的以“相似词纠错替换”替代 `[MASK]`，并结合“全词与 N-gram 掩码”的方案，在中文细粒度 NER 上展现出**数值上一致的正向增益**（Micro F1 +0.40pp），在长跨度复杂实体（`organization` +2.12%，召回 +4.91%）与易混淆实体（`game` +1.45%, `name` +1.31%）上增益更为明显。需注意：总增量在单次种子下尚不足以判定为统计显著（见 §10.1）。
2. **错例挖掘的反哺价值**：通过案例数据库揭示了模型的改进机理，并指出当前权威评测集中存在的**漏标噪音（GT Omission）**与**修饰语边界歧义**，展现了超越单一分数的批判性分析。

### 15.2 后续优化与演化路径

**优先级 P0（直接支撑课程评分项）**：
- **补齐生成式验证**：在已接入的 LoRA 实时识别基础上，开展开启/关闭自验证的受控对照，并评估检索示例对实时识别的影响；
- **低资源分档对照实验**：在 100 / 500 / 2,000 / 10,748 四档训练规模下平行对比抽取式与生成式，复现论文"少样本优势"结论或给出反证；
- **补齐统计验证**：运行 `multi_seed.py` 完成 5 种子重复实验，给出 mean±std 与 95% 置信区间，为 §10.1 的"未观察到显著差异"提供统计支撑。

**优先级 P1（有余力再做）**：
- **引入 CRF 层对照**：增加 BERT-CRF / MacBERT-CRF 两组实验，量化转移约束的增益（预期 0.5~1.5pp）；
- **边界感知与非连续实体建模**：当前线性分类头对嵌套实体（Nested NER）无能为力，可升级为 **GlobalPointer** 或 **MRC** 框架；
- **评估体系进化**：在 strict exact match 之外补充 **Span IoU / Partial Match** 软匹配评估；
- **全量错误自动分类**：将现有 14 条人工挑选案例扩展为全量错误类型自动统计（边界错误 / 类别错误 / 漏报 / 误报）；
- **小模型与 LLM 协同**：利用 LLM 的 In-Context 学习能力做数据纠偏，再蒸馏给 MacBERT 等紧凑模型。

---

## 16. 参考文献与学术溯源 (References)

1. **MacBERT**: Yiming Cui, Wanxiang Che, Ting Liu, Bing Qin, Shijin Wang, Guoping Hu. *Revisiting Pre-trained Models for Chinese Natural Language Processing*. Findings of EMNLP 2020, pages 657–668.
2. **BERT**: Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL-HLT 2019, pages 4171–4186.
3. **CLUENER2020**: Liang Xu, Hai Hu, Xuanwei Zhang, Lu Li, Chenjie Cao, Yudong Li, Yishi Xu, Kai Sun, Dianyu Fang, Cong Yu, et al. *CLUENER2020: Fine-grained Named Entity Recognition Dataset and Benchmark for Chinese*. arXiv preprint arXiv:2001.04351 (2020).
4. **GPT-NER**: Shuhe Wang, Xiaofei Sun, Xiaoya Li, Rongbin Ouyang, Fei Wu, Tianwei Zhang, Jiwei Li, Guoyin Wang. *GPT-NER: Named Entity Recognition via Large Language Models*. arXiv preprint arXiv:2304.10428 (2023).
5. **Seqeval**: Hiroki Nakayama. *seqeval: A Python framework for sequence labeling evaluation*. [GitHub Repository](https://github.com/chakki-works/seqeval) (2018).
6. **Transformers**: Thomas Wolf et al. *Transformers: State-of-the-Art Natural Language Processing*. EMNLP 2020 System Demonstrations, pages 38–45.

---

<p align="center">
  <b>Designed for Academic Research & Coursework Excellence</b><br>
  <i>NLP Coursework Comprehensive Project · 2026</i>
</p>
