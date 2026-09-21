# Journal - kev1n (Part 1)

> AI development session journal
> Started: 2026-09-19

---

## 2026-09-19: MacBERT 与中文 NER 经典论文分享与复现系统 (09-19-macbert-ner)

- **任务背景**：基于《作业说明.pdf》要求，实现任务一“命名实体识别”的经典论文分享系统，覆盖 9 大核心要素并打通与实验复现的闭环。
- **架构设计**：
  - 核心叙事：“一主两翼”（MacBERT 60% 核心创新 + BERT 20% 技术渊源 + CLUENER2020 20% 评测与复现落地）。
  - 形态：单文件纯离线 Standalone HTML (`第X组_命名实体识别_经典论文分享.html`)，0 外部 CDN 依赖，支持幻灯片演示模式与文档精读模式（Dual-View），支持深浅主题无缝切换与 `Ctrl+P` 打印导出 PDF。
  - 交互组件：
    1. 掩码机制演进对比模拟器（BERT vs WWM vs MacBERT）。
    2. CLUENER2020 细粒度实体标注与 BIO 探针（10 类实体过滤高亮）。
    3. 基线复现性能沙盘与 Bad Case 交互卡片（边界切偏、细粒度混淆、长实体截断）。
- **策略调整（2026-09-20）**：
  - 用户反馈先集中精力进行代码复现与实验验证，后续基于真实的实验数据和错例分析再制作展示汇报/PPT。
  - 已安全删除未结合真实实验数据的草稿 HTML 文件。
  - 下一步重点：准备复现环境、搭建 CLUENER2020 数据流与 Baseline（BERT-base / MacBERT-base）微调与评测代码。


