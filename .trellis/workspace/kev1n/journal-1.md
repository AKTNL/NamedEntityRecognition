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

- **工作台交付与全面质检（2026-09-20）**：
  - 成功研发并构建交互式复现工作台（Reproduction Workbench & Live Model Arena），支持 Flask 全栈在线推理 (`app.py`) 与单文件脱机即用 (`workbench.html`) 双模运行。
  - 覆盖四大核心模块：模型在线对决打靶（带边界修复智能 Diff 与底层 Token 探针）、自定义测试用例管理（增删改查、本地持久化与 JSON 导入导出）、复现源码深度透视（带技术考点联动批注）、实测学术指标大盘与代表错例展厅（支持一键载入现场重测）。
  - 完成全面质量验证：端到端 CLI 自检与自动化接口测试 100% 通过，修复了模型参数边界回退、代码批注对齐、自定义用例在线编辑与 HTML 特殊字符转义等细节问题。

- **黑白极简主义（Pure Minimalist Monochrome / Linear 风格）视觉重构（2026-09-24）**：
  - 彻底重构工作台前端色彩系统，去除原有的红绿蓝紫彩度混杂，升级为黑白/纯灰度极致反差风格（深色与浅色双模均满足严格的 $R = G = B$ 纯灰阶）。
  - 创新设计 10 类 CLUENER 实体类别结构化无彩度区分体系：通过反色胶囊、实线、虚线、点线、双实线及多阶灰度明暗实现视觉清晰隔离。
  - 重绘学术指标大盘 SVG 动态柱形图，采用纯白/纯黑高对比渲染 MacBERT 主模型，中灰渲染 BERT 对照基线，全面契合顶级科技与学术产品质感。
  - 经 `node test_frontend_logic.js` 与 `python test_workbench_verification.py` 自动化套件（16项用例）及 `trellis-check` 子智能体深度审查，全部无缝通过。






## Session 1: 校正 NER 报告结论并归档

**Date**: 2026-10-23
**Task**: 校正 NER 报告结论并归档
**Branch**: `main`

### Summary

完成两份报告的实测数据与结论校正，核对来源、样本范围和统计边界，提交推送报告修订并完成任务归档。

### Main Changes

#### 完成内容

- 校正 docs/报告1.md 与 docs/报告2.md 的实验结论、来源和比较边界。
- 区分论文 test/BIOS 与本地 dev/BIO；将 BERT/MacBERT 的差异表述为单次运行高 0.56 个百分点。
- 使用同一 120 句检索对照，分开记录 LoRA 全量开发集与早期小样本结果。
- 校正 LoRA 训练用时、损失、格式异常比例和冲突消解前后的预测计数。
- 保留报告2迁入 docs 的位置，修正两份报告的相对链接。

#### 验证与结论边界

- 抽取式指标、三组检索指标、全量 LoRA 指标和十类结果逐项核对通过。
- 数据规模、非实体字符比例和 tokenizer 长度重新统计；文献原文与训练日志核验完成。
- Markdown 链接、目录锚点、差异检查和归档后的任务上下文验证通过。
- 本次仅修改文档与任务记录，无需重复模型训练或实现测试；现有实验口径规范已覆盖这些修正，无需新增规范。

#### 收尾

- 工作提交：9f6c47b，已推送并核对 origin/main。
- 归档提交：285232e。
- 归档目录：.trellis/tasks/archive/2026-10/10-23-correct-ner-report-conclusions。
- 任务状态为 completed，完成日期 2026-10-23；本次会话的活动任务指针已清除。


### Git Commits

| Hash | Message |
|------|---------|
| `9f6c47b` | (see git log) |
| `285232e` | (see git log) |

### Testing

- [OK] 报告数值与原始 JSON/日志一致；数据和 tokenizer 统计核对通过。
- [OK] Markdown 链接、目录锚点、归档上下文验证及 git diff --check 通过。

### Status

[OK] **Completed**

### Next Steps

- None - task complete
