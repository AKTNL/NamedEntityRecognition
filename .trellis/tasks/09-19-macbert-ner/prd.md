# MacBERT与中文NER交互式复现工作台与全栈模型体验系统

## Goal
构建一个集“复现代码深度透视”、“跑起来的模型同屏对决打靶”、“用户自定义测试用例管理”、“真实评测指标大盘”与“深度错例挖掘展厅”于一体的现代化交互式复现工作台（Reproduction Workbench & Live Model Arena）。以轻量 Python Flask 服务 + 极简现代化高颜值 Web 前端为载体，既能实时加载运行保存的 BERT 与 MacBERT 真实权重进行毫秒级双模型推理对比，又能优雅进行离线回退演示，全方位展现代码复现工程深度与模型实际落地效果。

## Requirements
1. **工作台核心架构与运行形态（Workbench Architecture）**：
   - **B/S 轻量全栈服务（`app.py`）**：
     - 基于环境已有的 Flask，启动极快，常驻加载 `saved_models/bert` 与 `saved_models/macbert` 权重。
     - 提供 REST API：
       - `POST /api/predict`：接收输入文本，同时执行两个模型的实时前向序列标注与实体抽取，返回实体跨度、类型及 Token 级 BIO 标签。
       - `GET /api/metrics`：读取 `saved_models/*/eval_results.json`，返回两个模型的真实评估指标。
       - `GET /api/cases`：读取 `case_study_results.json`，返回挖掘出的代表性真实错例。
       - 提供静态前端托管与自动浏览器打开引导。
   - **离线优雅降级（Graceful Offline Fallback）**：
     - 前端启动时探测后端连通性。若未启动 Flask 后端，自动降级为“内置离线演示模式”，加载预置测试用例与静态推演结果，支持在任何脱机环境（双击 HTML）流畅浏览，绝不白屏或报错。
   - **现代学术科技风格（Theme & UI）**：
     - 科技深色（Deep Slate）与高校学术浅色（Academic Light）一键秒切。
     - 响应式自适应布局，零繁重编译依赖，极速秒开。

2. **四大核心功能模块**：
   - **模块 1：⚡ 模型对决与在线打靶工作台（Model Arena & Interactive Playground）**：
     - **富文本输入与快速词库**：输入框支持字数统计、快捷清空，内置涵盖 10 类实体的丰富预设测试句（政务、财经、文娱、生僻专有名词）。
     - **用户自定义用例系统（Custom Cases System）**：
       - 支持用户输入文本后一键“保存为自定义用例”，可自定义用例标签与备注；
       - 基于浏览器 `LocalStorage` 持久化存储，页面刷新不丢失；
       - 支持自定义用例的快捷载入、编辑与删除；
       - 提供“导出 JSON”与“导入 JSON”功能，方便团队成员间共享测试用例集。
     - **同屏双模型 PK（MacBERT vs BERT）**：
       - **实体高亮渲染**：10 种实体类型不同主题色标签（地址、书籍、公司、游戏、政府、电影、姓名、组织、职位、景点），悬浮展示实体类型、Span 区间 `[start, end]` 及置信度。
       - **差异对比视窗（Diff Mode）**：智能对比标注 MacBERT 与 BERT 识别不一致之处，直观突显 MacBERT 在“复合名词、长实体边界、易混淆类别”上的纠错与全词掩码优势。
       - **底层 Token 探针（可展开）**：逐字展示原始字符对应的预测 BIO 标签。
   - **模块 2：💻 复现代码深度透视（Codebase Walkthrough）**：
     - **模块化代码阅读器**：在 `dataset.py`（数据管道/BIO对齐）、`train.py`（微调训练循环/优化器）、`analyze_cases.py`（评估指标/错例挖掘）间流畅切换。
     - **“原理-代码”联动批注（Code Annotations）**：重点批注子词对齐难点、动态 Padding 原理、MacBERT 预训练模型加载与分类头适配、Seqeval 严格实体匹配算法。
   - **模块 3：📊 实测学术指标大盘（Benchmark Dashboard）**：
     - 基于真实评测数据展示：BERT ($F_1 = 74.96\%$) vs MacBERT ($F_1 = 76.58\%$, $+1.62\%$ 显著提升)。
     - 10 类细粒度实体分类指标对比柱状图 / 差距分析（Precision / Recall / $F_1$）。
     - 完整记录实验超参数基准（Batch Size=16, LR=3e-5, Epochs=3, Max Len=128）。
   - **模块 4：🔍 真实错例深度展厅（Bad Case Gallery）**：
     - 展示挖掘出的三类经典错例：实体边界截断、细粒度类别混淆、未登录词识别。
     - 每张错例卡片支持“**载入到工作台重测**”按钮，一键将错例送入模型对决区进行现场打靶复现。

## Acceptance Criteria
- [x] 启动 `python app.py` 能够稳定加载两个模型，提供 Web 服务并在控制台打印访问地址。
- [x] 模型对决工作台支持中文文本实时输入与打靶，能在 100ms 级别内返回并同屏渲染 BERT 与 MacBERT 的实体识别结果及差异对比。
- [x] 自定义用例系统运行稳定，支持新增、载入、删除用例，数据在 LocalStorage 中持久化，支持 JSON 导入导出。
- [x] 代码透视模块能够清晰高亮展示 `dataset.py`, `train.py`, `analyze_cases.py` 核心代码与技术批注。
- [x] 指标大盘准确呈现已跑出的真实评估数据（MacBERT 76.58% vs BERT 74.96%），图表清晰美观。
- [x] 错例展厅展示真实 case，且“载入重测”功能能无缝联动到工作台输入框。
- [x] 支持离线模式优雅降级（即使不启动后端直接在浏览器打开静态 HTML，也能浏览代码、指标大盘、错例库并体验离线测试样本）。

## Definition of Done
- 后端 `app.py` 代码健壮、注释规范，异常处理完善（如空输入、超长文本截断、特殊符号安全过滤）。
- 前端交互丝滑，无控制台报错，深浅色主题适配完美。
- 完整包含使用说明（如 README 中新增工作台启动指南）。
