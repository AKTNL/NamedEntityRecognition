# -*- coding: utf-8 -*-
"""
GPT-NER 链路 · 步骤 5：解析 → 定位 → 转 BIO → seqeval 评测
=========================================================================
**这是保证"两种范式数字可比"的关键模块。**

口径对齐（为什么必须这样做）
----------------------------
官方 GPT-NER 的评测是 **span 级、不含类型维度**（因为 per-type 提示里类型已隐含，
`compute_f1.py` 只比对 span 是否命中）。而本项目抽取式链路用的是
**seqeval 的 BIO 实体级 Micro F1**。两者**不可直接相减**。

因此本模块把生成结果强行桥接回抽取式的输入形态：
    模型输出 → 解析 @@## → 在**原文**中定位 span → 生成 BIO 标签链 → seqeval
这样生成式与抽取式跑的是**同一个 `f1_score` 调用**，数字才有可比性。

幻觉（hallucination）处理
------------------------
生成式方法的典型失效是生成原文中不存在的实体。本模块按如下策略处理：
  - 实体串能在原文中找到 → 正常计入预测
  - 找不到 → 记为**幻觉**，默认丢弃（不计入 TP/FP），并单独统计幻觉率
这是"保守对齐"策略：若把幻觉计入 FP，等于用**格式错误**去惩罚 F1，
会让生成式数字不可解释；单独统计反而更能讲清失效模式。

跨类型冲突消解
--------------
每个类型独立查询，可能出现同一片段被判为多类。
规则：重叠 span **保留更长者**，长度相同则按 `CATEGORIES` 顺序取先者。

用法
----
    from evaluate import GptNerEvaluator
    ev = GptNerEvaluator(sentence_index)
    ev.add_prediction(sample_id, entity_type, model_output, original_text)
    metrics = ev.compute()

作者：《自然语言处理》课程综合实践 · 生成式 NER 链路
"""

import os
import sys
import json
import re
from typing import List, Dict, Any, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gptner_format import CATEGORIES, OPEN_TOKEN, CLOSE_TOKEN, spans_to_bio  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from seqeval.metrics import f1_score, precision_score, recall_score, classification_report  # noqa: E402


_MARK_RE = re.compile(re.escape(OPEN_TOKEN) + r"(.*?)" + re.escape(CLOSE_TOKEN))


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------

def extract_entity_strings(output: str) -> List[str]:
    """从模型输出中抽取 `@@...##` 包裹的实体字符串。

    只保留**成对匹配**的部分；多余的 `@@` / `##` 一律忽略，
    这样即使模型输出格式不合规也不会污染结果。
    """
    ents: List[str] = []
    for m in _MARK_RE.finditer(output or ""):
        e = m.group(1).strip()
        if e:
            ents.append(e)
    return ents


def locate_in_text(text: str, entity: str, used: Optional[set] = None) -> int:
    """在原文中定位实体串，返回起始下标；找不到返回 -1。

    同一实体串可能在句中多次出现，这里按出现顺序依次占用，
    避免"两个相同实体都匹配到同一处"的重复计数。
    """
    used = used if used is not None else set()
    start = 0
    while True:
        idx = text.find(entity, start)
        if idx < 0:
            return -1
        if idx not in used:
            used.add(idx)
            return idx
        start = idx + 1


# ---------------------------------------------------------------------------
# 评测器
# ---------------------------------------------------------------------------

class GptNerEvaluator:
    """把 per-type 的生成结果汇总为句子级 BIO 序列并交给 seqeval。"""

    def __init__(self, sentence_index: List[Dict[str, Any]]):
        """
        参数:
            sentence_index: `sentence-index.{split}.json` 的内容，
                            每项含 sample_id / text / spans
        """
        self.index = {s["sample_id"]: s for s in sentence_index}
        self.ids: List[int] = [s["sample_id"] for s in sentence_index]

        # sample_id -> List[(start, end, type)]
        self.pred_spans: Dict[int, List[Tuple[int, int, str]]] = {i: [] for i in self.ids}
        # 统计
        self.n_output = 0
        self.n_hallucination = 0
        self.n_malformed = 0
        self.hallucination_samples: List[str] = []

    # ---------------- 收集预测 ----------------

    def add_prediction(
        self,
        sample_id: int,
        entity_type: str,
        model_output: str,
        original_text: str,
    ) -> None:
        """解析一条 per-type 模型输出并登记预测 span。"""
        self.n_output += 1
        ents = extract_entity_strings(model_output)

        # 格式不合规：有 @@ 但没有配对成功，或残留孤立标记
        raw_open = (model_output or "").count(OPEN_TOKEN)
        raw_close = (model_output or "").count(CLOSE_TOKEN)
        if raw_open != raw_close or raw_open != len(ents):
            self.n_malformed += 1

        for e in ents:
            pos = locate_in_text(original_text, e)
            if pos < 0:
                self.n_hallucination += 1
                if len(self.hallucination_samples) < 20:
                    self.hallucination_samples.append(e)
                continue
            self.pred_spans[sample_id].append((pos, pos + len(e) - 1, entity_type))

    # ---------------- 冲突消解 ----------------

    @staticmethod
    def resolve_conflicts(spans: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
        """重叠 span 保留更长者；长度相同按类型顺序取先者。"""
        if not spans:
            return []
        order = {c: i for i, c in enumerate(CATEGORIES)}
        # 长 span 优先，其次起点靠前，最后类型顺序
        ordered = sorted(spans, key=lambda x: (-(x[1] - x[0]), x[0], order.get(x[2], 99)))
        kept: List[Tuple[int, int, str]] = []
        for s, e, c in ordered:
            if all(e < ks or s > ke for ks, ke, _ in kept):
                kept.append((s, e, c))
        return sorted(kept)

    # ---------------- 计算指标 ----------------

    def compute(self) -> Dict[str, Any]:
        """生成 BIO 序列并调用 seqeval（与抽取式链路完全相同的调用方式）。"""
        y_true: List[List[str]] = []
        y_pred: List[List[str]] = []

        for sid in self.ids:
            text = self.index[sid]["text"]
            gold = [(s["start"], s["end"], s["type"]) for s in self.index[sid]["spans"]]
            pred = self.resolve_conflicts(self.pred_spans.get(sid, []))

            y_true.append(spans_to_bio(len(text), gold))
            y_pred.append(spans_to_bio(len(text), pred))

        metrics = {
            "micro_f1": round(float(f1_score(y_true, y_pred, zero_division=0)) * 100, 2),
            "micro_p": round(float(precision_score(y_true, y_pred, zero_division=0)) * 100, 2),
            "micro_r": round(float(recall_score(y_true, y_pred, zero_division=0)) * 100, 2),
            "report": classification_report(y_true, y_pred, digits=4, zero_division=0),
        }

        n_pred = sum(len(v) for v in self.pred_spans.values())
        n_gold = sum(len(self.index[i]["spans"]) for i in self.ids)
        metrics.update({
            "n_sentences": len(self.ids),
            "n_gold_entities": n_gold,
            "n_pred_entities": n_pred,
            "n_queries": self.n_output,
            "n_hallucination": self.n_hallucination,
            "n_malformed": self.n_malformed,
            "hallucination_rate": round(self.n_hallucination / self.n_output, 4)
            if self.n_output else 0.0,
            "malformed_rate": round(self.n_malformed / self.n_output, 4)
            if self.n_output else 0.0,
            "hallucination_samples": self.hallucination_samples,
        })
        return metrics

    # ---------------- 逐类型细粒度指标 ----------------

    def per_type(self) -> Dict[str, Dict[str, float]]:
        """按类型统计 P/R/F1（span 级精确匹配，含类型）。"""
        tp: Dict[str, int] = {c: 0 for c in CATEGORIES}
        ngold: Dict[str, int] = {c: 0 for c in CATEGORIES}
        npred: Dict[str, int] = {c: 0 for c in CATEGORIES}

        for sid in self.ids:
            gold = {(s["start"], s["end"], s["type"]) for s in self.index[sid]["spans"]}
            pred = set(self.resolve_conflicts(self.pred_spans.get(sid, [])))
            for g in gold:
                ngold[g[2]] += 1
            for p in pred:
                npred[p[2]] += 1
                if p in gold:
                    tp[p[2]] += 1

        out: Dict[str, Dict[str, float]] = {}
        for c in CATEGORIES:
            p = tp[c] / npred[c] if npred[c] else 0.0
            r = tp[c] / ngold[c] if ngold[c] else 0.0
            f = 2 * p * r / (p + r) if (p + r) else 0.0
            out[c] = {
                "precision": round(p * 100, 2),
                "recall": round(r * 100, 2),
                "f1": round(f * 100, 2),
                "support": ngold[c],
                "predicted": npred[c],
            }
        return out


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------

def selftest() -> bool:
    """构造一个可控的小样例，验证解析 / 定位 / BIO / seqeval 全链路。"""
    index = [
        {
            "sample_id": 0,
            "text": "浙商银行企业信贷部叶老桂博士则从另一个角度对五道门槛进行了解读。",
            "spans": [
                {"start": 0, "end": 3, "type": "company"},
                {"start": 9, "end": 11, "type": "name"},
            ],
        },
        {"sample_id": 1, "text": "这句话里没有任何实体。", "spans": []},
    ]
    ev = GptNerEvaluator(index)

    # 句 0：company 命中，name 幻觉（原文没有"张三"）
    ev.add_prediction(0, "company",
                      "@@浙商银行##企业信贷部叶老桂博士则从另一个角度对五道门槛进行了解读。",
                      index[0]["text"])
    ev.add_prediction(0, "name", "...@@张三##...", index[0]["text"])
    # 句 1：模型正确输出"无实体"
    ev.add_prediction(1, "company", "这句话里没有任何实体。", index[1]["text"])

    m = ev.compute()
    print(f"  micro P/R/F1 = {m['micro_p']}/{m['micro_r']}/{m['micro_f1']}")
    print(f"  幻觉 {m['n_hallucination']} 条（应为 1）| 格式异常 {m['n_malformed']} 条（应为 0）")

    ok = True
    if m["n_hallucination"] != 1:
        print("  [FAIL] 幻觉计数不对"); ok = False
    if m["n_pred_entities"] != 1:
        print(f"  [FAIL] 预测实体数应为 1，实际 {m['n_pred_entities']}"); ok = False
    if not (0 < m["micro_f1"] <= 100):
        print("  [FAIL] F1 越界"); ok = False
    return ok


if __name__ == "__main__":
    print("=" * 78)
    print("GPT-NER 评测模块 · 自检")
    print("=" * 78)
    print("  [OK] 全链路自检通过" if selftest() else "  [ERROR] 自检未通过")
