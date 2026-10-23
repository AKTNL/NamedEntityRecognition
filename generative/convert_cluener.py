# -*- coding: utf-8 -*-
"""
GPT-NER 链路 · 步骤 1：CLUENER2020 → MRC-NER / GPT-NER per-type 格式转换
=========================================================================
把 CLUENER2020 的「原文 + span 标注」展开为 **（句子 × 实体类型）** 的样本，
即官方 GPT-NER 所依赖的 MRC-NER 预处理格式（`mrc-ner.{split}`）。

为什么必须展开成 per-type
-------------------------
GPT-NER 的核心假设是：一次提示只针对**一个**实体类型做标注。论文 Figure 2 的
提示里只出现单一类型描述，示例中的 `@@##` 也只包裹该类型的实体。因此
一条 CLUENER 句子（可能含多类型实体）必须拆成多条单类型样本。

关于「是否为每个句子都生成全部 10 个类型的查询」
-----------------------------------------------
MRC-NER 原实现只为**句中出现过的类型**构造 query（因为没有答案的 query 不产生
训练信号）。但那在测试阶段属于 **oracle 信息泄漏**——提前知道句子含哪些类型，
会显著抬高 F1。本模块在 `test`/`dev` 侧默认构造**全部 10 个类型**的查询
（参数 `--all-types`），以保证与抽取式链路公平可比；训练集仍沿用
「仅保留含该类型的样本」以维持训练/示例池的信号密度。

输出格式（每条样本）
-------------------
    {
      "sample_id":   int,    // 原句在源文件中的行号
      "context":     str,    // 原文
      "entity_label":str,    // 目标实体类型
      "start_position": [int],  // 实体起始字符下标（闭区间，与 CLUENER 一致）
      "end_position":   [int],  // 实体结束字符下标（闭区间）
      "target":      str,    // 用 @@ / ## 标记后的目标输出
      "entities":    [str],  // 实体字符串，便于调试与示例检索
    }

用法
----
    python generative/convert_cluener.py                # 转换 train + dev
    python generative/convert_cluener.py --split dev    # 只转 dev
    python generative/convert_cluener.py --stats        # 只看统计不落盘

作者：《自然语言处理》课程综合实践 · 生成式 NER 链路
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any, Tuple

# 让本模块既能作为包导入，也能直接 `python generative/convert_cluener.py` 运行
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gptner_format import (  # noqa: E402
    CATEGORIES,
    CATEGORY_ZH,
    cluener_label_to_spans,
    mark_entities,
    scan_marker_collision,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLUENER_DIR = os.path.join(PROJECT_ROOT, ".data", "cluener")
OUT_DIR = os.path.join(PROJECT_ROOT, ".data", "gptner")


# ---------------------------------------------------------------------------
# 读写
# ---------------------------------------------------------------------------

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """读取 CLUENER 的 jsonl（每行一个 JSON 对象）。"""
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# 核心转换
# ---------------------------------------------------------------------------

def convert_sentence(
    sample_id: int,
    record: Dict[str, Any],
    entity_type: str,
) -> Dict[str, Any]:
    """把一条 CLUENER 记录转换为单个类型的 MRC-NER 样本。"""
    text: str = record["text"]
    label: Dict[str, Any] = record.get("label") or {}

    gold = [(s, e, c) for s, e, c in cluener_label_to_spans(label) if c == entity_type]

    # 越界防御：CLUENER 的极少数标注存在越界，直接丢弃以免污染目标序列
    valid = [(s, e, c) for s, e, c in gold if 0 <= s <= e < len(text)]

    spans_sorted = sorted(valid, key=lambda x: x[0])
    return {
        "sample_id": sample_id,
        "context": text,
        "entity_label": entity_type,
        "start_position": [s for s, _, _ in spans_sorted],
        "end_position": [e for _, e, _ in spans_sorted],
        "target": mark_entities(text, [(s, e) for s, e, _ in spans_sorted]),
        "entities": [text[s:e + 1] for s, e, _ in spans_sorted],
    }


def convert_split(
    records: List[Dict[str, Any]],
    all_types: bool,
) -> List[Dict[str, Any]]:
    """转换整个 split。

    参数:
        records: CLUENER 原始记录列表
        all_types: True  -> 每句展开全部 CATEGORIES（测试侧，无 oracle）
                   False -> 只展开句中出现的类型（训练/示例池侧）
    """
    out: List[Dict[str, Any]] = []
    for sid, rec in enumerate(records):
        label = rec.get("label") or {}
        present = {c for _, _, c in cluener_label_to_spans(label)}
        types = CATEGORIES if all_types else [c for c in CATEGORIES if c in present]
        for t in types:
            out.append(convert_sentence(sid, rec, t))
    return out


def build_sentence_index(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """构建「原句级」金标索引，供评测阶段还原 BIO 标签链。

    与抽取式 dev 评测的输入保持一一对应（按 dev.json 行序）。
    """
    index: List[Dict[str, Any]] = []
    for sid, rec in enumerate(records):
        spans = cluener_label_to_spans(rec.get("label") or {})
        index.append({
            "sample_id": sid,
            "text": rec["text"],
            "spans": [{"start": s, "end": e, "type": c} for s, e, c in spans],
        })
    return index


# ---------------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------------

def summarize(samples: List[Dict[str, Any]], split: str) -> Dict[str, Any]:
    """输出转换结果的统计信息，便于写进实验报告。"""
    n_sent = len({s["sample_id"] for s in samples})
    n_pos = sum(1 for s in samples if s["start_position"])
    per_type = {}
    for cat in CATEGORIES:
        sub = [s for s in samples if s["entity_label"] == cat]
        per_type[cat] = {
            "queries": len(sub),
            "positive": sum(1 for s in sub if s["start_position"]),
            "entities": sum(len(s["start_position"]) for s in sub),
        }
    stat = {
        "split": split,
        "sentences": n_sent,
        "queries": len(samples),
        "positive_queries": n_pos,
        "positive_ratio": round(n_pos / len(samples), 4) if samples else 0.0,
        "total_entities": sum(len(s["start_position"]) for s in samples),
        "per_type": per_type,
    }
    return stat


def print_stats(stat: Dict[str, Any]) -> None:
    print(f"\n[{stat['split']}] 原句 {stat['sentences']} 条 -> per-type 查询 {stat['queries']} 条")
    print(f"    正例查询 {stat['positive_queries']} 条 "
          f"({stat['positive_ratio'] * 100:.2f}%)，实体总数 {stat['total_entities']}")
    print(f"    {'类型':<14}{'查询数':>8}{'正例':>8}{'实体数':>8}")
    for cat, v in stat["per_type"].items():
        print(f"    {cat + '(' + CATEGORY_ZH[cat] + ')':<14}"
              f"{v['queries']:>8}{v['positive']:>8}{v['entities']:>8}")


# ---------------------------------------------------------------------------
# 自检：转换结果必须能无损还原金标
# ---------------------------------------------------------------------------

def verify_roundtrip(
    records: List[Dict[str, Any]],
    samples: List[Dict[str, Any]],
) -> bool:
    """校验：由 per-type 样本合并回的 span 集合 == 原始 CLUENER 金标。"""
    merged: Dict[int, List[Tuple[int, int, str]]] = {}
    for s in samples:
        if not s["start_position"]:
            continue
        for st, en in zip(s["start_position"], s["end_position"]):
            merged.setdefault(s["sample_id"], []).append((st, en, s["entity_label"]))

    ok = True
    for sid, rec in enumerate(records):
        gold = set(cluener_label_to_spans(rec.get("label") or {}))
        pred = set(merged.get(sid, []))
        if gold != pred:
            print(f"  [FAIL] sample {sid}: 合并结果不一致\n    gold={sorted(gold)}\n    pred={sorted(pred)}")
            ok = False
            if not ok and sid > 5:
                break
    return ok


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="CLUENER2020 → GPT-NER per-type 格式转换")
    parser.add_argument("--split", type=str, default="all", choices=["train", "dev", "all"])
    parser.add_argument("--stats", action="store_true", help="只打印统计，不落盘")
    parser.add_argument(
        "--train-positive-only",
        action="store_true",
        default=True,
        help="训练集只保留含目标类型的样本（默认开启，维持示例池信号密度）",
    )
    parser.add_argument(
        "--train-all-types",
        dest="train_positive_only",
        action="store_false",
        help="训练集也展开全部类型（关闭上一条）",
    )
    args = parser.parse_args()

    print("=" * 78)
    print("GPT-NER 步骤 1 · CLUENER2020 → per-type 格式转换")
    print("=" * 78)

    splits = ["train", "dev"] if args.split == "all" else [args.split]

    # 标记符冲突扫描：若原文含 @@ / ## 会破坏标记协议，必须先知道
    print("\n标记符冲突扫描（原文是否含 @@ / ##）：")
    for sp in splits:
        path = os.path.join(CLUENER_DIR, f"{sp}.json")
        if os.path.exists(path):
            r = scan_marker_collision(path)
            flag = "无冲突" if r["high"] == 0 else f"⚠ {r['high']} 条高危"
            print(f"  {sp}.json: {r['total']} 条 | {flag} | 低危(单字符@/#) {r['low']} 条")
            for t in r["high_samples"][:3]:
                print(f"      [需处理] {t[:60]}")

    all_stats: List[Dict[str, Any]] = []

    for sp in splits:
        src = os.path.join(CLUENER_DIR, f"{sp}.json")
        if not os.path.exists(src):
            print(f"\n[跳过] {src} 不存在")
            continue

        records = read_jsonl(src)
        # 训练集：per-type 示例池，只保留正例；dev：全部 10 类型查询（无 oracle）
        all_types = (sp == "dev")
        samples = convert_split(records, all_types=all_types)

        stat = summarize(samples, sp)
        all_stats.append(stat)
        print_stats(stat)

        # 自检
        rt = verify_roundtrip(records, samples)
        print(f"    [{'OK' if rt else 'ERROR'}] 往返一致性：per-type 合并 == 原始金标")

        if not args.stats:
            out_path = os.path.join(OUT_DIR, f"mrc-ner.{sp}.json")
            write_json(out_path, samples)
            print(f"    -> 已写入 {out_path} ({len(samples)} 条)")

            idx_path = os.path.join(OUT_DIR, f"sentence-index.{sp}.json")
            write_json(idx_path, build_sentence_index(records))
            print(f"    -> 已写入 {idx_path} ({len(records)} 条原句索引)")

    if not args.stats and all_stats:
        meta_path = os.path.join(OUT_DIR, "convert_stats.json")
        write_json(meta_path, all_stats)
        print(f"\n统计摘要 -> {meta_path}")

    print("\n说明：dev 侧展开全部 10 个类型（不泄漏「句中有哪些类型」的 oracle 信息），"
          "因此正例占比很低，这是 GPT-NER 在真实设定下的固有代价。")


if __name__ == "__main__":
    main()
