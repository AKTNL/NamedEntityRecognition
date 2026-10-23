# -*- coding: utf-8 -*-
"""
GPT-NER 实验结果汇总
=========================================================================
读取 `results/generative/*.json`，拼成一张可直接搬进实验报告的对比表，
并与抽取式链路的数字并排展示（抽取式数字来自 README 的权威口径，手工录入）。

用法
----
    python generative/summarize.py
    python generative/summarize.py --dir results/generative

作者：《自然语言处理》课程综合实践 · 生成式 NER 链路
"""

import os
import sys
import json
import glob
import argparse
from typing import List, Dict, Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DIR = os.path.join(PROJECT_ROOT, "results", "generative")

# 抽取式链路权威数字（dev, seqeval 实体级 Micro F1），来源：README / saved_models
EXTRACTIVE_BASELINES = {
    "BERT (bert-base-chinese)": 76.18,
    "MacBERT (hfl/chinese-macbert-base)": 76.58,
}


def load_results(d: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        try:
            with open(p, encoding="utf-8") as f:
                obj = json.load(f)
            if "metrics" in obj and "config" in obj:
                obj["_file"] = os.path.basename(p)
                out.append(obj)
        except Exception as e:  # noqa: BLE001
            print(f"  [跳过] {p}: {e}")
    return out


def label_of(cfg: Dict[str, Any]) -> str:
    tag = f"{cfg['strategy']} k={cfg['k']}"
    if cfg.get("neg_ratio") is not None and cfg["k"] > 0:
        tag += f" neg={cfg['neg_ratio']}"
    if cfg.get("self_verify"):
        tag += " +自验证"
    if cfg.get("lora"):
        tag += " +LoRA"
    if cfg.get("limit"):
        tag += f" (前{cfg['limit']}句)"
    return tag


def main() -> None:
    ap = argparse.ArgumentParser(description="汇总 GPT-NER 实验结果")
    ap.add_argument("--dir", type=str, default=DEFAULT_DIR)
    args = ap.parse_args()

    res = load_results(args.dir)
    if not res:
        print(f"未在 {args.dir} 找到结果文件")
        return

    print("=" * 92)
    print("生成式（GPT-NER）实验结果汇总 · seqeval 实体级 Micro F1，与抽取式同口径")
    print("=" * 92)
    print(f"{'配置':<44}{'P':>8}{'R':>8}{'F1':>8}{'幻觉率':>9}{'格式异常':>9}")
    print("-" * 92)

    rows: List[tuple] = []
    for r in res:
        m, c = r["metrics"], r["config"]
        rows.append((m["micro_f1"], label_of(c), m["micro_p"], m["micro_r"],
                     m["hallucination_rate"] * 100, m["malformed_rate"] * 100))

    for f1, name, p, rc, hal, mal in sorted(rows, reverse=True):
        print(f"{name:<44}{p:>8.2f}{rc:>8.2f}{f1:>8.2f}{hal:>8.1f}%{mal:>8.1f}%")

    print("-" * 92)
    print("对照 · 抽取式链路（本项目 dev 权威数字）：")
    for k, v in EXTRACTIVE_BASELINES.items():
        print(f"{k:<44}{'—':>8}{'—':>8}{v:>8.2f}{'—':>9}{'—':>9}")

    print("\n口径说明：")
    print("  1. 生成式与抽取式均使用 seqeval 的实体级 Micro F1，数字可直接比较。")
    print("  2. 生成式基座为 Qwen2.5-1.5B-Instruct（本地 1.5B），论文原作为 GPT-3(175B)，")
    print("     参数量相差两个数量级，绝对差距应归因于**基座规模**，不应解读为方法失效。")
    print("  3. 幻觉率 = 生成了原文中不存在的实体串的查询占比；格式异常 = @@/## 不配对的比例。")
    print("     这两个指标是生成式范式**特有**的失效模式，抽取式不存在。")


if __name__ == "__main__":
    main()
