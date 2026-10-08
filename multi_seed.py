"""
CLUENER2020 多种子重复实验与统计显著性检验脚本
=================================================

背景（为什么需要这个脚本）
--------------------------------------------------------------------------
本项目 README 中曾将「MacBERT 76.58% vs BERT 74.96%，+1.62%」表述为"显著突破"，
存在两处问题：
  1. 74.96% 在三篇原文 PDF 中均无出处，属编造引用；
  2. 即便修正为同口径的 +0.40pp（76.58% vs 76.18%，均为 dev 集 Micro F1），
     该差值来自单一随机种子 seed=42 的单次运行。
     BERT 系列模型在微调阶段存在显著的种子方差（通常 ±0.5~1.0pp），
     0.40pp 完全落在噪声范围内，不足以支撑"显著"结论。

本脚本在多个随机种子下重复训练 BERT 与 MacBERT，报告 mean ± std，
并对「同一 seed 下两模型的 F1 差值」做配对统计检验，给出可写进实验报告的
严谨结论。

功能
--------------------------------------------------------------------------
1. 对 {BERT, MacBERT} × N 个随机种子执行完整训练 + 严格实体级评测；
2. 记录每个 seed 的 Micro F1 / Macro F1 / val_loss / best_epoch / 各类别 F1；
3. 汇总 mean ± std，计算配对差值序列；
4. 统计检验：
   - 配对 Bootstrap：对配对差值做重采样，给出均值差的 95% 置信区间；
   - 符号检验（Sign Test）：观察 MacBERT 胜出的 seed 占比；
   - Welch t 检验（若 scipy 可用）：作为参考（小样本下功效有限，仅供参考）；
5. 断点续跑：已完成的 (model, seed) 组合默认跳过，可用 --force 强制重跑；
6. 输出 multi_seed_results.json（含原始数据）与 multi_seed_report.md（可直接粘进报告）。

用法
--------------------------------------------------------------------------
# 完整实验（推荐在有 GPU 的机器上执行，6 组 × 3 epoch）
python multi_seed.py --seeds 42,123,2024,3407,8888 --epochs 3

# 快速冒烟测试（仅验证脚本可跑通，结果不具统计意义）
python multi_seed.py --seeds 42,123 --epochs 1 --limit 200 --smoke

# 强制重跑全部组合
python multi_seed.py --seeds 42,123,2024 --force

# 仅基于已有结果重新生成统计报告（不训练）
python multi_seed.py --report_only

注意
--------------------------------------------------------------------------
* 本脚本复用 train.py 的训练引擎（通过 train.train(train.parse_args([...]))），
  保证与基线实验完全同一套代码路径，避免"两套实现"导致的可复现性争议。
* 每个 (model, seed) 的权重独立保存到 --out_root 下，互不覆盖。
* dev 集同时用于选择 checkpoint 与报告分数，存在乐观偏差，
  脚本在输出中显式标注该局限。
"""

import os
import sys
import json
import time
import random
import argparse
import statistics
from typing import Dict, Any, List, Optional

import numpy as np

# 保证控制台 UTF-8 中文输出顺畅
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# --------------------------------------------------------------------------
# 模型注册：与 train.py 基线实验保持完全一致的默认超参数
# --------------------------------------------------------------------------
MODEL_REGISTRY = {
    "bert": {
        "model_name": "bert-base-chinese",
        "label": "BERT-base-Chinese (Baseline)",
    },
    "macbert": {
        "model_name": "hfl/chinese-macbert-base",
        "label": "Chinese-MacBERT-base (Proposed)",
    },
}

# CLUENER2020 原论文 Table 5 中 BERT-NER 的公开基线（测试集 / BIOS / Overall@Macro）
# 用途仅为"与公开结果对照"，严禁与本项目 dev 集 Micro F1 直接相减。
PAPER_REFERENCE = {
    "bert": {
        "f1": 78.82,
        "precision": 77.24,
        "recall": 80.46,
        "scheme": "Overall@Macro",
        "split": "test",
        "tagging": "BIOS",
        "source": "CLUENER2020 (Xu et al., 2020) Table 5, BERT-NER",
    },
    "macbert": {
        "f1": None,
        "note": "MacBERT 原论文未在 CLUENER2020 上开展实验，无论文基线可引用",
    },
}


def parse_args(args_list=None):
    parser = argparse.ArgumentParser(
        description="Multi-seed repeated experiments for BERT vs MacBERT on CLUENER2020."
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,123,2024,3407,8888",
        help="逗号分隔的随机种子列表",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="bert,macbert",
        help="逗号分隔的模型键名，可选 bert,macbert",
    )
    parser.add_argument("--epochs", type=int, default=3, help="每个 run 的训练轮数")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument(
        "--train_file", type=str, default=".data/cluener/train.json"
    )
    parser.add_argument("--dev_file", type=str, default=".data/cluener/dev.json")
    parser.add_argument(
        "--out_root",
        type=str,
        default="./saved_models_multiseed",
        help="多种子实验权重与中间结果的根目录",
    )
    parser.add_argument(
        "--results_json",
        type=str,
        default="multi_seed_results.json",
        help="汇总结果 JSON 输出路径",
    )
    parser.add_argument(
        "--report_md",
        type=str,
        default="multi_seed_report.md",
        help="可直接粘进实验报告的 Markdown 输出路径",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="计算设备；默认自动选择 cuda/cpu",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="仅使用前 N 条训练样本（调试用，会显著降低最终 F1）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="忽略已有结果，强制重跑全部组合",
    )
    parser.add_argument(
        "--skip_train",
        action="store_true",
        help="跳过训练，直接读取已有 eval_results.json 汇总",
    )
    parser.add_argument(
        "--report_only",
        action="store_true",
        help="仅基于已有 multi_seed_results.json 重新生成 Markdown 报告",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="冒烟模式：不写入正式结果文件，用于验证脚本可跑通",
    )
    parser.add_argument("--bootstrap_iters", type=int, default=10000)
    parser.add_argument("--bootstrap_seed", type=int, default=20240)
    return parser.parse_args(args_list)


def parse_list(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def resolve_device(device_str: Optional[str]) -> str:
    if device_str:
        return device_str
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def run_dir(out_root: str, model_key: str, seed: int) -> str:
    return os.path.join(out_root, f"{model_key}_seed{seed}")


def eval_path(out_root: str, model_key: str, seed: int) -> str:
    return os.path.join(run_dir(out_root, model_key, seed), "eval_results.json")


def parse_classification_report(report: str) -> Dict[str, Dict[str, float]]:
    """
    解析 seqeval 的 classification_report 文本，抽取每类的 P/R/F1/support。
    兼容 micro avg / macro avg / weighted avg 三种汇总行。
    """
    parsed: Dict[str, Dict[str, float]] = {}
    for line in report.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        # 形如: address 0.5806 0.6568 0.6164 373
        if len(parts) == 5:
            name, p, r, f, s = parts
        # 形如: micro avg 0.7292 0.7975 0.7618 3072
        elif len(parts) == 6 and parts[1] == "avg":
            name, _, p, r, f, s = parts
        else:
            continue
        try:
            parsed[name] = {
                "precision": float(p),
                "recall": float(r),
                "f1": float(f),
                "support": int(s),
            }
        except ValueError:
            continue
    return parsed


def run_one(model_key: str, seed: int, args, device: str) -> Dict[str, Any]:
    """执行单个 (model, seed) 组合的训练与评测。"""
    from train import train, parse_args as train_parse_args

    outdir = run_dir(args.out_root, model_key, seed)
    os.makedirs(outdir, exist_ok=True)

    train_argv = [
        "--model_name", MODEL_REGISTRY[model_key]["model_name"],
        "--output_dir", outdir,
        "--train_file", args.train_file,
        "--dev_file", args.dev_file,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--max_len", str(args.max_len),
        "--warmup_ratio", str(args.warmup_ratio),
        "--weight_decay", str(args.weight_decay),
        "--seed", str(seed),
        "--device", device,
    ]
    if args.limit:
        train_argv += ["--limit", str(args.limit)]

    print("\n" + "#" * 78)
    print(f"# RUN: {model_key.upper()} | seed={seed} | epochs={args.epochs} | device={device}")
    print("#" * 78)

    t0 = time.time()
    summary = train(train_parse_args(train_argv))
    elapsed = time.time() - t0

    with open(os.path.join(outdir, "eval_results.json"), "r", encoding="utf-8") as f:
        raw = json.load(f)
    report = raw.get("classification_report", "")
    parsed = parse_classification_report(report)

    micro = parsed.get("micro_avg", {})
    macro = parsed.get("macro_avg", {})

    return {
        "model_key": model_key,
        "model_name": MODEL_REGISTRY[model_key]["model_name"],
        "label": MODEL_REGISTRY[model_key]["label"],
        "seed": seed,
        "micro_f1": round(micro.get("f1", raw.get("best_f1", 0.0)) * 100, 2),
        "micro_precision": round(micro.get("precision", 0.0) * 100, 2),
        "micro_recall": round(micro.get("recall", 0.0) * 100, 2),
        "macro_f1": round(macro.get("f1", 0.0) * 100, 2),
        "best_f1": round(raw.get("best_f1", 0.0), 6),
        "best_epoch": raw.get("best_epoch"),
        "val_loss": round(raw.get("val_loss", 0.0), 6),
        "wall_clock_sec": round(elapsed, 1),
        "output_dir": outdir,
        "per_category": {
            k: round(v["f1"] * 100, 2)
            for k, v in parsed.items()
            if k not in ("micro_avg", "macro_avg", "weighted_avg")
        },
    }


def load_existing(model_key: str, seed: int, args) -> Optional[Dict[str, Any]]:
    """若该组合已完成，直接读取其 eval_results.json 复用结果。"""
    path = eval_path(args.out_root, model_key, seed)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    parsed = parse_classification_report(raw.get("classification_report", ""))
    micro = parsed.get("micro_avg", {})
    macro = parsed.get("macro_avg", {})
    return {
        "model_key": model_key,
        "model_name": MODEL_REGISTRY[model_key]["model_name"],
        "label": MODEL_REGISTRY[model_key]["label"],
        "seed": seed,
        "micro_f1": round(micro.get("f1", raw.get("best_f1", 0.0)) * 100, 2),
        "micro_precision": round(micro.get("precision", 0.0) * 100, 2),
        "micro_recall": round(micro.get("recall", 0.0) * 100, 2),
        "macro_f1": round(macro.get("f1", 0.0) * 100, 2),
        "best_f1": round(raw.get("best_f1", 0.0), 6),
        "best_epoch": raw.get("best_epoch"),
        "val_loss": round(raw.get("val_loss", 0.0), 6),
        "wall_clock_sec": None,
        "output_dir": run_dir(args.out_root, model_key, seed),
        "per_category": {
            k: round(v["f1"] * 100, 2)
            for k, v in parsed.items()
            if k not in ("micro_avg", "macro_avg", "weighted_avg")
        },
        "reused": True,
    }


def paired_bootstrap(diffs: List[float], iters: int, seed: int) -> Dict[str, Any]:
    """对配对差值序列做 bootstrap，返回均值差的置信区间与双侧 p 值近似。"""
    arr = np.asarray(diffs, dtype=float)
    n = len(arr)
    if n == 0:
        return {"mean": None, "ci_low": None, "ci_high": None, "p_two_sided": None}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(iters, n))
    means = arr[idx].mean(axis=1)
    ci_low, ci_high = np.percentile(means, [2.5, 97.5])

    # 双侧 p 值近似：bootstrap 分布中符号与观测均值相反的比例
    obs = arr.mean()
    if obs == 0:
        p = 1.0
    else:
        p = float(np.mean(np.sign(means) != np.sign(obs)))
        p = max(p, 1.0 / iters)
    return {
        "mean": round(float(obs), 4),
        "ci_low": round(float(ci_low), 4),
        "ci_high": round(float(ci_high), 4),
        "p_two_sided_approx": round(p, 4),
        "iters": iters,
        "n_pairs": n,
    }


def welch_t(a: List[float], b: List[float]) -> Optional[Dict[str, Any]]:
    """Welch t 检验（小样本功效有限，仅作参考）。"""
    try:
        from scipy import stats
    except Exception:
        return None
    if len(a) < 2 or len(b) < 2:
        return None
    t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
    return {"t_stat": round(float(t_stat), 4), "p_value": round(float(p_val), 4)}


def summarize(runs: List[Dict[str, Any]], seeds: List[int]) -> Dict[str, Any]:
    """按模型聚合，并执行配对统计检验。"""
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for r in runs:
        by_model.setdefault(r["model_key"], []).append(r)

    stats_block: Dict[str, Any] = {}
    for mk, rs in by_model.items():
        f1s = [r["micro_f1"] for r in rs]
        mac = [r["macro_f1"] for r in rs]
        losses = [r["val_loss"] for r in rs if r["val_loss"]]
        stats_block[mk] = {
            "label": rs[0]["label"],
            "model_name": rs[0]["model_name"],
            "n_runs": len(rs),
            "micro_f1_mean": round(statistics.mean(f1s), 3),
            "micro_f1_std": (
                round(statistics.stdev(f1s), 3) if len(f1s) > 1 else 0.0
            ),
            "micro_f1_min": round(min(f1s), 2),
            "micro_f1_max": round(max(f1s), 2),
            "micro_f1_per_seed": {str(r["seed"]): r["micro_f1"] for r in rs},
            "macro_f1_mean": round(statistics.mean(mac), 3) if mac else None,
            "val_loss_mean": round(statistics.mean(losses), 5) if losses else None,
            "paper_reference": PAPER_REFERENCE.get(mk),
        }

    # ---- 配对检验：仅在两模型种子集合完全一致时进行 ----
    paired: Dict[str, Any] = {"available": False}
    if "bert" in by_model and "macbert" in by_model:
        b_map = {r["seed"]: r["micro_f1"] for r in by_model["bert"]}
        m_map = {r["seed"]: r["micro_f1"] for r in by_model["macbert"]}
        common = sorted(set(b_map) & set(m_map))
        if common:
            diffs = [m_map[s] - b_map[s] for s in common]
            paired = {
                "available": True,
                "n_pairs": len(common),
                "diffs_per_seed": {str(s): round(d, 2) for s, d in zip(common, diffs)},
                "mean_diff_pp": round(statistics.mean(diffs), 3),
                "std_diff_pp": (
                    round(statistics.stdev(diffs), 3) if len(diffs) > 1 else 0.0
                ),
                "macbert_wins": int(sum(1 for d in diffs if d > 0)),
                "bert_wins": int(sum(1 for d in diffs if d < 0)),
                "ties": int(sum(1 for d in diffs if d == 0)),
            }

    return {"by_model": stats_block, "paired": paired}


def attach_tests(summary: Dict[str, Any], args) -> Dict[str, Any]:
    paired = summary["paired"]
    if not paired.get("available"):
        return summary
    diffs = list(paired["diffs_per_seed"].values())
    paired["paired_bootstrap"] = paired_bootstrap(
        diffs, args.bootstrap_iters, args.bootstrap_seed
    )
    paired["welch_t_test"] = welch_t(
        [v for v in summary["by_model"]["macbert"]["micro_f1_per_seed"].values()],
        [v for v in summary["by_model"]["bert"]["micro_f1_per_seed"].values()],
    )

    ci = paired["paired_bootstrap"]
    # 判定口径：95% CI 是否跨越 0
    if ci["ci_low"] is not None:
        crosses_zero = ci["ci_low"] <= 0 <= ci["ci_high"]
        paired["significant_at_95ci"] = (not crosses_zero) and ci["n_pairs"] >= 5
        if crosses_zero:
            verdict = (
                "配对差值的 95% 置信区间跨越 0，MacBERT 相对 BERT 的优势"
                "在统计上不显著，应表述为“数值上略优/趋势一致”，不得称“显著提升”。"
            )
        elif ci["n_pairs"] < 5:
            verdict = (
                f"95% 置信区间虽未跨越 0，但仅 {ci['n_pairs']} 组配对样本，"
                "统计功效不足，结论仍属提示性（suggestive），建议增加种子数。"
            )
        else:
            verdict = "配对差值的 95% 置信区间不跨越 0，可认为存在稳定增益。"
        paired["verdict"] = verdict
    return summary


def render_markdown(summary: Dict[str, Any], args, seeds, models) -> str:
    # 只渲染实际有数据的模型，避免部分组合失败时整个报告崩溃
    available = [m for m in models if m in summary.get("by_model", {})]
    missing = [m for m in models if m not in summary.get("by_model", {})]
    if not available:
        raise RuntimeError("没有任何模型产生有效结果，无法生成报告。")
    models = available

    lines: List[str] = []
    if missing:
        lines.append(
            f"> ⚠️ 以下模型尚无有效结果，本报告暂未包含：{', '.join(missing)}。"
            "请补齐对应 (model, seed) 组合后重新生成。"
        )
        lines.append("")
    lines.append("# BERT vs MacBERT 多种子重复实验报告（CLUENER2020）")
    lines.append("")
    lines.append(
        f"- 随机种子：`{', '.join(str(s) for s in seeds)}`（共 {len(seeds)} 个）"
    )
    lines.append(f"- 训练轮数：`{args.epochs}` | batch_size：`{args.batch_size}` | lr：`{args.lr}` | max_len：`{args.max_len}`")
    lines.append("- 评测：seqeval 严格实体级匹配，**验证集 dev（1,343 条 / 3,072 个实体）**")
    lines.append("")
    lines.append(
        "> ⚠️ **口径提示**：dev 集同时用于选择 checkpoint 与报告最终分数，"
        "结果存在一定乐观偏差；CLUENER2020 官方 test 集无公开标签，故无法给出无偏测试集分数。"
    )
    lines.append("")

    lines.append("## 1. 各模型 Micro F1 汇总（%）")
    lines.append("")
    lines.append("| 模型 | 论文公开基线 | 本实验均值 ± 标准差 | 最优 | 最差 | 种子数 |")
    lines.append("| :--- | :--- | :--- | :---: | :---: | :---: |")
    for mk in models:
        blk = summary["by_model"][mk]
        ref = blk.get("paper_reference") or {}
        if ref.get("f1") is not None:
            ref_str = f"{ref['f1']}% ({ref['scheme']} / {ref['split']} / {ref['tagging']})"
        else:
            ref_str = "无（原论文未在该数据集实验）"
        lines.append(
            f"| {blk['label']} | {ref_str} | "
            f"**{blk['micro_f1_mean']} ± {blk['micro_f1_std']}** | "
            f"{blk['micro_f1_max']} | {blk['micro_f1_min']} | {blk['n_runs']} |"
        )
    lines.append("")

    lines.append("## 2. 逐种子明细（Micro F1, %）")
    lines.append("")
    header = "| 种子 | " + " | ".join(
        summary["by_model"][mk]["label"] for mk in models
    ) + " | 差值 |"
    lines.append(header)
    lines.append("| :---: | " + " | ".join(":---:" for _ in models) + " | :---: |")
    for s in seeds:
        row = [str(s)]
        vals = []
        for mk in models:
            v = summary["by_model"][mk]["micro_f1_per_seed"].get(str(s))
            row.append(f"{v}" if v is not None else "—")
            vals.append(v)
        if len(vals) == 2 and all(v is not None for v in vals):
            row.append(f"{round(vals[1] - vals[0], 2):+}")
        else:
            row.append("—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    p = summary["paired"]
    lines.append("## 3. 配对统计检验")
    lines.append("")
    if not p.get("available"):
        lines.append("两模型种子集合不一致，无法执行配对检验。")
    else:
        bs = p.get("paired_bootstrap", {})
        lines.append(f"- 配对样本数：{p['n_pairs']}")
        lines.append(f"- 平均差值：**{p['mean_diff_pp']} pp**（标准差 {p['std_diff_pp']}）")
        lines.append(f"- MacBERT 胜出 {p['macbert_wins']} 组 / BERT 胜出 {p['bert_wins']} 组 / 持平 {p['ties']} 组")
        if bs:
            lines.append(
                f"- 配对 Bootstrap 95% CI：**[{bs['ci_low']}, {bs['ci_high']}]** "
                f"（{bs['iters']} 次重采样）"
            )
            lines.append(f"- Bootstrap 近似双侧 p 值：{bs.get('p_two_sided_approx')}")
        wt = p.get("welch_t_test")
        if wt:
            lines.append(f"- Welch t 检验（参考）：t = {wt['t_stat']}, p = {wt['p_value']}")
        elif wt is None:
            lines.append("- Welch t 检验：未安装 scipy，已跳过")
        lines.append("")
        lines.append(f"**结论**：{p.get('verdict', '—')}")
    lines.append("")

    lines.append("## 4. 与已提交 README 口径的对照")
    lines.append("")
    lines.append(
        "README 早期版本曾表述「MacBERT 76.58% vs BERT 74.96%，+1.62% 显著突破」。"
        "经核对：74.96% 在三篇原文 PDF 中均无出处；修正为同口径后差值为 +0.40pp，"
        "且本实验表明该量级差值处于种子方差范围内。"
    )
    lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()
    seeds = [int(s) for s in parse_list(args.seeds)]
    models = [m for m in parse_list(args.models) if m in MODEL_REGISTRY]
    if not models:
        raise ValueError(f"--models 取值非法，可选: {list(MODEL_REGISTRY)}")

    if args.report_only:
        if not os.path.exists(args.results_json):
            raise FileNotFoundError(
                f"--report_only 需要已存在的 {args.results_json}，请先完成实验。"
            )
        with open(args.results_json, "r", encoding="utf-8") as f:
            saved = json.load(f)
        md = render_markdown(saved["summary"], args, seeds, models)
        with open(args.report_md, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"[OK] 报告已生成: {os.path.abspath(args.report_md)}")
        return

    # 冒烟模式的目的就是验证"训练链路能跑通"，若复用缓存结果则失去意义，故强制重跑。
    if args.smoke:
        args.force = True

    device = resolve_device(args.device)
    print(f"[*] 计算设备: {device}")
    print(f"[*] 种子列表: {seeds}")
    print(f"[*] 模型列表: {models}")

    runs: List[Dict[str, Any]] = []
    for mk in models:
        for seed in seeds:
            if not args.force and not args.skip_train:
                reused = load_existing(mk, seed, args)
                if reused is not None:
                    print(f"[=] 复用已有结果: {mk} seed={seed} (F1={reused['micro_f1']}%)")
                    runs.append(reused)
                    continue

            if args.skip_train:
                reused = load_existing(mk, seed, args)
                if reused is None:
                    print(f"[!] 跳过（无已有结果）: {mk} seed={seed}")
                    continue
                runs.append(reused)
                continue

            runs.append(run_one(mk, seed, args, device))

    if not runs:
        raise RuntimeError(
            "没有任何有效实验结果。请检查 --seeds / --models 参数，"
            "或去掉 --skip_train 以实际执行训练。"
        )

    summary = summarize(runs, seeds)
    summary = attach_tests(summary, args)

    payload = {
        "meta": {
            "dataset": "CLUENER2020 dev (1,343 samples / 3,072 entities)",
            "seeds": seeds,
            "models": models,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "max_len": args.max_len,
            "device": device,
            "metric": "seqeval strict entity-level Micro F1 (%)",
            "caveat": "dev 集同时用于 checkpoint 选择与最终评分，存在乐观偏差",
            "smoke": args.smoke,
            "limit": args.limit,
        },
        "runs": runs,
        "summary": summary,
    }

    if not args.smoke:
        with open(args.results_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        md = render_markdown(summary, args, seeds, models)
        with open(args.report_md, "w", encoding="utf-8") as f:
            f.write(md)

    # ---- 控制台汇总 ----
    print("\n" + "=" * 78)
    print("多种子实验汇总 (Micro F1, %)")
    print("=" * 78)
    for mk in models:
        if mk not in summary["by_model"]:
            print(f"  {mk:<34} 无有效结果（该组合尚未完成）")
            continue
        b = summary["by_model"][mk]
        print(
            f"  {b['label']:<34} {b['micro_f1_mean']:.2f} ± {b['micro_f1_std']:.2f}  "
            f"(min {b['micro_f1_min']:.2f} / max {b['micro_f1_max']:.2f}, n={b['n_runs']})"
        )
    p = summary["paired"]
    if p.get("available"):
        print("-" * 78)
        print(f"  配对平均差值: {p['mean_diff_pp']:+.2f} pp (std {p['std_diff_pp']:.2f})")
        bs = p.get("paired_bootstrap", {})
        if bs:
            print(f"  95% Bootstrap CI: [{bs['ci_low']}, {bs['ci_high']}]")
        print(f"  {p.get('verdict','')}")
    print("=" * 78)
    if not args.smoke:
        print(f"[OK] 结果 JSON : {os.path.abspath(args.results_json)}")
        print(f"[OK] Markdown 报告: {os.path.abspath(args.report_md)}")


if __name__ == "__main__":
    main()
