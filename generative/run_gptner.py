# -*- coding: utf-8 -*-
"""
GPT-NER 链路 · 主入口：端到端跑通「检索 → 提示 → 生成 → 自验证 → seqeval 评测」
=========================================================================
把前四个模块串成一条可复现的实验流水线，产出与抽取式链路**同口径**的实测数字。

流程
----
    1. 载入 per-type 数据（convert_cluener.py 产出）
    2. 示例检索（random / sentence / entity）
    3. 构造提示（官方对齐的 zh_mrc 模板）
    4. 本地 LLM 批量生成
    5. 【可选】自验证过滤
    6. 解析 → 原文定位 → BIO → seqeval

典型用法
--------
    # 冒烟：只跑前 50 句 × 10 类型
    python generative/run_gptner.py --limit 50

    # 句子级 kNN、8 示例、全量 dev
    python generative/run_gptner.py --strategy sentence --k 8

    # 实体级 kNN + 自验证（论文完整配置）
    python generative/run_gptner.py --strategy entity --k 8 --self-verify

⚠️ 口径声明
-----------
本链路的 Micro F1 与抽取式链路（`train.py`）**共用 seqeval 的同一调用**，
因此两范式数字可直接比较。但基座是 1.5B 本地模型，而非论文所用的 GPT-3(175B)，
绝对数值必然远低于论文；差异应归因于**基座规模**，不应解读为"方法失效"。

作者：《自然语言处理》课程综合实践 · 生成式 NER 链路
"""

import os
import sys
import json
import time
import argparse
from typing import List, Dict, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gptner_format import CATEGORIES, CATEGORY_ZH  # noqa: E402
from convert_cluener import OUT_DIR  # noqa: E402
from retriever import ExampleRetriever  # noqa: E402
from prompt_builder import build_extraction_prompt  # noqa: E402
from evaluate import GptNerEvaluator, extract_entity_strings  # noqa: E402
from llm_local import LocalLLMEngine  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "generative")


# ---------------------------------------------------------------------------
# 数据载入
# ---------------------------------------------------------------------------

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def subset_by_sentence(
    dev_samples: List[Dict[str, Any]],
    limit: Optional[int],
) -> List[Dict[str, Any]]:
    """按**原句**截断（而不是按查询条数），保证每个句子都跑满 10 个类型。"""
    if not limit:
        return dev_samples
    keep = {s["sample_id"] for s in dev_samples}
    keep = sorted(keep)[:limit]
    keep_set = set(keep)
    return [s for s in dev_samples if s["sample_id"] in keep_set]


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> Dict[str, Any]:
    t_start = time.time()

    print("=" * 78)
    print("GPT-NER 生成式链路 · 端到端实验")
    print("=" * 78)
    print(f"  检索策略 : {args.strategy}")
    print(f"  示例数 k : {args.k}（负例占比 {args.neg_ratio}）")
    print(f"  LoRA     : {args.lora or '无（纯 few-shot）'}")
    print(f"  提示语言 : {args.lang}")
    print(f"  chat模板 : {args.chat_template}")
    print(f"  自验证   : {args.self_verify}")
    print(f"  评测范围 : {'全量 dev' if not args.limit else f'前 {args.limit} 句'}")
    print(f"  随机种子 : {args.seed}")

    # ---- 1. 数据 ----
    train_sent_index = load_json(os.path.join(OUT_DIR, "sentence-index.train.json"))
    dev_all = load_json(os.path.join(OUT_DIR, "mrc-ner.dev.json"))
    sent_index = load_json(os.path.join(OUT_DIR, "sentence-index.dev.json"))

    if args.types:
        types = [t.strip() for t in args.types.split(",")]
        dev_all = [s for s in dev_all if s["entity_label"] in types]
        print(f"  类型子集 : {types}")

    dev = subset_by_sentence(dev_all, args.limit)
    n_sent = len({s["sample_id"] for s in dev})

    # ⚠ 关键：金标索引必须裁到"实际参与评测的句子"。
    # 若 --limit 只跑前 N 句却拿全量 1343 句的 gold 去算，召回会被稀释 N/1343 倍，
    # 得到一个荒谬的数字（曾因此得到 Recall=0.42%）。
    eval_ids = {s["sample_id"] for s in dev}
    sent_index = [s for s in sent_index if s["sample_id"] in eval_ids]

    n_gold = sum(len(s["spans"]) for s in sent_index)
    print(f"\n[1/6] 数据：训练语料 {len(train_sent_index)} 句｜评测 {len(dev)} 条查询"
          f"（{n_sent} 个原句 × {len(CATEGORIES) if not args.types else len(types)} 类型）"
          f"｜金标实体 {n_gold} 个")

    # ---- 2. 检索 ----
    # k=0 表示零样本：不检索、不拼示例。LoRA 微调后的模型走这条路，
    # 提示长度能从 ~880 字符降到 ~100 字符，推理快 5 倍以上。
    if args.k == 0:
        print("\n[2/6] 示例检索：跳过（k=0，零样本）")
        example_demos = [[] for _ in dev]
    else:
        print(f"\n[2/6] 示例检索（{args.strategy}，负例占比 {args.neg_ratio}）...")
        retriever = ExampleRetriever(
            strategy=args.strategy, k=args.k, neg_ratio=args.neg_ratio, seed=args.seed
        ).fit(train_sent_index)
        example_demos = retriever.retrieve(dev, tag="train", k=args.k)
        # 检索用编码器已用完，及时释放显存给 LLM
        retriever.release()
        del retriever

    # ---- 3. 构造提示 ----
    print("\n[3/6] 构造提示...")
    prompts: List[str] = []
    for i, s in enumerate(dev):
        prompts.append(
            build_extraction_prompt(s["context"], s["entity_label"],
                                    example_demos[i], lang=args.lang)
        )

    if args.show_prompt:
        print("\n" + "-" * 78)
        print("【示例提示】")
        print("-" * 78)
        print(prompts[0])
        print("-" * 78)

    lens = [len(p) for p in prompts]
    print(f"  提示 {len(prompts)} 条｜字符数 平均 {sum(lens) / len(lens):.0f} / 最长 {max(lens)}")

    # ---- 4. 生成 ----
    print("\n[4/6] 本地 LLM 生成...")
    engine = LocalLLMEngine(
        model_dir=args.model_dir, chat_template=args.chat_template,
        lora_dir=args.lora,
    )
    outputs = engine.generate(
        prompts,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    # ---- 5. 自验证 ----
    kept_flags: Optional[List[List[bool]]] = None
    if args.self_verify:
        print("\n[5/6] 自验证过滤...")
        from self_verify import SelfVerifier
        verifier = SelfVerifier(engine, train_sent_index, lang=args.lang, seed=args.seed)

        flat: List[tuple] = []           # [(原文, 实体, 类型)]
        owner: List[int] = []            # flat 中每条属于第几个 query
        for qi, (s, out) in enumerate(zip(dev, outputs)):
            for e in extract_entity_strings(out):
                flat.append((s["context"], e, s["entity_label"]))
                owner.append(qi)

        print(f"  待验证候选实体 {len(flat)} 个")
        flags_flat = verifier.verify_batch(flat, verbose=True)

        # 把扁平结果按出现顺序回填到各自的 query
        kept_flags = [[] for _ in dev]
        for fi, qi in enumerate(owner):
            kept_flags[qi].append(bool(flags_flat[fi]))
        n_dropped = sum(1 for f in flags_flat if not f)
        print(f"  自验证剔除 {n_dropped} / {len(flags_flat)} 个候选"
              f"（{n_dropped / len(flags_flat) * 100:.1f}%）")
    else:
        print("\n[5/6] 自验证：跳过（未开启）")

    # ---- 6. 评测 ----
    print("\n[6/6] 解析与评测...")
    evaluator = GptNerEvaluator(sent_index)

    for qi, s in enumerate(dev):
        out = outputs[qi]
        ents = extract_entity_strings(out)
        if kept_flags is not None:
            flags = kept_flags[qi]
            ents = [e for i, e in enumerate(ents) if i < len(flags) and flags[i]]
            # 用保留下来的实体重建输出，使开关自验证两条路径走完全相同的解析代码
            evaluator.add_prediction(s["sample_id"], s["entity_label"],
                                     _rebuild_output(ents), s["context"])
        else:
            evaluator.add_prediction(s["sample_id"], s["entity_label"], out, s["context"])

    metrics = evaluator.compute()
    metrics["per_type"] = evaluator.per_type()

    # ---- 汇总 ----
    elapsed = time.time() - t_start
    result = {
        "config": {
            "strategy": args.strategy,
            "k": args.k,
            "lang": args.lang,
            "chat_template": args.chat_template,
            "self_verify": args.self_verify,
            "limit": args.limit,
            "types": args.types,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "elapsed_minutes": round(elapsed / 60, 2),
        },
        "metrics": metrics,
    }

    print("\n" + "=" * 78)
    print("结果（seqeval 实体级 Micro，与抽取式链路同口径）")
    print("=" * 78)
    print(f"  Precision = {metrics['micro_p']:.2f}")
    print(f"  Recall    = {metrics['micro_r']:.2f}")
    print(f"  Micro F1  = {metrics['micro_f1']:.2f}")
    print(f"  金标实体 {metrics['n_gold_entities']} 个｜预测实体 {metrics['n_pred_entities']} 个")
    print(f"  幻觉 {metrics['n_hallucination']} 个（率 {metrics['hallucination_rate'] * 100:.2f}%）｜"
          f"格式异常 {metrics['n_malformed']} 条（率 {metrics['malformed_rate'] * 100:.2f}%）")

    print(f"\n  逐类型 F1：")
    print(f"    {'类型':<18}{'P':>8}{'R':>8}{'F1':>8}{'金标':>8}{'预测':>8}")
    for c, v in metrics["per_type"].items():
        print(f"    {c + '(' + CATEGORY_ZH[c] + ')':<18}"
              f"{v['precision']:>8.2f}{v['recall']:>8.2f}{v['f1']:>8.2f}"
              f"{v['support']:>8}{v['predicted']:>8}")

    if metrics["hallucination_samples"]:
        print(f"\n  幻觉样例（原文中找不到）：{metrics['hallucination_samples'][:8]}")

    print(f"\n[seqeval 分类报告]\n{metrics['report']}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    name = f"gptner_{args.strategy}_k{args.k}"
    if args.self_verify:
        name += "_verify"
    if args.lora:
        name += "_lora"
    if args.limit:
        name += f"_n{args.limit}"
    out_path = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n结果已写入：{out_path}")

    return result


def _rebuild_output(ents: List[str]) -> str:
    """把保留下来的实体重新拼成 `@@e##` 串，供评测器按同一路径解析。

    这样自验证的开/关两条路径走的是**完全相同的解析与定位代码**，
    避免统计口径不一致。
    """
    return "".join(f"@@{e}##" for e in ents)


# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GPT-NER 生成式链路端到端实验")
    p.add_argument("--strategy", type=str, default="sentence",
                   choices=["random", "sentence", "entity"],
                   help="示例检索策略（论文消融：random < sentence < entity）")
    p.add_argument("--k", type=int, default=8, help="少样本示例总条数")
    p.add_argument("--neg-ratio", type=float, default=0.5,
                   help="示例中'不含目标类型'的负例占比。"
                        "dev 展开全部类型后约 83.5%% 是负例，"
                        "示例全为正例会诱导模型过预测，建议 0.3~0.7")
    p.add_argument("--lang", type=str, default="zh_mrc", choices=["zh_mrc", "zh", "en"],
                   help="提示模板：zh_mrc=官方对齐格式，zh=简化中文，en=英文原版")
    p.add_argument("--chat-template", action="store_true",
                   help="套用 Qwen chat 模板（默认关闭，走 completion 续写）")
    p.add_argument("--self-verify", action="store_true",
                   help="开启自验证过滤（论文第三个机制，默认关闭）")
    p.add_argument("--limit", type=int, default=None,
                   help="只评测前 N 个原句（每句仍跑满全部类型）")
    p.add_argument("--types", type=str, default=None,
                   help="只评测指定类型，逗号分隔，如 name,company")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model-dir", type=str, default=None,
                   help="基座模型目录，默认 models/Qwen2.5-1.5B-Instruct")
    p.add_argument("--lora", type=str, default=None,
                   help="LoRA 适配器目录（由 train_lora.py 产出），挂载后做有监督推理")
    p.add_argument("--show-prompt", action="store_true", help="打印第一条提示，便于检查")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
