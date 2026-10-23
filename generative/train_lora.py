# -*- coding: utf-8 -*-
"""
GPT-NER 链路 · 补充：LoRA 有监督微调（SFT）
=========================================================================
论文里的 GPT-NER 是 **纯 few-shot 上下文学习，不训练**（GPT-3 打不起）。
本项目把基座换成 1.5B 的 Qwen2.5 后，few-shot 的绝对效果必然很低，
直接拿它跟微调过的 BERT（76.18%）比并不公平。

因此补一条 **LoRA SFT** 路线，回答一个更有意义的问题：
> 同一个生成式目标格式（`@@实体##` 标记 + per-type 提示），
> 在**有监督微调**后能达到什么水平？

这样"生成式 vs 抽取式"的对照才站得住：两者都在 CLUENER 训练集上学习过，
只是**输出范式**不同（序列标注 vs 文本生成）。

显存方案（8GB）
--------------
1.5B bf16 权重 ≈ 3.1GB + LoRA 参数与梯度（<0.1GB）+ AdamW 状态（<0.5GB）
+ 激活值。靠 **梯度检查点** + **batch=4 / 累积=4** 控制在 8GB 内。
实测若 OOM，先调小 `--batch-size` 再调 `--max-len`。

训练样本构造
------------
    输入 : 任务说明（含类型描述）+ 若干示例 + "给定句子：X\n标注句子："
    目标 : "X 的标注版本" + EOS
**只对目标 token 计算 loss**（prompt 部分 mask 成 -100），
否则模型会把力气花在学"复述提示"上。

用法
----
    python generative/train_lora.py --max-train-samples 6000 --epochs 1
    python generative/run_gptner.py --lora saved_models/gptner_lora --limit 100

作者：《自然语言处理》课程综合实践 · 生成式 NER 链路
"""

import os
import sys
import json
import time
import math
import random
import argparse
from typing import List, Dict, Any, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gptner_format import CATEGORIES, mark_entities  # noqa: E402
from convert_cluener import OUT_DIR  # noqa: E402
from prompt_builder import build_extraction_prompt  # noqa: E402
from retriever import ExampleRetriever  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch  # noqa: E402
from torch.utils.data import Dataset, DataLoader  # noqa: E402
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "Qwen2.5-1.5B-Instruct")
DEFAULT_OUT_DIR = os.path.join(PROJECT_ROOT, "saved_models", "gptner_lora")


# ---------------------------------------------------------------------------
# 数据集
# ---------------------------------------------------------------------------

class SFTDataset(Dataset):
    """per-type 的 (prompt, target) 对，target 外全部 mask。"""

    def __init__(
        self,
        train_samples: List[Dict[str, Any]],
        tokenizer: Any,
        example_idx: List[List[int]],
        max_len: int = 512,
        n_shot: int = 0,
    ):
        self.samples = train_samples
        self.tok = tokenizer
        self.example_idx = example_idx
        self.max_len = max_len
        self.n_shot = n_shot

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        s = self.samples[i]
        demos = []
        if self.n_shot:
            demos = [{"input": self.samples[j]["context"], "output": self.samples[j]["target"]}
                     for j in self.example_idx[i][:self.n_shot]]
        prompt = build_extraction_prompt(
            s["context"], s["entity_label"], demos, lang="zh_mrc"
        )
        target = s["target"]

        p_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        t_ids = self.tok(target, add_special_tokens=False)["input_ids"] + [self.tok.eos_token_id]

        # 超长时优先截 prompt 的**前部**（任务说明可压缩），保住 target 完整性
        budget = self.max_len - len(t_ids)
        if budget < 32:
            t_ids = t_ids[:max(8, self.max_len - 32)]
            budget = self.max_len - len(t_ids)
        if len(p_ids) > budget:
            p_ids = p_ids[-budget:]

        input_ids = p_ids + t_ids
        labels = [-100] * len(p_ids) + t_ids[:]
        return {"input_ids": input_ids, "labels": labels}


def collate(batch: List[Dict[str, Any]], pad_id: int) -> Dict[str, "torch.Tensor"]:
    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, mask = [], [], []
    for b in batch:
        pad = maxlen - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        labels.append(b["labels"] + [-100] * pad)
        mask.append([1] * len(b["input_ids"]) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(mask, dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(model_dir: str, lora_r: int, lora_alpha: int, lora_dropout: float):
    from peft import LoraConfig, get_peft_model

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return tokenizer, model


def main() -> None:
    ap = argparse.ArgumentParser(description="GPT-NER 生成式目标格式的 LoRA 微调")
    ap.add_argument("--model-dir", type=str, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    ap.add_argument("--max-train-samples", type=int, default=6000,
                    help="训练样本条数（per-type 展开后的条数）")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--n-shot", type=int, default=0,
                    help="训练时是否附带示例；0=零样本指令微调（省算力，推荐）")
    ap.add_argument("--neg-per-sentence", type=int, default=3,
                    help="每句抽几个'不含该类型'的负例，控制训练分布接近推理分布")
    ap.add_argument("--neg-ratio", type=float, default=0.5,
                    help="n-shot>0 时示例中负例占比")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    set_seed(args.seed)

    print("=" * 78)
    print("GPT-NER 生成式目标格式 · LoRA 微调")
    print("=" * 78)
    print(f"  基座       : {os.path.basename(args.model_dir)}")
    print(f"  训练样本   : {args.max_train_samples} 条 / epoch {args.epochs}")
    print(f"  batch      : {args.batch_size} × 累积 {args.grad_accum}")
    print(f"  LoRA       : r={args.lora_r} alpha={args.lora_alpha}")
    print(f"  训练时示例 : {args.n_shot}-shot（0 表示零样本指令）")

    sent_index = json.load(open(os.path.join(OUT_DIR, "sentence-index.train.json"),
                               encoding="utf-8"))
    rng = random.Random(args.seed)

    # 构造 per-type 训练样本，分布刻意对齐推理侧：
    #   正例 = 句中出现过的类型（要标出来）
    #   负例 = 每句再随机抽 --neg-per-sentence 个**不存在**的类型（原样输出）
    # 推理时 dev 展开全部 10 类型、约 83.5% 是负例，训练分布必须同样以负例为主，
    # 否则微调后的模型会延续"每句必须标一个"的偏置。
    all_samples: List[Dict[str, Any]] = []
    for s in sent_index:
        text = s["text"]
        if "@@" in text or "##" in text:
            continue
        present = {sp["type"] for sp in s["spans"]}
        for c in present:
            spans = [(sp["start"], sp["end"]) for sp in s["spans"] if sp["type"] == c]
            all_samples.append({"context": text, "entity_label": c,
                                "target": mark_entities(text, spans)})
        absent = [c for c in CATEGORIES if c not in present]
        rng.shuffle(absent)
        for c in absent[:args.neg_per_sentence]:
            all_samples.append({"context": text, "entity_label": c, "target": text})

    rng.shuffle(all_samples)
    picked = all_samples[:args.max_train_samples]
    n_pos = sum(1 for s in picked if s["target"] != s["context"])
    print(f"  样本池     : {len(all_samples)} 条（负例每句 {args.neg_per_sentence} 个）")
    print(f"  实际抽样   : {len(picked)} 条，其中正例 {n_pos} 条 "
          f"（{n_pos / len(picked) * 100:.1f}%）")

    # 需要示例时才构造检索（零样本下省掉这一步）
    example_demos: List[List[Dict[str, str]]] = [[] for _ in picked]
    if args.n_shot:
        print("  构造检索示例 ...")
        r = ExampleRetriever(strategy="sentence", k=args.n_shot,
                             neg_ratio=args.neg_ratio, seed=args.seed).fit(sent_index)
        example_demos = r.retrieve(picked, tag="train", k=args.n_shot)
        r.release()
        del r
        torch.cuda.empty_cache()

    tokenizer, model = build_model(args.model_dir, args.lora_r, args.lora_alpha,
                                   args.lora_dropout)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    ds = SFTDataset(picked, tokenizer, example_demos,
                    max_len=args.max_len, n_shot=args.n_shot)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
    )

    steps_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * 0.05), total_steps
    )

    print(f"\n  开始训练：{len(loader)} batch/epoch × {args.epochs} = {total_steps} 优化步")
    t0 = time.time()
    gstep = 0
    running = 0.0

    for ep in range(args.epochs):
        model.train()
        for bi, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / args.grad_accum
            loss.backward()
            running += out.loss.item()

            if (bi + 1) % args.grad_accum == 0 or bi == len(loader) - 1:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                gstep += 1

                if gstep % args.log_every == 0:
                    el = time.time() - t0
                    avg = running / (args.log_every * args.grad_accum)
                    mem = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0
                    eta = (total_steps - gstep) * el / gstep / 60
                    print(f"    step {gstep}/{total_steps} | loss {avg:.4f} | "
                          f"显存峰值 {mem:.2f}GB | 已用 {el / 60:.1f}min | 剩余约 {eta:.1f}min",
                          flush=True)
                    running = 0.0

    os.makedirs(args.out_dir, exist_ok=True)
    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    with open(os.path.join(args.out_dir, "train_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    print(f"\n  训练完成，用时 {(time.time() - t0) / 60:.1f} 分钟")
    print(f"  LoRA 适配器已保存：{args.out_dir}")


if __name__ == "__main__":
    main()
