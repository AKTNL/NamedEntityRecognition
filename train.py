"""
CLUENER2020 细粒度中文命名实体识别（NER）基线训练与评估脚本
支持原生 BERT (bert-base-chinese) 与 MacBERT (hfl/chinese-macbert-base) 的微调与严格实体级评测。
"""

import os
import sys
import json
import random
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    get_linear_schedule_with_warmup
)
from seqeval.metrics import f1_score, classification_report
from tqdm import tqdm

from dataset import CluenerDataset, LABEL2ID, ID2LABEL, LABELS

# 确保在国内网络环境下顺畅下载 Hugging Face 权重
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def set_seed(seed: int = 42):
    """固定随机种子，保证实验的可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Train and evaluate BERT/MacBERT on CLUENER2020 dataset.")

    # 模型与数据路径参数
    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-base-chinese",
        help="Pretrained model name or path (e.g. 'bert-base-chinese', 'hfl/chinese-macbert-base')"
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=".data/cluener/train.json",
        help="Path to training json file"
    )
    parser.add_argument(
        "--dev_file",
        type=str,
        default=".data/cluener/dev.json",
        help="Path to development/validation json file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./saved_models/bert",
        help="Directory to save the best model and tokenizer"
    )

    # 训练超参数
    parser.add_argument(
        "--max_len",
        type=int,
        default=128,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training and evaluation"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-5,
        help="Learning rate for AdamW optimizer"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Total training epochs"
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Warmup ratio for linear schedule"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for AdamW"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm for gradient clipping"
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=50,
        help="Log loss every N steps"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Only evaluate the model on dev_file without training"
    )

    return parser.parse_args()


def evaluate(model, dataloader, device):
    """
    评估模型在给定数据加载器上的表现。
    使用 seqeval 计算实体级别的 Micro F1 值，并打印分类报告。
    严格忽略 label 为 -100 的特殊/填充 token（如 [CLS], [SEP], [PAD]）。

    参数:
        model: Token 分类模型
        dataloader: 评估集 DataLoader
        device: 计算设备 (torch.device)

    返回:
        eval_loss (float): 平均损失值
        eval_f1 (float): 实体级 Micro F1
        report (str): 详细分类评估报告
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            total_loss += outputs.loss.item()
            logits = outputs.logits  # (batch_size, seq_len, num_labels)
            preds = torch.argmax(logits, dim=-1)  # (batch_size, seq_len)

            preds_cpu = preds.detach().cpu().numpy()
            labels_cpu = labels.detach().cpu().numpy()

            for p_seq, l_seq in zip(preds_cpu, labels_cpu):
                pred_list = []
                label_list = []
                for p, l in zip(p_seq, l_seq):
                    if l != -100:
                        pred_list.append(ID2LABEL[p])
                        label_list.append(ID2LABEL[l])
                all_preds.append(pred_list)
                all_labels.append(label_list)

    eval_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    eval_f1 = f1_score(all_labels, all_preds, zero_division=0)
    report = classification_report(all_labels, all_preds, digits=4, zero_division=0)

    return eval_loss, eval_f1, report


def train(args):
    """执行模型训练与周期评估闭环流程"""
    set_seed(args.seed)
    device = torch.device(args.device)
    print(f"[*] 使用计算设备: {device}")

    # 1. 加载分词器
    print(f"[*] 正在加载分词器: {args.model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # 2. 准备数据集与 DataLoader
    print(f"[*] 正在加载验证集: {args.dev_file} ...")
    dev_dataset = CluenerDataset(args.dev_file, tokenizer, max_len=args.max_len)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"[*] 验证集样本数: {len(dev_dataset)}")

    # 3. 加载预训练模型
    print(f"[*] 正在加载预训练模型: {args.model_name} ...")
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )
    model.to(device)

    # 若指定 --eval_only 则直接评估并退出
    if args.eval_only:
        print("[*] 运行独立评估模式 (--eval_only)...")
        val_loss, val_f1, report = evaluate(model, dev_dataloader, device)
        print(f"[*] 验证损失 (Val Loss): {val_loss:.4f}")
        print(f"[*] 实体级 F1 (Entity F1): {val_f1:.4f}")
        print("\n[分类报告 (Classification Report)]:")
        print(report)
        return

    # 4. 加载训练集
    print(f"[*] 正在加载训练集: {args.train_file} ...")
    train_dataset = CluenerDataset(args.train_file, tokenizer, max_len=args.max_len)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    print(f"[*] 训练集样本数: {len(train_dataset)}")

    # 5. 配置优化器与学习率 Warmup 调度器
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)

    total_steps = len(train_dataloader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    print(f"[*] 总训练步数: {total_steps} | 预热步数: {warmup_steps}")

    # 6. 训练主循环
    best_f1 = -1.0
    global_step = 0

    print("\n" + "=" * 60)
    print(f"开始训练: {args.model_name}")
    print(f"轮次: {args.epochs} | 批次大小: {args.batch_size} | 学习率: {args.lr}")
    print("=" * 60)

    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0.0
        step_loss = 0.0
        epoch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for step, batch in enumerate(epoch_iterator):
            model.train()
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            current_loss = loss.item()
            total_train_loss += current_loss
            step_loss += current_loss
            global_step += 1

            epoch_iterator.set_postfix({
                "loss": f"{current_loss:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}"
            })

            if global_step % args.logging_steps == 0:
                avg_step_loss = step_loss / args.logging_steps
                print(f"\n[Step {global_step:04d}/{total_steps}] "
                      f"Avg Loss: {avg_step_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
                step_loss = 0.0

        avg_epoch_loss = total_train_loss / len(train_dataloader)
        print(f"\n[Epoch {epoch + 1} 训练结束] 平均训练损失: {avg_epoch_loss:.4f}")

        # 验证集评估
        print(f"[*] 正在进行 Epoch {epoch + 1} 验证集评测...")
        val_loss, val_f1, report = evaluate(model, dev_dataloader, device)

        print(f"[Epoch {epoch + 1} 验证结果]")
        print(f"  验证损失 (Val Loss): {val_loss:.4f}")
        print(f"  实体级 F1 (Entity F1): {val_f1:.4f}")
        print("\n[分类报告 (Classification Report)]:")
        print(report)

        # 检查并保存更优模型
        if val_f1 > best_f1:
            best_f1 = val_f1
            print(f"[*] >>> 突破历史最佳 F1 ({best_f1:.4f})，正在保存模型至 {args.output_dir} <<<")
            os.makedirs(args.output_dir, exist_ok=True)
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

            metrics_payload = {
                "model_name": args.model_name,
                "best_f1": best_f1,
                "best_epoch": epoch + 1,
                "val_loss": val_loss,
                "max_len": args.max_len,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "epochs": args.epochs,
                "classification_report": report
            }
            with open(os.path.join(args.output_dir, "eval_results.json"), "w", encoding="utf-8") as f:
                json.dump(metrics_payload, f, indent=2, ensure_ascii=False)
            print(f"[*] 最佳模型及指标已成功持久化至 {args.output_dir}\n")

    print("\n" + "=" * 60)
    print(f"训练流程顺利结束！最优实体级验证 F1: {best_f1:.4f}")
    print(f"模型与分词器保存路径: {args.output_dir}")
    print("=" * 60)


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
