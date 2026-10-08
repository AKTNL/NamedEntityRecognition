import json
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast

# 10 大实体类别
CATEGORIES = [
    "address", "book", "company", "game", "government",
    "movie", "name", "organization", "position", "scene"
]

# 构建 21 维标签体系：0 为 O，其余为各类的 B- 和 I-
LABELS = ["O"]
for cat in CATEGORIES:
    LABELS.append(f"B-{cat}")
    LABELS.append(f"I-{cat}")

LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}

class CluenerDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len=128, limit=None, strict_spans=False):
        """
        参数:
            file_path: CLUENER 格式的 jsonl 数据文件
            tokenizer: 分词器
            max_len: 最大序列长度
            limit: 仅取前 N 条样本（调试/冒烟测试用）
            strict_spans: 为 True 时，统计越界或非法标注 span 的数量，
                          便于完成“数据清洗过程留痕”的复现要求
        """
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []
        self.invalid_spans = 0

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if limit is not None and len(self.samples) >= limit:
                    break
                self.samples.append(json.loads(line))

        if strict_spans:
            for item in self.samples:
                n = len(item["text"])
                for _cat, entities in item.get("label", {}).items():
                    for _name, span_list in entities.items():
                        for start, end in span_list:
                            if start < 0 or end >= n or start > end:
                                self.invalid_spans += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        text = item["text"]

        # 1. 初始化原始字符标签为 "O"
        char_labels = ["O"] * len(text)
        for cat, entities in item.get("label", {}).items():
            for ent_name, span_list in entities.items():
                for start, end in span_list:
                    # 越界/非法 span 静默跳过会导致数据清洗过程无留痕，此处显式计数
                    if start < 0 or start >= len(char_labels) or end < start:
                        self.invalid_spans += 1
                        continue
                    char_labels[start] = f"B-{cat}"
                    for i in range(start + 1, min(end + 1, len(char_labels))):
                        char_labels[i] = f"I-{cat}"

        # 2. 转换为字符列表并用分词器编码
        chars = list(text)
        encoding = self.tokenizer(
            chars,
            is_split_into_words=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # 3. 将字符级标签对齐到 Tokenizer 输出（处理 [CLS], [SEP], [PAD]）
        word_ids = encoding.word_ids(batch_index=0)
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                # [CLS], [SEP], [PAD] 设为 -100，计算损失时自动忽略
                label_ids.append(-100)
            else:
                label_ids.append(LABEL2ID[char_labels[word_idx]])

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label_ids, dtype=torch.long)
        }

if __name__ == "__main__":
    # 使用预训练的中文 BERT 分词器
    print("正在加载分词器...")
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
    dataset = CluenerDataset(".data/cluener/train.json", tokenizer, max_len=32)

    first_item = dataset[0]
    print("\n【处理后的张量维度】:")
    print("input_ids 形状:", first_item["input_ids"].shape)
    print("attention_mask 形状:", first_item["attention_mask"].shape)
    print("labels 形状:", first_item["labels"].shape)

    print("\n【前 15 个 Token 对应关系预览】:")
    tokens = tokenizer.convert_ids_to_tokens(first_item["input_ids"][:15])
    labels = first_item["labels"][:15].tolist()
    for tok, lbl_id in zip(tokens, labels):
        lbl_str = ID2LABEL[lbl_id] if lbl_id != -100 else "[忽略:-100]"
        print(f"Token: {tok:<6} | Label ID: {lbl_id:<4} | Label 含义: {lbl_str}")