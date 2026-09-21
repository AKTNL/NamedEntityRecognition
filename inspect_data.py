import json

# 1. 读取第一条训练样本
with open('.data/cluener/train.json', 'r', encoding='utf-8') as f:
    sample = json.loads(f.readline())

text = sample['text']
labels = ['O'] * len(text)

# 2. 根据实体区间标注 BIO
for category, entities in sample.get('label', {}).items():
    for ent_name, span_list in entities.items():
        for start, end in span_list:
            labels[start] = f'B-{category}'
            for i in range(start + 1, end + 1):
                labels[i] = f'I-{category}'

# 3. 打印展示
print("【原文】:", text)
print("-" * 30)
print(f"{'字符':<6} | {'BIO 标签'}")
print("-" * 30)
for char, label in zip(text, labels):
    print(f"{char:<6} | {label}")