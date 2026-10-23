# -*- coding: utf-8 -*-
"""
GPT-NER 生成式范式：数据格式转换与解析工具
=========================================================================
本模块实现 CLUENER2020 的 span 标注与 GPT-NER 的 `@@实体##` 标记文本之间的
**双向无损转换**，并提供与抽取式链路完全一致的 BIO 标签链生成函数，
以保证两种范式最终接入同一套 seqeval 严格实体级评测。

设计要点
--------
1. **不依赖 torch**：本模块为纯 Python 实现，可在无 GPU / 无深度学习框架的
   环境下运行与自检（抽取式链路的 dataset.py 依赖 torch，此处刻意解耦）。
2. **双向无损**：`mark -> parse` 的往返转换必须还原出完全相同的 span 集合，
   由 `selftest()` 在模块级强制校验。
3. **per-type 提示**：GPT-NER 对每种实体类型单独构造一次提示，因此
   `build_per_type_record()` 只标记目标类型的实体。

作者：《自然语言处理》课程综合实践 · 生成式 NER 链路
"""

import re
import json
import sys
from typing import List, Dict, Tuple, Any, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# 常量定义：与抽取式链路（dataset.py）保持严格一致
# ---------------------------------------------------------------------------

CATEGORIES: List[str] = [
    "address", "book", "company", "game", "government",
    "movie", "name", "organization", "position", "scene",
]

# 中文类型名：用于构造中文提示词，让基座 LLM 更容易理解任务
CATEGORY_ZH: Dict[str, str] = {
    "address": "地址",
    "book": "书籍",
    "company": "公司",
    "game": "游戏",
    "government": "政府机构",
    "movie": "电影",
    "name": "人名",
    "organization": "组织机构",
    "position": "职位",
    "scene": "景点",
}

# GPT-NER 标记协议
OPEN_TOKEN = "@@"
CLOSE_TOKEN = "##"

# 非贪婪匹配：@@ 与 ## 之间即为实体内容
_MARK_PATTERN = re.compile(re.escape(OPEN_TOKEN) + r"(.*?)" + re.escape(CLOSE_TOKEN))


# ---------------------------------------------------------------------------
# 基础转换：span <-> 标记文本
# ---------------------------------------------------------------------------

def cluener_label_to_spans(label: Dict[str, Any]) -> List[Tuple[int, int, str]]:
    """将 CLUENER2020 的 label 字典展开为扁平 span 列表。

    输入形如: {"name": {"叶老桂": [[9, 11]]}, "company": {"浙商银行": [[0, 3]]}}
    输出形如: [(0, 3, "company"), (9, 11, "name")]        # span 为闭区间 [start, end]
    """
    spans: List[Tuple[int, int, str]] = []
    for cat, entities in (label or {}).items():
        for _ent_name, span_list in entities.items():
            for start, end in span_list:
                spans.append((int(start), int(end), cat))
    # 按起点升序、终点降序排序，便于后续贪心去重
    spans.sort(key=lambda x: (x[0], -x[1]))
    return spans


def mark_entities(
    text: str,
    spans: List[Tuple[int, int]],
    open_tok: str = OPEN_TOKEN,
    close_tok: str = CLOSE_TOKEN,
) -> str:
    """将给定 span 用标记符包裹，输出 GPT-NER 的目标序列。

    参数中的 span 为**闭区间** [start, end]（与 CLUENER2020 一致）。
    重叠或越界的 span 会被安全跳过，避免产生嵌套标记。
    """
    if not spans:
        return text

    ordered = sorted(spans, key=lambda x: (x[0], x[1]))
    parts: List[str] = []
    cursor = 0
    prev_end = -1

    for start, end in ordered:
        # 越界 / 非法 / 与已标记区间重叠 -> 跳过（保持输出结构合法）
        if start < 0 or end >= len(text) or start > end or start <= prev_end:
            continue
        parts.append(text[cursor:start])
        parts.append(open_tok + text[start:end + 1] + close_tok)
        cursor = end + 1
        prev_end = end

    parts.append(text[cursor:])
    return "".join(parts)


def parse_marked_text(
    marked: str,
    open_tok: str = OPEN_TOKEN,
    close_tok: str = CLOSE_TOKEN,
) -> Tuple[str, List[Dict[str, Any]]]:
    """解析标记文本，还原出原文与实体 span。

    返回:
        clean_text : 去掉所有标记符后的原文（应与输入文本完全一致）
        entities   : [{"start": int, "end": int, "entity": str}, ...]  # 闭区间
    """
    pattern = re.compile(re.escape(open_tok) + r"(.*?)" + re.escape(close_tok))
    clean_parts: List[str] = []
    entities: List[Dict[str, Any]] = []
    cursor = 0
    offset = 0

    for m in pattern.finditer(marked):
        clean_parts.append(marked[cursor:m.start()])
        offset += len(marked[cursor:m.start()])

        ent = m.group(1)
        if ent:  # 空实体（@@##）直接丢弃
            start = offset
            offset += len(ent)
            entities.append({"start": start, "end": offset - 1, "entity": ent})
            clean_parts.append(ent)

        cursor = m.end()

    clean_parts.append(marked[cursor:])
    return "".join(clean_parts), entities


def spans_to_bio(text_len: int, spans: List[Tuple[int, int, str]]) -> List[str]:
    """将 span 列表转换为 BIO 标签链（与 dataset.py 的逻辑严格一致）。

    这是保证生成式链路与抽取式链路**共用同一套 seqeval 评测**的关键桥梁。
    """
    tags = ["O"] * text_len
    for start, end, cat in sorted(spans, key=lambda x: (x[0], -x[1])):
        if start < 0 or start >= text_len:
            continue
        tags[start] = f"B-{cat}"
        for i in range(start + 1, min(end + 1, text_len)):
            tags[i] = f"I-{cat}"
    return tags


# ---------------------------------------------------------------------------
# 样本级转换
# ---------------------------------------------------------------------------

def build_per_type_record(
    text: str,
    label: Dict[str, Any],
    entity_type: str,
) -> Dict[str, Any]:
    """构造单条单类型的 GPT-NER 训练/推理样本。

    返回:
        {
          "text": 原文,
          "entity_type": 目标类型,
          "entity_type_zh": 中文类型名,
          "target": 带标记的目标输出（模型应生成的内容）,
          "gold_spans": 该类型的金标 span 列表（闭区间）
        }
    """
    all_spans = cluener_label_to_spans(label)
    gold = [(s, e, c) for s, e, c in all_spans if c == entity_type]
    plain = [(s, e) for s, e, _ in gold]
    return {
        "text": text,
        "entity_type": entity_type,
        "entity_type_zh": CATEGORY_ZH.get(entity_type, entity_type),
        "target": mark_entities(text, plain),
        "gold_spans": gold,
    }


def parse_prediction(marked: str, text: str, entity_type: str) -> List[Tuple[int, int, str]]:
    """将模型生成的标记文本解析回 span，并做一次原文合法性校验。

    生成式方法的典型失效模式是"幻觉"——生成原文中不存在的字符串。
    此处通过 `text.find()` 定位实体在原文中的真实位置；若找不到，
    则以 (-1, -1) 标记该实体为**幻觉候选**，交由自验证环节处理。
    """
    clean, entities = parse_marked_text(marked)
    spans: List[Tuple[int, int, str]] = []
    for ent in entities:
        name = ent["entity"]
        # 优先信任解析出的偏移（若与原文吻合），否则回退到字符串查找
        s, e = ent["start"], ent["end"]
        if not (0 <= s <= e < len(clean)) or clean[s:e + 1] != name:
            idx = clean.find(name)
            if idx < 0:
                spans.append((-1, -1, entity_type))  # 幻觉候选
                continue
            s, e = idx, idx + len(name) - 1
        spans.append((s, e, entity_type))
    return spans


# ---------------------------------------------------------------------------
# 自检：往返一致性 + 与 seqeval 输入的兼容性
# ---------------------------------------------------------------------------

def selftest(verbose: bool = True) -> bool:
    """往返一致性自检。任何一项失败即返回 False。"""
    cases = [
        {
            "text": "浙商银行企业信贷部叶老桂博士则从另一个角度对五道门槛进行了解读。",
            "label": {"name": {"叶老桂": [[9, 11]]}, "company": {"浙商银行": [[0, 3]]}},
        },
        {
            "text": "目前主赞助商暂时空缺，他们的球衣上印的是“unicef”（联合国儿童基金会），是公益性质的广告；",
            "label": {"organization": {"unicef": [[21, 26]], "联合国儿童基金会": [[29, 36]]}},
        },
        {"text": "这句话里没有任何实体。", "label": {}},
    ]

    ok = True
    for i, case in enumerate(cases, 1):
        text, label = case["text"], case["label"]
        spans = cluener_label_to_spans(label)

        # 1) per-type 往返
        for cat in CATEGORIES:
            rec = build_per_type_record(text, label, cat)
            clean, parsed = parse_marked_text(rec["target"])
            recovered = [(e["start"], e["end"], cat) for e in parsed]
            gold = [(s, e, c) for s, e, c in spans if c == cat]
            if clean != text:
                print(f"  [FAIL] case{i}/{cat}: 去标记后原文不一致")
                ok = False
            if recovered != gold:
                print(f"  [FAIL] case{i}/{cat}: span 不一致 {recovered} != {gold}")
                ok = False

        # 2) 全类型合并标记的往返
        marked = mark_entities(text, [(s, e) for s, e, _ in spans])
        clean, parsed = parse_marked_text(marked)
        recovered = [(e["start"], e["end"], e["entity"]) for e in parsed]
        gold_named = [(s, e, text[s:e + 1]) for s, e, _ in spans]
        if clean != text or recovered != gold_named:
            print(f"  [FAIL] case{i}/ALL: 往返不一致")
            ok = False

        # 3) BIO 标签链与 dataset.py 逻辑对齐
        tags = spans_to_bio(len(text), spans)
        if len(tags) != len(text):
            print(f"  [FAIL] case{i}: BIO 长度不匹配")
            ok = False

        if verbose:
            print(f"  case{i}: {text[:30]}... -> {recovered}")

    if verbose:
        print("  [OK] 往返一致性自检通过" if ok else "  [ERROR] 自检未通过")
    return ok


def scan_marker_collision(file_path: str) -> Dict[str, Any]:
    """扫描数据集中原文与标记符的冲突情况，并区分风险等级。

    风险分级：
        - **高危 (high)**：原文含连续 `@@` 或 `##`，会与标记协议产生真实歧义，
          导致 `parse_marked_text()` 解析出错误的实体边界。
        - **低危 (low)**：仅含单个 `@` 或 `#`。由于解析正则是 `@@(.*?)##`，
          必须成对出现才触发，单个字符不构成歧义。

    返回: {"total": int, "high": int, "low": int, "high_samples": [text, ...]}
    """
    total = high = low = 0
    high_samples: List[str] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            text = json.loads(line)["text"]
            if OPEN_TOKEN in text or CLOSE_TOKEN in text:
                high += 1
                high_samples.append(text)
            elif "@" in text or "#" in text:
                low += 1
    return {"total": total, "high": high, "low": low, "high_samples": high_samples}


if __name__ == "__main__":
    print("=" * 78)
    print("GPT-NER 格式转换模块 · 自检")
    print("=" * 78)
    selftest()

    print("\n标记符冲突扫描（原文是否与 @@ / ## 协议冲突）：")
    for split in ("train", "dev"):
        path = f".data/cluener/{split}.json"
        try:
            r = scan_marker_collision(path)
            print(f"  {split}.json: 总 {r['total']} 条 | 高危(含连续@@/##) {r['high']} 条 "
                  f"| 低危(仅单字符) {r['low']} 条")
            for t in r["high_samples"][:3]:
                print(f"      [需处理] {t[:60]}")
        except FileNotFoundError:
            print(f"  {split}.json: 文件不存在，跳过")

    print("\n标记示例：")
    demo_text = "浙商银行企业信贷部叶老桂博士则从另一个角度对五道门槛进行了解读。"
    demo_label = {"name": {"叶老桂": [[9, 11]]}, "company": {"浙商银行": [[0, 3]]}}
    for cat in ("company", "name"):
        rec = build_per_type_record(demo_text, demo_label, cat)
        print(f"  [{cat}] {rec['target']}")
