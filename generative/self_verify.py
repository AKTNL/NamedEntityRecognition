# -*- coding: utf-8 -*-
"""
GPT-NER 链路 · 步骤 4b：自验证（Self-Verification）
=========================================================================
论文 Figure 3 的第三个机制。动机很直接：LLM 做 NER 时普遍**过预测**
（over-prediction）——把不是实体的词也标上 `@@##`。自验证让模型再回答一次
"这个词到底是不是 X 类实体"，用 yes/no 把假阳性滤掉。

论文给出的收益：OntoNotes5.0 上 89.97 → 90.91（+0.94）。
注意这是**在小得多的基座上与过预测更强时**才明显的收益；本项目 1.5B 基座
的失效模式可能以"漏报"为主，此时自验证**可能反而降低 F1**（把仅有的正确
预测也误杀）。因此本模块默认**关闭**，由 `--self-verify` 显式开启，
作为一项消融实验来报告，而不是默认配置。

提示格式（对齐官方 `verify_results.py`）
----------------------------------------
    你是一名优秀的语言学专家。下面的任务是判断给定句子中的词语是否为
    {类型}类实体，这类实体{类型描述}。

    给定句子：{句子}
    给定句子中的"{实体}"是{类型}类实体吗？请回答 是 或 否。
    {答案}

    ...（若干示例）...

    给定句子：{待验证句}
    给定句子中的"{待验证实体}"是{类型}类实体吗？请回答 是 或 否。

示例构造：正例取该类型真实实体，负例取**其他类型**的实体（问"它是不是本类型"
答案为否），两者交替，避免模型学会"一律回答是"。

作者：《自然语言处理》课程综合实践 · 生成式 NER 链路
"""

import os
import sys
import json
import random
from typing import List, Dict, Any, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gptner_format import CATEGORIES, CATEGORY_ZH  # noqa: E402
from prompt_builder import (  # noqa: E402
    TEMPLATES,
    CATEGORY_DESC,
    build_verification_prompt,
    parse_verification_answer,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


class SelfVerifier:
    """对候选实体做一轮 yes/no 自验证。"""

    def __init__(
        self,
        engine: Any,
        train_samples: List[Dict[str, Any]],
        lang: str = "zh_mrc",
        n_pos: int = 2,
        n_neg: int = 2,
        seed: int = 42,
        batch_size: int = 32,
        max_new_tokens: int = 8,
    ):
        """
        参数:
            engine: `llm_local.LocalLLMEngine` 实例
            train_sentence_index: 训练集句级索引，用于构造固定的正/负示例
            n_pos / n_neg: 每个类型各取多少正/负示例
            max_new_tokens: 只回答 是/否，几个 token 就够
        """
        self.engine = engine
        self.lang = lang
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.demos = self._build_demos(train_sentence_index, n_pos, n_neg, seed)

    # ---------------- 示例构造 ----------------

    def _build_demos(
        self,
        train_sentence_index: List[Dict[str, Any]],
        n_pos: int,
        n_neg: int,
        seed: int,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """为每种类型构造固定的正反示例（不随查询变化，保证可复现）。

        参数:
            train_sentence_index: `sentence-index.train.json`，每项
                                  {sample_id, text, spans:[{start,end,type}]}
        """
        rng = random.Random(seed)
        by_type: Dict[str, List[Dict[str, Any]]] = {c: [] for c in CATEGORIES}
        for s in train_sentence_index:
            text = s["text"]
            if "@@" in text or "##" in text:
                continue
            for sp in s.get("spans", []):
                by_type[sp["type"]].append({
                    "sentence": text,
                    "entity": text[sp["start"]:sp["end"] + 1],
                })

        demos: Dict[str, List[Dict[str, Any]]] = {}
        for c in CATEGORIES:
            pos = by_type[c][:]
            # 负例：拿**别的类型**的实体，问它是不是 c 类 → 答案"否"
            neg_pool: List[Dict[str, Any]] = []
            for other in CATEGORIES:
                if other != c:
                    neg_pool.extend(by_type[other])

            rng.shuffle(pos)
            rng.shuffle(neg_pool)

            items: List[Dict[str, Any]] = []
            for d in pos[:n_pos]:
                items.append({**d, "answer": True})
            for d in neg_pool[:n_neg]:
                items.append({**d, "answer": False})
            rng.shuffle(items)
            demos[c] = items
        return demos

    # ---------------- 验证 ----------------

    def verify_batch(
        self,
        items: List[Tuple[str, str, str]],
        verbose: bool = True,
    ) -> List[bool]:
        """批量验证。

        参数:
            items: [(原文, 候选实体, 目标类型), ...]

        返回:
            与 items 等长的布尔列表，True 表示**保留**该实体。
        """
        if not items:
            return []

        prompts: List[str] = []
        for text, ent, typ in items:
            prompts.append(
                build_verification_prompt(
                    text=text,
                    entity=ent,
                    entity_type=typ,
                    demonstrations=self.demos.get(typ, []),
                    lang=self.lang,
                )
            )

        raws = self.engine.generate(
            prompts,
            batch_size=self.batch_size,
            max_new_tokens=self.max_new_tokens,
            verbose=verbose,
        )
        lang_key = "zh" if self.lang.startswith("zh") else "en"
        return [parse_verification_answer(r, lang=lang_key) for r in raws]


# ---------------------------------------------------------------------------
# 自检 / 演示
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 78)
    print("GPT-NER 自验证模块 · 提示结构演示")
    print("=" * 78)

    fake_train = [
        {"context": "他前往上海市浦东新区出差。", "entity_label": "address",
         "entities": ["上海市浦东新区"], "start_position": [3], "end_position": [9],
         "target": "", "sample_id": 0},
        {"context": "华为公司发布了新手机。", "entity_label": "company",
         "entities": ["华为"], "start_position": [0], "end_position": [1],
         "target": "", "sample_id": 1},
        {"context": "张三担任总经理一职。", "entity_label": "position",
         "entities": ["总经理"], "start_position": [3], "end_position": [5],
         "target": "", "sample_id": 2},
    ]
    v = SelfVerifier.__new__(SelfVerifier)      # 不加载模型，只看提示
    v.demos = v._build_demos(fake_train, n_pos=2, n_neg=2, seed=42)
    v.lang = "zh_mrc"

    for typ in ("address", "company"):
        p = build_verification_prompt(
            text="北京时间3月15日，华为在东莞松山湖基地发布了新款折叠屏手机。",
            entity="华为",
            entity_type=typ,
            demonstrations=v.demos.get(typ, []),
            lang="zh_mrc",
        )
        print(f"\n【{typ}（{CATEGORY_ZH[typ]}）】验证提示")
        print("-" * 78)
        print(p)

    print("\n回答解析：")
    for raw in ["是。", "否。", "是的，华为是公司。", "不是，华为是品牌。"]:
        print(f"  {raw!r:<24} -> 保留 = {parse_verification_answer(raw, lang='zh')}")
