# -*- coding: utf-8 -*-
"""
GPT-NER 生成式范式：提示词构造器
=========================================================================
按论文（Wang et al., 2023, arXiv:2304.10428）Figure 2 / Figure 3 的结构，
构造两类提示词：

1. **实体抽取提示**（per-type）：给定句子与一个目标实体类型，要求模型
   用 `@@实体##` 标记该类型实体后原样输出。
2. **自验证提示**：给定句子与一个候选实体，要求模型以 yes/no 回答
   该实体是否真属于目标类型，用于过滤过预测（over-prediction）。

设计要点
--------
- **纯字符串处理，不依赖 torch**：可在无深度学习框架环境下构造与检查提示词，
  便于在基座模型就位之前先行验证链路。
- **中文模板**：原论文为英文模板，本项目面向中文 CLUENER2020，
  提供 `zh` 模板；同时保留 `en` 模板以便对照论文原设置。
- **示例按类型隔离**：抽取提示中的示例只标记当前目标类型，与论文 per-type
  提示的设计一致。

作者：《自然语言处理》课程综合实践 · 生成式 NER 链路
"""

import sys
from typing import List, Dict, Any, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

if __package__:
    from .gptner_format import CATEGORIES, CATEGORY_ZH, mark_entities, cluener_label_to_spans
else:
    from gptner_format import CATEGORIES, CATEGORY_ZH, mark_entities, cluener_label_to_spans

# ---------------------------------------------------------------------------
# CLUENER2020 十类实体的中文语义描述
# ---------------------------------------------------------------------------
# 对应官方 `dataset_name.py` 中 FULL_DATA 的第二个字段（sub_prompt）。
# 官方用一句自然语言描述类型边界，例如：
#     "ORG": ("organization", "are limited to named corporate, governmental,
#              or other organizational entities")
# 中文描述同样承担"消歧"职责：CLUENER 同时存在 company / government /
# organization 三类易混类型，描述必须点明边界，否则小模型几乎必然串类。
CATEGORY_DESC: Dict[str, str] = {
    "address": "按地理或行政划分的地址，包括国家、省、市、区县、乡镇、街道、村落等",
    "book": "书籍、刊物、教材、文献等出版物的名称",
    "company": "公司、企业、集团、厂商、银行等以营利为目的的法人组织",
    "game": "电子游戏、网络游戏、棋牌游戏等游戏产品的名称",
    "government": "政府机关、行政部门、党政机构、公检法等国家权力机关",
    "movie": "电影、电视剧、纪录片、综艺等影视作品的名称",
    "name": "真实人物的姓名，也包括虚构的人物形象",
    "organization": "公司与企业之外的组织机构，包括学校、医院、社团、球队、"
                    "新闻机构、国际组织等",
    "position": "职务、职称、头衔、职业身份等表示职位的名词",
    "scene": "风景名胜区、公园、寺庙、纪念馆等可供游览的场所",
}


# ---------------------------------------------------------------------------
# 模板定义
# ---------------------------------------------------------------------------

TEMPLATES = {
    "zh": {
        "role": "你是一名优秀的语言学专家。",
        "task": "任务是在给定句子中标注出所有“{type_zh}”类实体。",
        "format": "请用 @@ 和 ## 将实体包裹起来，句子的其余部分保持原样输出。如果该句中没有此类实体，请原样输出句子。",
        "examples_header": "下面是一些示例。",
        "input_prefix": "输入：",
        "output_prefix": "输出：",
        "verify_task": "下面的任务是判断给定句子中的词语是否为“{type_zh}”类实体。",
        "verify_question": "句子中的“{entity}”是“{type_zh}”类实体吗？请回答 是 或 否。",
        "verify_input": "句子：{sentence}",
        "yes": "是",
        "no": "否",
    },
    # 官方 GPT-NER 格式的中文化版本：字段与 `get_results_mrc_knn.py` 一一对应
    #   官方 task 句： "You are an excellent linguist. Within the {dataset} dataset,
    #                   the task is to label {type} entities that {desc}.
    #                   Below are some examples, and you should make the same
    #                   prediction as the examples."
    #   官方示例句：   "The given sentence: {context}\nThe labeled sentence: {labels}"
    "zh_mrc": {
        "role": "你是一名优秀的语言学专家。",
        "task": "在 CLUENER2020 数据集中，任务是标注出{type_zh}类实体，"
                "这类实体指的是{type_desc}。",
        "format": "下面是一些示例，你需要做出与示例一致的预测。",
        "examples_header": "",
        "input_prefix": "给定句子：",
        "output_prefix": "标注句子：",
        "verify_task": "下面的任务是判断给定句子中的词语是否为{type_zh}类实体，"
                       "这类实体指的是{type_desc}。",
        "verify_question": "给定句子中的“{entity}”是{type_zh}类实体吗？请回答 是 或 否。",
        "verify_input": "给定句子：{sentence}",
        "yes": "是",
        "no": "否",
    },
    "en": {
        "role": "I am an excellent linguist.",
        "task": "The task is to label {type_en} entities in the given sentence.",
        "format": "Wrap the entities with @@ and ##, and keep the rest of the sentence unchanged.",
        "examples_header": "Below are some examples.",
        "input_prefix": "Input:",
        "output_prefix": "Output:",
        "verify_task": "The task is to verify whether the word is a {type_en} entity extracted from the given sentence.",
        "verify_question": "Is the word \"{entity}\" in the input sentence a {type_en} entity? Please answer with yes or no.",
        "verify_input": "The input sentence: {sentence}",
        "yes": "Yes",
        "no": "No",
    },
}


# ---------------------------------------------------------------------------
# 示例构造
# ---------------------------------------------------------------------------

def make_demonstration(text: str, label: Dict[str, Any], entity_type: str) -> Dict[str, str]:
    """由一条 CLUENER 样本构造单类型的 (Input, Output) 示例对。"""
    spans = cluener_label_to_spans(label)
    plain = [(s, e) for s, e, c in spans if c == entity_type]
    return {
        "input": text,
        "output": mark_entities(text, plain),
    }


# ---------------------------------------------------------------------------
# 提示词构造
# ---------------------------------------------------------------------------

def build_extraction_prompt(
    text: str,
    entity_type: str,
    demonstrations: List[Dict[str, str]],
    lang: str = "zh",
) -> str:
    """构造实体抽取提示（论文 Figure 2 / 论文正文示例结构）。

    参数:
        text: 待标注的句子
        entity_type: 目标实体类型（如 "address"）
        demonstrations: 由 `make_demonstration()` 生成的示例列表
        lang: "zh" 或 "en"
    """
    t = TEMPLATES[lang]
    type_zh = CATEGORY_ZH.get(entity_type, entity_type)
    fmt = dict(type_zh=type_zh, type_en=entity_type,
               type_desc=CATEGORY_DESC.get(entity_type, ""))

    lines = [
        t["role"],
        t["task"].format(**fmt),
        t["format"],
        t["examples_header"],
    ]
    for demo in demonstrations:
        lines.append(f'{t["input_prefix"]}{demo["input"]}')
        lines.append(f'{t["output_prefix"]}{demo["output"]}')

    lines.append(f'{t["input_prefix"]}{text}')
    lines.append(t["output_prefix"])  # 末尾留空，交由模型续写
    # 过滤空行，保持提示紧凑（zh_mrc 的 examples_header 为空）
    return "\n".join(x for x in lines if x != "")


def build_verification_prompt(
    text: str,
    entity: str,
    entity_type: str,
    demonstrations: Optional[List[Dict[str, str]]] = None,
    lang: str = "zh",
) -> str:
    """构造自验证提示（论文 Figure 3 结构）。

    参数:
        text: 原句
        entity: 待验证的候选实体字符串
        entity_type: 目标实体类型
        demonstrations: 三行式示例 [{"sentence","entity","answer"}...]，
                        为 None 时构造 zero-shot 版本（论文消融中的 zero-shot 配置）
    """
    t = TEMPLATES[lang]
    type_zh = CATEGORY_ZH.get(entity_type, entity_type)
    fmt = dict(type_zh=type_zh, type_en=entity_type,
               type_desc=CATEGORY_DESC.get(entity_type, ""))

    blocks = [t["verify_task"].format(**fmt)]
    for demo in (demonstrations or []):
        # 官方 verify_results.py 的块结构：句子 / 问题 / 答案，块间空一行
        blocks.append(
            t["verify_input"].format(sentence=demo["sentence"]) + "\n"
            + t["verify_question"].format(entity=demo["entity"], **fmt) + "\n"
            + (t["yes"] if demo.get("answer", True) else t["no"])
        )
    # 末块只有问题，留空让模型回答
    blocks.append(
        t["verify_input"].format(sentence=text) + "\n"
        + t["verify_question"].format(entity=entity, **fmt)
    )
    return "\n\n".join(blocks)


# 中文否定线索：必须先于肯定词匹配，否则“不是”会被其中的“是”误判为肯定
_ZH_NEGATIVE_CUES = ("否", "不是", "不属于", "不正确", "并非", "不算", "错误", "不对", "no")
_ZH_POSITIVE_CUES = ("是", "属于", "正确", "对", "yes")


def parse_verification_answer(raw: str, lang: str = "zh") -> bool:
    """解析自验证的回答，返回是否保留该实体。

    对无法判断的模糊回答，默认**保留**（保守策略：宁可不过滤，也不误删正确实体）。

    ⚠ 中文解析的已知陷阱：否定表达里往往含有肯定字（如“不是”含“是”），
    因此必须**先扫描否定线索**再扫描肯定线索，否则会把否定回答误判为肯定。
    """
    s = (raw or "").strip().lower()

    if lang == "en":
        if s.startswith("yes"):
            return True
        if s.startswith("no"):
            return False
        return True

    # 中文：否定优先
    for cue in _ZH_NEGATIVE_CUES:
        if cue in s:
            return False
    for cue in _ZH_POSITIVE_CUES:
        if cue in s:
            return True
    return True


# ---------------------------------------------------------------------------
# 演示
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 78)
    print("GPT-NER 提示词构造器 · 演示")
    print("=" * 78)

    demo_text = "北京时间3月15日，华为在东莞松山湖基地发布了新款折叠屏手机。"
    demo_label = {
        "company": {"华为": [[11, 12]]},
        "address": {"东莞": [[15, 16]], "松山湖": [[17, 19]]},
    }

    demos = [
        make_demonstration("他前往上海市浦东新区出差。",
                           {"address": {"上海市浦东新区": [[3, 9]]}}, "address"),
        make_demonstration("我校位于北京市海淀区中关村大街。",
                           {"address": {"北京市海淀区中关村大街": [[4, 14]]}}, "address"),
    ]

    print("\n【1】实体抽取提示（address 类型）")
    print("-" * 78)
    print(build_extraction_prompt(demo_text, "address", demos, lang="zh"))

    print("\n【2】自验证提示（few-shot，验证“东莞”是否为地址实体）")
    print("-" * 78)
    vdemos = [
        {"sentence": "他前往上海市浦东新区出差。", "entity": "上海市浦东新区", "answer": True},
        {"sentence": "华为公司发布了新手机。", "entity": "华为", "answer": False},
    ]
    print(build_verification_prompt(demo_text, "东莞", "address", vdemos, lang="zh"))

    print("\n【3】回答解析测试（中文）")
    print("-" * 78)
    for raw in ["是。", "否。", "是的，东莞是一个地名。", "不是，东莞是城市名但不是地址。", "这个问题我无法回答"]:
        print(f"  {raw!r:<28} -> 保留实体 = {parse_verification_answer(raw, lang='zh')}")

    print("\n【4】回答解析测试（英文，对照论文原设置）")
    print("-" * 78)
    for raw in ["Yes", "No", "Yes.", "I think so"]:
        print(f"  {raw!r:<28} -> 保留实体 = {parse_verification_answer(raw, lang='en')}")
