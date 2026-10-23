# -*- coding: utf-8 -*-
"""
GPT-NER 链路 · 步骤 4：本地开源 LLM 推理引擎
=========================================================================
替换官方实现中**付费的 OpenAI GPT-3 API**（`base_access.py` / `AccessBase`）。

为什么用 completion 模式而不是 chat 模式
---------------------------------------
官方调用的是 `text-davinci-003`，属于 **completion** 接口：把提示整段喂给模型
让其续写，不经过 chat 模板。GPT-NER 的少样本提示（示例 + `The given sentence:`
/ `The labeled sentence:`）正是为续写设计的。因此本模块默认 **不套 chat 模板**，
直接做提示续写（参数 `--chat-template` 可切换，用于额外对照实验）。

关键工程点
----------
1. **贪心解码**：官方 `temperature=0.0, top_p=1, best_of=1`，即确定性解码。
2. **左填充**：batch 生成必须左填充，否则每个序列的起始位置不一致，
   KV cache 与位置编码都会错位。
3. **换行停止**：目标输出是**一行**标注句，模型写完就会另起一行继续编造示例。
   必须在第一个换行处截断，否则后续解析会把编造内容当成预测。
4. **按长度排序组批**：提示长度差异大，排序后组批可显著减少 padding 浪费。

用法
----
    engine = LocalLLMEngine(model_dir)
    outs = engine.generate(prompts, batch_size=16, max_new_tokens=96)

作者：《自然语言处理》课程综合实践 · 生成式 NER 链路
"""

import os
import sys
import time
from typing import List, Optional, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch  # noqa: E402
from transformers import AutoTokenizer, AutoModelForCausalLM  # noqa: E402


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "Qwen2.5-1.5B-Instruct")


# 停止串：目标输出只有**一行**标注句。小模型常见的失效模式有三个，都必须拦住：
#   1. 写完一行后另起新行继续编造示例         -> "\n"
#   2. **不换行而是用空格**接着写下一个示例    -> "给定句子：" / "标注句子："
#   3. 切换到别的提示格式                      -> "输入：" / "输出："
# 实测 Qwen2.5-1.5B 在长提示下会稳定命中第 2 种，因此这组停止串是必需的，
# 只判 "\n" 会导致模型一直续写到 max_new_tokens，既浪费算力又污染解析。
DEFAULT_STOP_STRINGS = ("\n", "\r", "给定句子：", "标注句子：", "输入：", "输出：",
                        "The given sentence:", "The labeled sentence:")


class NewlineStoppingCriteria:
    """批内所有序列都已产出任意停止串（或 EOS）时停止生成。

    判定需要解码整批已生成文本，这一步并非免费：batch=16、max_new_tokens=96 时
    会产生近 1500 次解码调用，实测能占到生成耗时的相当一部分。
    因此这里**每 `check_every` 步才检查一次**，最多多生成 `check_every` 个 token，
    后处理的 `_truncate` 会把多余部分截掉，对结果无影响。
    """

    def __init__(self, tokenizer: Any, prompt_len: int,
                 stop_strings=DEFAULT_STOP_STRINGS, check_every: int = 4):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.stop_strings = tuple(stop_strings)
        self.check_every = max(1, check_every)
        self.step = 0

    def __call__(self, input_ids: "torch.Tensor", scores: "torch.Tensor", **kwargs) -> bool:
        self.step += 1
        if self.step % self.check_every != 0:
            return False
        gen = input_ids[:, self.prompt_len:]
        texts = self.tokenizer.batch_decode(gen, skip_special_tokens=True)
        for t in texts:
            if not any(s in t for s in self.stop_strings):
                return False
        return True


class LocalLLMEngine:
    """本地 Qwen2.5-1.5B-Instruct 的批量续写引擎。"""

    def __init__(
        self,
        model_dir: str = DEFAULT_MODEL_DIR,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        chat_template: bool = False,
        lora_dir: Optional[str] = None,
        local_files_only: bool = True,
    ):
        model_dir = model_dir or DEFAULT_MODEL_DIR
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(
                f"找不到本地模型目录：{model_dir}\n"
                f"请先从 ModelScope 下载：\n"
                f"  git clone https://www.modelscope.cn/Qwen/Qwen2.5-1.5B-Instruct.git {model_dir}"
            )

        self.model_dir = model_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.chat_template = chat_template

        # 8GB 显存：bf16 足够稳，且比 fp32 省一半显存
        if dtype is None:
            dtype = "bfloat16" if str(self.device).startswith("cuda") else "float32"
        self.torch_dtype = getattr(torch, dtype)

        print(f"  [引擎] 加载 {os.path.basename(model_dir)} "
              f"({dtype}, device={self.device}) ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=local_files_only)
        # 左填充：batch 生成的必需设置
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=self.torch_dtype, local_files_only=local_files_only
        )

        if lora_dir:
            from peft import PeftModel
            print(f"  [引擎] 挂载 LoRA 适配器：{lora_dir}")
            self.model = PeftModel.from_pretrained(self.model, lora_dir, local_files_only=local_files_only)

        self.model.to(self.device).eval()

        n_param = sum(p.numel() for p in self.model.parameters())
        n_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  [引擎] 就绪，参数量 {n_param / 1e9:.2f}B"
              + (f"（可训练 {n_train / 1e6:.1f}M）" if lora_dir else ""))

        if torch.cuda.is_available():
            used = torch.cuda.memory_allocated() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  [引擎] 显存占用 {used:.2f} / {total:.2f} GB")

    # ------------------------------------------------------------------

    def _apply_template(self, prompt: str) -> str:
        """可选：把提示包进 Qwen 的 chat 模板（默认不使用）。"""
        if not self.chat_template:
            return prompt
        messages = [{"role": "user", "content": prompt}]
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:  # noqa: BLE001
            return prompt

    @torch.no_grad()
    def generate(
        self,
        prompts: List[str],
        batch_size: int = 16,
        max_new_tokens: int = 96,
        stop_at_newline: bool = True,
        verbose: bool = True,
    ) -> List[str]:
        """批量续写。

        返回与 `prompts` 等长的字符串列表，已按第一个换行符截断并 strip。
        """
        results: List[Optional[str]] = [None] * len(prompts)
        # 按提示长度升序组批，减少 padding
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))

        t0 = time.time()
        for s in range(0, len(order), batch_size):
            ids = order[s:s + batch_size]
            texts = [self._apply_template(prompts[i]) for i in ids]

            enc = self.tokenizer(
                texts, padding=True, truncation=True,
                max_length=4096, return_tensors="pt"
            ).to(self.device)
            prompt_len = enc["input_ids"].shape[1]

            stopping = None
            if stop_at_newline:
                from transformers import StoppingCriteriaList
                stopping = StoppingCriteriaList([
                    NewlineStoppingCriteria(self.tokenizer, prompt_len)
                ])

            # 官方设置：temperature=0.0, top_p=1, best_of=1 → 等价贪心解码
            out = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=stopping,
                use_cache=True,
            )

            gen_ids = out[:, prompt_len:]
            decoded = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            for local, gi in enumerate(ids):
                results[gi] = self._truncate(decoded[local])

            if verbose and ((s // batch_size) % 5 == 0 or s + len(ids) >= len(order)):
                done = s + len(ids)
                el = time.time() - t0
                rate = done / el if el > 0 else 0
                eta = (len(order) - done) / rate if rate > 0 else 0
                print(f"    {done}/{len(order)} 条 | {rate:.2f} 条/秒 | 预计剩余 {eta / 60:.1f} 分钟",
                      flush=True)

        if verbose:
            print(f"    生成完毕：{len(prompts)} 条，用时 {(time.time() - t0) / 60:.2f} 分钟", flush=True)
        return [r if r is not None else "" for r in results]

    @staticmethod
    def _truncate(text: str) -> str:
        """截到最早出现的任一停止串为止，并去掉首尾空白。

        停止阶段只保证"批内都停了"，截断仍要在后处理做一次，
        因为停止串本身所在的那一步也会被解码出来。
        """
        cut = len(text)
        for s in DEFAULT_STOP_STRINGS:
            p = text.find(s)
            if 0 <= p < cut:
                cut = p
        return text[:cut].strip()


# ---------------------------------------------------------------------------
# 冒烟测试
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 78)
    print("GPT-NER 本地推理引擎 · 冒烟测试")
    print("=" * 78)

    demo_prompt = (
        "你是一名优秀的语言学专家。任务是在给定句子中标注出所有“公司”类实体。\n"
        "请用 @@ 和 ## 将实体包裹起来，句子的其余部分保持原样输出。"
        "如果该句中没有此类实体，请原样输出句子。\n"
        "下面是一些示例。\n"
        "给定句子：我校位于北京市海淀区中关村大街。\n"
        "标注句子：我校位于北京市海淀区中关村大街。\n"
        "给定句子：华为公司发布了新手机。\n"
        "标注句子：@@华为##公司发布了新手机。\n"
        "给定句子：北京时间3月15日，华为在东莞松山湖基地发布了新款折叠屏手机。\n"
        "标注句子："
    )

    engine = LocalLLMEngine()
    outs = engine.generate([demo_prompt, demo_prompt], batch_size=2, max_new_tokens=64)
    for i, o in enumerate(outs, 1):
        print(f"\n  输出 {i}: {o}")
