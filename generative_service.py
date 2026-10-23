"""Bounded, lazy inference using the project's trained Qwen + LoRA pipeline."""

import importlib.util
import threading
import time
from pathlib import Path

from generative.gptner_format import CATEGORIES, CATEGORY_ZH, parse_marked_text
from generative.prompt_builder import build_extraction_prompt


class GenerativeServiceError(Exception):
    def __init__(self, message, code="unavailable", status=503):
        super().__init__(message)
        self.code = code
        self.status = status


def validate_request(data):
    """Validate before touching model weights; offsets reference unmodified input."""
    def invalid(message):
        raise GenerativeServiceError(message, "invalid_request", 400)

    if not isinstance(data, dict):
        invalid("请求体必须是 JSON 对象。")
    text = data.get("text")
    if not isinstance(text, str) or not text.strip():
        invalid("请输入非空的中文文本。")
    if len(text) > 128:
        invalid("单次最多识别 128 个字符，请缩短文本。")
    if any(char in text for char in ("\n", "\r", "\x00")):
        invalid("当前模型使用单句续写，请输入不含换行的单行文本。")
    if "@@" in text or "##" in text:
        invalid("输入包含模型保留的 @@ 或 ## 标记，请移除这些标记后重试。")
    categories = data.get("categories", CATEGORIES)
    if (not isinstance(categories, list) or not 1 <= len(categories) <= len(CATEGORIES)
            or any(not isinstance(item, str) or item not in CATEGORIES for item in categories)):
        invalid("categories 必须是包含 1 至 10 个合法实体类别的数组。")
    if len(set(categories)) != len(categories):
        invalid("实体类别不能重复。")
    max_new_tokens = data.get("max_new_tokens", 96)
    if type(max_new_tokens) is not int or not 32 <= max_new_tokens <= 256:
        invalid("max_new_tokens 必须是 32 至 256 之间的整数。")
    return text, [cat for cat in CATEGORIES if cat in categories], max_new_tokens


def parse_generation(text, category, output):
    """Only accept spans when removing well-formed markers reproduces the input."""
    clean, parsed = parse_marked_text(output)
    malformed = (
        output.count("@@") != len(parsed)
        or output.count("##") != len(parsed)
        or "@@" in clean or "##" in clean
        or any("@@" in item["entity"] or "##" in item["entity"] for item in parsed)
    )
    matches = clean == text
    warnings = []
    if malformed:
        warnings.append("生成标记未正确配对，该类别不输出实体。")
    if not matches:
        warnings.append("生成内容与原文不一致（可能改写、遗漏或截断），该类别不输出实体。")
    entities = []
    if matches and not malformed:
        for item in parsed:
            start, end = item["start"], item["end"]
            if text[start:end + 1] != item["entity"]:
                raise RuntimeError("Entity alignment invariant violated")
            entities.append({
                "text": item["entity"], "start": start, "end": end,
                "category": category, "category_cn": CATEGORY_ZH[category],
            })
    return {
        "category": category, "category_cn": CATEGORY_ZH[category],
        "output": output, "entities": entities, "valid": matches and not malformed,
        "malformed": malformed, "text_matches": matches, "warnings": warnings,
    }


class GenerativeNERService:
    """One cached adapter-backed model and one generation request per process."""

    def __init__(self, model_dir, lora_dir, device="auto", engine_factory=None):
        self.model_dir = Path(model_dir)
        self.lora_dir = Path(lora_dir)
        self.device = device
        self._engine_factory = engine_factory
        self._engine = None
        self._lock = threading.Lock()
        self._phase = "unloaded"
        self._last_error = None

    def _unavailable_reason(self):
        if self.device not in ("auto", "cpu", "cuda"):
            return "NER_GENERATIVE_DEVICE 仅支持 auto、cpu 或 cuda。"
        if not (self.model_dir / "config.json").is_file() or not any(self.model_dir.glob("*.safetensors")):
            return "未找到本地 Qwen 模型配置或 safetensors 权重，请检查 NER_GENERATIVE_MODEL_DIR。"
        if not (self.lora_dir / "adapter_config.json").is_file() or not (self.lora_dir / "adapter_model.safetensors").is_file():
            return "未找到训练后的 LoRA 适配器，请检查 NER_GENERATIVE_LORA_DIR。"
        if importlib.util.find_spec("peft") is None:
            return "缺少 peft 依赖，请安装 requirements.txt 中的生成式依赖。"
        return None

    def status(self):
        reason = self._unavailable_reason()
        return {
            "available": reason is None,
            "loaded": self._engine is not None,
            "busy": self._lock.locked(), "phase": self._phase,
            "device": str(self._engine.device) if self._engine else self.device,
            "model": "Qwen2.5-1.5B-Instruct", "adapter": self.lora_dir.name,
            "mode": "lora_zero_shot", "categories": CATEGORIES,
            "max_text_length": 128, "error": reason or self._last_error,
        }

    def predict(self, data):
        text, categories, max_new_tokens = validate_request(data)
        if not self._lock.acquire(blocking=False):
            raise GenerativeServiceError("生成式模型正在加载或处理其他请求，请稍后重试。", "busy", 409)
        started = time.perf_counter()
        load_ms = 0.0
        try:
            reason = self._unavailable_reason()
            if reason:
                raise GenerativeServiceError(reason)
            if self._engine is None:
                self._phase = "loading"
                load_started = time.perf_counter()
                factory = self._engine_factory
                if factory is None:
                    from generative.llm_local import LocalLLMEngine
                    factory = LocalLLMEngine
                # Both paths are mandatory. Never fall back to the untuned base.
                self._engine = factory(
                    model_dir=str(self.model_dir), lora_dir=str(self.lora_dir),
                    device=None if self.device == "auto" else self.device,
                    chat_template=False, local_files_only=True,
                )
                load_ms = (time.perf_counter() - load_started) * 1000
            self._phase = "generating"
            prompts = [build_extraction_prompt(text, category, [], lang="zh_mrc") for category in categories]
            generation_started = time.perf_counter()
            outputs = self._engine.generate(
                prompts, batch_size=2 if str(self._engine.device) == "cpu" else 10,
                max_new_tokens=max_new_tokens, verbose=False,
            )
            generation_ms = (time.perf_counter() - generation_started) * 1000
            if len(outputs) != len(categories) or any(not isinstance(output, str) for output in outputs):
                raise RuntimeError("Generation returned an invalid output count or type")
            details = [parse_generation(text, category, output) for category, output in zip(categories, outputs)]
            entities = sorted((entity for detail in details for entity in detail["entities"]),
                              key=lambda item: (item["start"], item["end"], item["category"]))
            self._last_error = None
            return {
                "text": text, "source": "live", "model": "Qwen2.5-1.5B-Instruct",
                "adapter": self.lora_dir.name, "device": str(self._engine.device),
                "mode": "lora_zero_shot", "entities": entities, "outputs": details,
                "entity_count": len(entities), "categories": categories,
                "warning_count": sum(not item["valid"] for item in details),
                "offset_unit": "unicode_codepoint", "offset_end": "inclusive",
                "config": {"k": 0, "lang": "zh_mrc", "chat_template": False,
                           "self_verify": False, "max_new_tokens": max_new_tokens},
                "timing": {"load_ms": round(load_ms, 1), "generation_ms": round(generation_ms, 1),
                           "total_ms": round((time.perf_counter() - started) * 1000, 1)},
            }
        except GenerativeServiceError as exc:
            self._last_error = str(exc)
            raise
        except Exception as exc:
            self._last_error = "模型加载或生成失败，请检查本地权重、依赖和可用内存后重试。"
            raise GenerativeServiceError(self._last_error, "inference_failed") from exc
        finally:
            self._phase = "ready" if self._engine is not None else "unloaded"
            self._lock.release()
