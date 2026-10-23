# Local generative inference contract

## 1. Scope / Trigger
The browser connects to the user's trained Qwen2.5-1.5B + LoRA checkpoint. Reuse `generative/llm_local.py` and the training-compatible `build_extraction_prompt(..., [], lang="zh_mrc")`; do not silently use the base model or demonstration data.

## 2. Signatures
- `GET /api/generative/status` -> `{success: true, data: {available, loaded, busy, phase, device, model, adapter, error, categories, max_text_length}}`.
- `POST /api/generative/predict` accepts `{text, categories?, max_new_tokens?}` and returns `{success: true, data: {text, source: "live", model, adapter, device, entities, outputs, timing, config, warning_count, offset_unit, offset_end}}`.
- Errors: `{success: false, code, error}` with the HTTP status below.

## 3. Contracts
- `text`: nonblank string, 1–128 Unicode code points, single line; NUL and reserved `@@`/`##` markers rejected. Preserve the input; do not strip/truncate and then return offsets into a different string.
- `categories`: optional, defaults to all ten; otherwise 1–10 unique category strings from CLUENER. Server uses canonical category order.
- `max_new_tokens`: integer 32–256, default 96; booleans/strings rejected.
- `entities`: `[{text, start, end, category, category_cn}]`, Unicode code-point inclusive bounds. No fabricated confidence field. Preserve overlapping category predictions for inspection.
- `outputs`: one item per requested category with raw `output`, validated `entities`, `valid`, `malformed`, `text_matches`, and `warnings`.
- Remove paired markers with the shared parser; require exact reconstruction of the original text and correct marker counts. Invalid output yields no spans for that category, but remains visible. Do not call existing `parse_prediction`, whose offsets are checked against generated text. This strict live alignment differs from the historical evaluator's permissive substring locating; never rewrite historical metrics from it.
- Server-only env: `NER_GENERATIVE_MODEL_DIR` (default `models/Qwen2.5-1.5B-Instruct`), `NER_GENERATIVE_LORA_DIR` (default `saved_models/gptner_lora`), `NER_GENERATIVE_DEVICE` (`auto`, `cuda`, `cpu`). Request bodies cannot override paths.
- Load base and adapter together with `local_files_only=True`, no chat template, greedy decoding. A process-wide service lock serializes load/generation; concurrent requests are rejected, not queued. Status does not load weights.

## 4. Validation & Error Matrix
| Condition | Response |
| --- | --- |
| Invalid JSON/input/category/token limit | 400 `invalid_request`, before loading |
| Loading/generation already active | 409 `busy`, `Retry-After: 3` |
| Missing base/adapter/dependency or invalid device config | 503 `unavailable` |
| Loading/generation error or wrong output count | 503 `inference_failed`, no fallback prediction |
| Malformed/rewritten generated sentence | 200 with category warnings and no spans for that category |

Failures always release the service lock; failed initialization is retryable. Frontend status timeout is 5 seconds, prediction timeout 180 seconds. Aborting the browser request does not cancel GPU execution. Buttons remain disabled if a status refresh still reports busy.

## 5. Good / Base / Bad Cases
- Good: `@@张三##遇见@@张三##。` returns both original occurrences.
- Base: unchanged original with no markers is a valid empty result.
- Bad: `@@李四##在北京。` for input `张三在北京。` must not yield a span.

## 6. Tests Required
Run `python -m unittest test_generative_api test_workbench_verification` and both Node frontend checks. Cover lazy adapter load/reuse, lock rejection/release, missing adapter/dependency, bounded validation, generation failures, repeated occurrences, malformed/rewritten text and emoji offsets. Browser smoke tests must exercise real LoRA predictions and show actual model/device plus raw outputs.

## 7. Wrong vs Correct
- Wrong: use `text.find(entity)` to relocate an entity after the model rewrote the sentence.
- Correct: reject the category output unless removing markers exactly reconstructs the original.
- Wrong: slice JavaScript UTF-16 string using Python code-point offsets.
- Correct: validate/render slices of `Array.from(text)`.
- Wrong: show historical/dictionary output after generation fails.
- Correct: clear the live result, show an actionable error and keep experiment browsing independent.
