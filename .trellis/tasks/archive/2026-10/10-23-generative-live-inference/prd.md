# Live generative NER

## Goal and authorization
The user explicitly requested implementing the missing generative inference interface after the presentation frontend was delivered. Connect their existing Qwen2.5-1.5B-Instruct base and trained gptner_lora adapter, including an interactive frontend and real inference verification.

## Requirements
- Reuse the local inference engine and training-compatible zh_mrc completion prompt, zero demonstrations, greedy decoding.
- Lazy-load and cache the base plus adapter; prefer available CUDA, allow CPU configuration. Never silently drop the adapter or substitute mock results.
- Provide status and prediction routes, strict bounded validation, clear missing-model/busy/inference errors and serialized generation.
- Support selected categories (default all ten), bounded output length, original raw outputs, exact original-text entity offsets and warnings for malformed or rewritten generations.
- Frontend: input, category selection, loading/error/empty states, model status, raw-output details, entity highlights and result export. Keep historical experiment views independent.
- Preserve unrelated working-tree changes. Update README, dependencies and executable contracts.

## Acceptance
- [x] Unit/integration tests cover request validation, adapter loading, concurrency, no mock fallback, repeated entity offsets, malformed and hallucinated output, Unicode offsets and frontend response handling.
- [x] A real local GPU prediction through HTTP and browser works with the trained adapter.
- [x] Existing extraction and standalone frontends still work; both generated HTML outputs match.

## Decisions
- Start with the trained zero-shot LoRA path. Retrieval and self-verification switches are out of scope because they require additional retrievers/validation and are not used by the selected trained run.
- Only exact reconstruction of the original sentence may produce spans; mismatched output remains visible with warnings. Do not use the existing parse_prediction helper, which checks the generated clean string rather than the actual original text.
- Request paths cannot override model/adapter paths. Server config uses NER_GENERATIVE_MODEL_DIR, NER_GENERATIVE_LORA_DIR and NER_GENERATIVE_DEVICE.

## Repository findings
Local base weights and adapter_model.safetensors exist. Torch CUDA is available on an RTX 5060 Laptop with 8 GB VRAM; roughly 7 GB is free at inspection. Existing extraction models run on CPU. app.py and the untracked generative/ tree contain user work to preserve.
