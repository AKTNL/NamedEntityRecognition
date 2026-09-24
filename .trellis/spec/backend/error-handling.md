# Error Handling

> How errors are handled in this project.

---

## Overview

The Flask backend (`app.py`) provides REST endpoints for real-time model inference (`/api/predict`), metrics inspection (`/api/metrics`), case study browsing (`/api/cases`), and codebase walkthrough (`/api/code`). All routes must guarantee no unhandled 500 exceptions, returning standard JSON envelopes with descriptive error messages and fallback states.

---

## Standard API Response Envelope

All API endpoints must return a consistent JSON response contract:

```json
{
  "success": true, // or false
  "error": "Descriptive error message on failure, or null",
  "data": { ... } // or endpoint-specific fields
}
```

For `/api/predict`:
- Missing or non-string `text` -> HTTP 400: `{"success": false, "error": "Missing or invalid 'text' field (string required)"}`
- Empty text after stripping -> HTTP 400: `{"success": false, "error": "Input text cannot be empty or purely whitespace"}`
- Unknown `model` parameter -> Defaults to `"both"` gracefully without throwing errors.
- Text exceeding length limit -> Safe character truncation with `"is_truncated": true` flag.

---

## Inference Error Mitigation

1. **Subword & Token Alignment**:
   - `tokenizer(text, return_offsets_mapping=True)` produces character offsets.
   - Special tokens (`[CLS]`, `[SEP]`, `[PAD]`) map to `(0, 0)` or `None` word indices, which must be ignored or mapped to `-100`.
   - Any character index out-of-range must be safely clamped to prevent `IndexError`.

2. **Isolated I- Tag Self-Healing**:
   - When decoding raw BIO predictions from model logits, if an isolated `I-XXX` tag appears without a preceding `B-XXX`, the decoder self-heals by treating it as a new entity start.

---

## Common Mistakes

- Direct indexing without bounds check when slicing multi-byte unicode or special characters.
- Crashing on unknown query parameters instead of safe fallback.
- Throwing unhandled exceptions during JSON serialization of PyTorch tensors or numpy types. Always convert to native Python primitives (`float`, `int`, `list`).

