# Dual-paradigm frontend contract

## 1. Scope / Trigger
The generative presentation interface consumes original experiment artifacts alongside the existing extraction API and standalone bundle.

Live inference is implemented in `frontend/generative-live.js`, inlined into the same closure. It uses independent `liveGenerationState` and `/api/generative/status` plus `/api/generative/predict`; the historical experiment selector never changes the live model. See [backend contract](../backend/generative-inference.md). Validate the live response's source, input and Unicode code-point spans before rendering; errors clear previous results and never fall back to simulation.

## 2. Signatures
- Build: `python build_workbench_html.py`.
- Read: `GET /api/metrics`, `GET /api/cases`; both return `{success, data}`.
- Predict: `POST /api/predict` with `{text, model: "both"}`.
- Original generation artifacts: `results/generative/*.json` with `{config, metrics}`.

## 3. Contracts
`bundle.experiments[]` contains `{id, label, group, source, config, metrics}`. `micro_f1`, `micro_p`, `micro_r` and `per_type.*` scores are percentages (0–100). `malformed_rate` and `hallucination_rate` are fractions (0–1), converted once for presentation. Zero is a valid score; missing values render as an em dash. `n_sentences` and `n_queries` identify evaluation scope. The original file is named in the UI and JSON export.

New sources live in `frontend/dual-paradigm.{html,css,js}` and are embedded by the builder into two identical HTML outputs. JavaScript shares the existing closure and escaping helpers; no additional dependencies or script blocks are needed. JSON embedded in HTML escapes `<`, `>` and `&` to prevent script termination.

## 4. Validation & Error Matrix
- Missing original experiment → omit that experiment; no experiments → empty state, export disabled.
- Metrics/cases HTTP error, invalid shape, or 5-second timeout → preserve corresponding embedded snapshot and label fallback.
- Extraction request failure or 30-second timeout → clearly labeled recorded/dictionary result.
- Concurrent extraction requests → only latest request renders and clears loading.
- Dictionary simulation → no confidence, no timing claims, no BIO probe.

## 5. Good / Base / Bad Cases
- Good: compare random/sentence/entity retrieval on the shared 120-sentence k=8 configuration.
- Base: show a standalone 1,343-sentence LoRA run with its configuration.
- Bad: use old aggregate API values to overwrite original experiment records, or claim a cross-configuration difference is caused only by retrieval.

## 6. Tests Required
Run `node test_dual_paradigm_frontend.js`, `node test_frontend_logic.js`, and `python -m unittest test_workbench_verification`. Test current model artifacts rather than historical literal F1 values. Verify both themes, narrow/desktop widths, keyboard tabs, live inference and offline source labels in a browser.

## 7. Wrong vs Correct
- Wrong: reuse cached spans when `text.includes(cachedText)`; start/end offsets no longer match.
- Correct: only reuse exact text / matching preset records, otherwise use labeled dictionary demonstration.
- Wrong: show `0` as missing, or display rate `0.0015` as `0.0015%`.
- Correct: use finite-number checks; render that rate as `0.15%`.
- Wrong: render entity categories or raw generation without HTML escaping.
- Correct: route external strings through the shared `escapeHtml` helper.
