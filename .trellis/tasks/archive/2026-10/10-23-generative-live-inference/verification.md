# Verification — 2026-10-23

## Automated
- `python -m unittest test_generative_api test_workbench_verification`: 31 passed.
- `node test_dual_paradigm_frontend.js`: passed, including actual live UI functions, Unicode offsets, response validation, raw-output escaping, failed-request clearing and file-mode disabling.
- `node test_frontend_logic.js`: passed; two valid script blocks, self-contained bundle.
- Python compilation and scoped `git diff --check`: passed.
- Generated Flask and standalone HTML outputs are identical (asserted by Node regression).

## Actual trained model
- Loaded local `models/Qwen2.5-1.5B-Instruct` plus `saved_models/gptner_lora` on CUDA using the existing engine; no downloads or training.
- HTTP smoke input: 腾讯公司位于深圳，马化腾担任董事会主席。
- Selecting company/address/name returned 腾讯公司 [0,3], 深圳 [6,7], 马化腾 [9,11], with all three raw outputs exactly reconstructing the input.
- Initial test server: about 4.4 s load + 1.6 s generation. Warm browser request on the final server: about 0.82 s. These are observations of a single short sample, not benchmark guarantees.
- All-ten-category browser request also completed; the model produced overlapping company/government/organization predictions and an incorrect name/position overlap. These remain visible as model outputs rather than being silently filtered. Formatting checks do not validate semantic correctness.

## Browser and running preview
- Preview runs at http://127.0.0.1:5000, with the trained LoRA cached on CUDA.
- Checked category presets, all-ten-category generation, empty-input validation, error/result state, highlighted entities and table.
- Narrow default viewport and 1440px desktop layout checked; no page overflow. Browser console errors: none.
- Screenshot saved in the chat visualization folder as generative-live-preview.jpg.
- Temporary test server on port 5001 stopped; final service uses port 5000.

## Scope and maintenance
- Live inference uses zero-shot zh_mrc completion and greedy decoding. Retrieval and self-verification were not added to the live endpoint.
- README, dependency requirements, frontend guide and backend/frontend contracts updated.
- Original model weights and unrelated staged PDF/data changes were preserved.
