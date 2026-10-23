# Proposed commit

`feat: 完善抽取式与生成式 NER 双范式工作台`

Include only this task's changes:

- `README.md` (user explicitly requested its update and commit)
- `build_workbench_html.py`
- `frontend/dual-paradigm.html`
- `frontend/dual-paradigm.css`
- `frontend/dual-paradigm.js`
- `frontend/README.md`
- `templates/index.html`
- `workbench.html`
- `test_dual_paradigm_frontend.js`
- `test_workbench_verification.py`
- `.trellis/spec/frontend/index.md`
- `.trellis/spec/frontend/dual-paradigm-contract.md`
- `.trellis/tasks/10-23-dual-paradigm-frontend/` (new task metadata, requirements, verification and this plan)

## Existing changes excluded

The following predate this task and must remain untouched in the working tree/index:

- `app.py`, `case_study_results.json`, `static_offline_bundle.json`
- `saved_models/bert/{config.json,eval_results.json,tokenizer_config.json}`
- `saved_models/macbert/{config.json,eval_results.json,tokenizer_config.json}`
- Existing staged PDF addition and moves under `docs/`
- `.data/gptner/`, `GPT-NER/`, `PLAN.md`, `docs/报告1.md`, `generative/`, `logs/`, `models/`, `results/`, `saved_models/gptner_lora/`, `报告2.md`

The generated HTML intentionally embeds the current experiment snapshots; original data/model/source files are excluded from this commit. Preserve the existing staged PDF changes by committing only explicitly selected task paths. Do not use a plain commit that includes the entire existing index. Do not amend.

The user explicitly authorized README updates, commit and push on 2026-10-23. Commit these scoped changes and push to origin/main.
