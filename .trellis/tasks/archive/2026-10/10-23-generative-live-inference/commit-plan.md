# Authorized publish plan — 2026-10-23

The user explicitly requested committing and pushing the completed live generative interface.

Work commit: `feat: 接入训练后 LoRA 生成式 NER 实时推理`

Include the new inference service, its three required local Python modules (`gptner_format.py`, `prompt_builder.py`, `llm_local.py`), API routes, live frontend, generated pages, dependencies, tests, README and task/spec records.

`app.py` also contains older uncommitted experiment-summary edits. Commit only the new service import, service initialization and two live inference routes, preserving those older changes in the working tree. Use an isolated Git index so already-staged PDF changes are not included. Verify the candidate API module before committing.

Do not include model weights, model configs, datasets, raw experiment artifacts, unrelated generative training scripts, or the existing staged PDF changes. Archive this completed task in a separate bookkeeping commit, then push the scoped commits to origin/main without force.
