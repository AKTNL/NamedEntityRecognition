# Dual-paradigm NER workbench

## Goal
Complete the Chinese course demonstration frontend for extractive and generative NER.

## Confirmed scope
The user selected a complete presentation frontend using existing experiment data and cases. No live generative inference endpoint is required.

## Requirements
- Preserve the zero-dependency Flask / standalone HTML deployment and monochrome dark/light design.
- Add an overview with clear entry points for BERT/MacBERT extraction and GPT-NER/Qwen generation.
- Present existing generative ablations, per-category metrics, marker protocol and failure cases.
- Distinguish live inference, recorded results, dictionary simulation and illustrative protocol examples.
- Show evaluation scope; do not claim cross-sample F1 values are a controlled ranking.
- Preserve existing extraction, code walkthrough, benchmarks and custom-case workflows.
- Provide responsive layouts, keyboard navigation and clear empty/error states.

## Acceptance criteria
- [x] Both generated HTML outputs are identical and self-contained.
- [x] Overview, extraction, generation, benchmark, code and cases navigation works.
- [x] Generative experiment selection changes metrics; raw outputs and ground truth are visible in cases.
- [x] Existing data is rendered without fabricated predictions or improvements.
- [x] Offline fallback, API refresh/failure, narrow viewport and theme switching are verified.
- [x] JavaScript syntax and meaningful interaction regressions pass.

## Technical approach
Keep build_workbench_html.py as the build entry. Put new source fragments in frontend/ and inline them during build to avoid a frontend package dependency and duplicated generated edits. Reuse existing state, escaping and entity renderers. Source research found the generative API summary is older than original results. Therefore embed selected results/generative/*.json directly with configs and sample counts, and use GET /api/metrics and /api/cases only to refresh extraction metrics and cases. Preserve legacy API behavior.

## Existing work
README.md, app.py, static_offline_bundle.json, case_study_results.json, model artifacts and other paths were already dirty before this task; preserve them. The frontend source and generated pages were clean.
