# Verification — 2026-10-23

## Automated
- `python build_workbench_html.py`: both outputs generated.
- `node test_frontend_logic.js`: passed.
- `node test_dual_paradigm_frontend.js`: passed. Tests generated application functions, artifact consistency, rate units, exact text span reuse, missing data, XSS escaping and case filtering.
- `python -m unittest test_workbench_verification`: all 16 passed with local BERT/MacBERT weights.
- `python -m py_compile build_workbench_html.py` and `node --check frontend/dual-paradigm.js`: passed.
- Scoped `git diff --check`: passed.

## Browser
- Static HTTP with missing APIs: embedded-data fallback and correct footer source label.
- Live Flask: metrics and cases refreshed; a new sentence produced actual live entities from both models.
- Experiment selector, P/R/F1 selector, three protocol categories, generative case shortcut, search/no matches and raw output disclosure worked.
- Overview, extraction, generation, code, benchmarks and cases navigation checked; arrow-key tab navigation checked.
- Desktop 1440px and narrow 390px layouts checked in both themes; no page-level horizontal overflow. Wide tables scroll inside their container.
- Console errors on live page: none.
- Screenshot saved outside the repository in the chat visualization directory.

## Decisions and limitations
- Generative inference was explicitly excluded by the user; this is an experiment presentation interface.
- Generated snapshots include existing working-tree experiment data. Original user changes to backend, models, data and documents were preserved.
- The stale metric test now validates API scores against current model artifacts rather than old hardcoded scores.
- Frontend source/maintenance guide and executable contract added. No dependencies installed.
- Local live preview: http://127.0.0.1:5000.
- Follow-up: the user authorized README updates, commit and push. README now covers six modules, live/offline behavior, data sources, rebuild instructions and current-versus-historical results. Frontend checks passed again; missing optional original experiment files are supported by the regression suite, and older cases APIs retain embedded generative cases.
