# NER Lab frontend

The application uses vanilla JavaScript and CSS with no package install or external CDN.

## Build and open

```powershell
python build_workbench_html.py
python app.py --no-browser
```

Open `http://127.0.0.1:5000` for live BERT/MacBERT inference. Open the generated `workbench.html` directly for offline demonstrations and experiment browsing.

## Source ownership

- `build_workbench_html.py`: existing extraction, benchmark, code walkthrough and case UI; build entry.
- `frontend/dual-paradigm.html`: overview and generative laboratory markup.
- `frontend/dual-paradigm.css`: responsive monochrome design additions.
- `frontend/dual-paradigm.js`: experiment views, protocol illustration, data refresh and navigation.
- `templates/index.html` and `workbench.html`: identical generated outputs; edit sources and rebuild both.

## Data and interpretation

The builder embeds `static_offline_bundle.json`, selected original `results/generative/*.json` experiments and training configuration from `saved_models/*/eval_results.json`. Optional original artifacts may be missing on another checkout: absent experiments show an empty state rather than invented scores. To publish updated generative results, rebuild the HTML after updating the source artifacts. The generated HTML carries the data and works without the original files.

On HTTP, existing `/api/metrics` and `/api/cases` endpoints refresh extraction metrics and case records. Request failures preserve the embedded data and report the source in the footer. The legacy `/api/generative_metrics` summary is deliberately not used because it mixes early experimental subsets. The original experiment files supply sample counts and run configurations alongside every score.

The generative laboratory is a presentation interface, not live generation. Its marker demonstration is a fixed, explicitly labeled teaching example. Historical case annotations are shown as recorded and may refer to older runs. Retrieval ablations use 120 sentences, 8 demonstrations and 72 generation tokens; the LoRA full-development-set result uses 1,343 sentences and a different training/decoding setup.

Extraction results distinguish live responses, exact-text recorded predictions, and dictionary demonstrations. Dictionary results have no model confidence or model timing and cannot populate the BIO probe. A partial text match must never reuse recorded entity offsets.

## Verification

```powershell
node test_frontend_logic.js
node test_dual_paradigm_frontend.js
python -m unittest test_workbench_verification
```

The Python integration suite needs the existing local model checkpoints. The new JavaScript regressions execute actual generated functions and cover artifact fidelity, percentage units, exact-span reuse, protocol offsets, escaped case output, search and missing-data behavior.
