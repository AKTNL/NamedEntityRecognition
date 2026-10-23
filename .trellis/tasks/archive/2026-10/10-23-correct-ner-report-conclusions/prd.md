# Correct NER report conclusions

## Goal
Correct `docs/报告1.md` and `docs/报告2.md` against saved experiments, logs, evaluation code and the cited papers. The user explicitly requested correction after the completion audit identified unsupported comparisons.

## Scope
- Preserve the reports' teaching structure while correcting quantitative claims and conclusions.
- Distinguish paper/test/BIOS/Overall@Macro results from local dev/BIO/seqeval results.
- Describe BERT versus MacBERT as a checkpoint comparison, with a single-run numerical difference rather than statistical or causal proof.
- Use the shared 120-sentence k=8 experiments for retrieval comparison; distinguish full-dev LoRA from historical small runs.
- Explain SFT with zero prompt examples, disabled self-verification, decoding/postprocessing limitations and incomplete run metadata.
- Verify resource and dataset claims against artifacts; remove unsupported generalizations and error-cause narratives.
- Do not modify code, weights, result JSON, historical logs or unrelated work.

## Acceptance
- Every reported experiment table agrees with its explicitly cited source and sample scope.
- No cross-sample or cross-protocol claim of a causal gain, significance, paper superiority or exact reproduction remains.
- Full-dev LoRA F1 41.59%, 20 malformed queries / 13,430, 27 unlocatable entity occurrences / 13,430; historical 100-sentence LoRA F1 41.87% is kept separate.
- BERT/MacBERT F1 76.05% / 76.61%, 4 epochs, batch 32; describe +0.56 percentage points as a single-run difference.
- Markdown links, heading anchors, table numbers, units and `git diff --check` verified.

## Workflow
Codex config documents inline dispatch as the default. Trellis skills are not installed in project or user skill directories; follow the available workflow/spec files directly for this documentation-only change. No new model training or implementation tests are needed.
