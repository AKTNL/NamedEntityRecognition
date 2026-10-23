# Evidence audit — 2026-10-23

## Sources and scope
- `saved_models/{bert,macbert}/eval_results.json`: 4 epochs, batch 32, LR 3e-5, max_len 128; best epoch 4 for both. F1 76.045863% / 76.606286%. No seed/warmup/timing/memory in these JSONs.
- CLUENER2020 PDF pages 5–6: Table 4 matches those four BERT settings; results are evaluated on test, use BIOS, and Table 5 reports Overall@Macro 77.24 / 80.46 / 78.82. Local seqeval macro row 72.59 / 79.68 / 75.95 is not a comparable replication difference.
- Read-only recount of local train/dev/test: 10748 / 1343 / 1345 sentences; 23971 / 3072 / unknown entities. Non-entity character fractions (union of annotated spans): 73.3139355% train, 73.1138082% dev. Do not treat missing test labels as zero true entities.
- Re-encoded all three splits using both saved tokenizer.json files, pretokenized character inputs, no truncation/padding: maximum 52 tokens including specials for all six model/split combinations; zero above 128.
- GPT-NER PDF first page lists Peking University, Shannon.AI, Zhejiang University, Nanyang Technological University and Amazon; page 6 gives the Hendrix location-error example.

## Generative evidence
- Shared retrieval experiment: dev first 120 sentences, 1200 queries, 274 gold entities, k=8, seed=42, batch=64, max_new_tokens=72, zh_mrc, no chat template or self-verification. Logs record no LoRA and neg_ratio=0.5. F1 random/sentence/entity = 7.46 / 27.55 / 33.57. Rounded malformed rates = 45.17 / 29.00 / 30.58%; hallucination count ratios = 7.17 / 4.25 / 7.00%.
- Full-dev LoRA JSON plus `logs/lora_eval_full.log`: 1343 sentences, 13430 queries, P/R/F1 43.12 / 40.17 / 41.59, 20 malformed queries, 27 unlocatable entity occurrences. 6184 pre-conflict candidates versus 2862 final entities summed from per_type.predicted; gold 3072.
- Historical LoRA first 100: P/R/F1 43.12 / 40.69 / 41.87, malformed=0, unlocatable=1. Separate from full dev.
- Early random 4-shot 5.13% uses 10 sentences, sentence 4-shot 25.00% uses 5. Not a matched experiment.
- `gptner_sentence_k0_n100.json` reports 42.37 but does not record adapter identity; do not infer a no-LoRA baseline merely from the filename.
- Current result config omits model/adapter paths, neg_ratio and revision. Logs supply some identities, not all provenance.
- All saved self_verify values are false.

## Evaluation semantics
`generative/evaluate.py`: regex extraction, original-text find, drop/count unlocatable entities, longest-first conflict resolution, BIO/seqeval scoring. Hallucination counter increments per unlocatable entity, denominator is number of queries; it is not a unique-query probability. Repeated strings may map to the first occurrence because add_prediction does not share used positions. n_pred_entities is before conflict resolution; per_type predicted is after it. These are limitations to disclose, not code changes in this task.

## Training log
`saved_models/gptner_lora/train_args.json` and `logs/lora_train.log`: 6000 per-type queries, 2058 positive/3942 negative, 1 epoch/375 updates, batch4/accum4, max_len512, seed42, r16/alpha32/dropout.05, lr1e-4. Trainable 4,358,144 / 1,548,072,448 = 0.2815%. Logged average losses at steps 50/100/200/350 = .6336/.0646/.0465/.0338. Last logged cumulative allocated memory peak 4.75 GB; training completion time 43.7 minutes, not the unsupported 22.4 in the draft.

## Spec review
Existing `.trellis/spec/frontend/dual-paradigm-contract.md` already requires source/run scope, rate unit conversion, shared-120 retrieval comparison and forbids causal claims across configurations. No code or new system contract was introduced; no spec change required. No training runs or implementation tests are needed for these Markdown-only corrections.
