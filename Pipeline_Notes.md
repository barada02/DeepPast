# DeepPast Data Pipeline Notes

This document explains the current data pipeline implemented in `scripts/data_layer.py`.

---

## 1) Current Scope (Minimal)

The script currently implements **2 pipelines only**:

1. **Canonical Train Pairs**
   - Source: `train.csv`
   - Output: normalized document-level transliteration/translation pairs.

2. **Sentence-Level Pair Builder**
   - Sources:
     - `train.csv` (document-level transliteration)
     - `Sentences_Oare_FirstWord_LinNum.csv` (sentence boundaries + sentence translations)
   - Output: sentence-level source-target pairs for seq2seq training.

It intentionally does **not** process `test.csv`, `published_texts.csv`, or other supplemental files in this version.

---

## 2) File Paths

Top-level editable defaults in script:

- `DEFAULT_INPUT_DIR`
- `DEFAULT_OUTPUT_DIR`

Current default values are in `scripts/data_layer.py`.

---

## 3) Canonical Normalization Rules

### Transliteration normalization (`normalize_transliteration`)

- `Ḫ` -> `H`, `ḫ` -> `h`
- `[x]` -> `<gap>`
- `[...]` or `...` -> `<big_gap>`
- subscript numbers/`ₓ` -> normal characters
- remove selected editorial symbols (`! ? / [ ] < >`)
- replace `:` and `.` with space
- collapse repeated whitespace

### Translation normalization (`normalize_translation`)

- remove `< >`
- collapse repeated whitespace

---

## 4) Split Strategy

The script uses deterministic split by hashed id:

- function: `deterministic_split_by_id`
- default validation ratio: `0.1`
- default split salt: `deep_past_split_v1`

This keeps train/valid stable across runs.

---

## 5) Sentence Pair Construction Logic

Function: `build_sentence_level_pairs`

For overlapping documents (`text_uuid == oare_id`):

1. Sort sentence markers by:
   - `first_word_obj_in_text`
   - `sentence_obj_in_text`
2. Try to locate sentence starts via **robust marker search**:
   - compare normalized marker token with normalized document tokens
   - also compare compact forms (hyphen-insensitive)
   - search sequentially through document token stream
3. If marker search fails, fallback to:
   - `first_word_obj_in_text` (word-index boundary)
4. If index fallback is missing, fallback to:
   - proportional boundary estimate by sentence position in document
5. Repair boundaries to monotonic, bounded start indices.
6. Slice document transliteration between consecutive boundaries.
7. Normalize sliced source sentence.
8. Pair with sentence translation from sentence map.

Quality helper columns are written for analysis:

- `first_word_match`
- `boundary_method` (`marker_search`, `first_word_obj_fallback`, `proportional_fallback`)
- `source_start_idx_repaired` (whether start index was adjusted during monotonic repair)

---

## 6) Outputs

Generated in output directory:

1. `canonical_train_pairs.csv`
   - document-level normalized pairs
2. `sentence_level_pairs.csv`
   - sentence-level pairs for seq2seq
3. `sentence_level_pairs_training_ready.csv`
   - sentence-level pairs after full post-alignment normalization and non-empty filtering
4. `pipeline_run_history.csv`
   - appended run log (one row per script execution) with timestamp, settings, row counts, and match metrics

Recommended immediate seq2seq training file:

- **`sentence_level_pairs_training_ready.csv`**
  - source: `source_sentence_clean`
  - target: `target_sentence_clean`

---

## 7) Current Known Limitations

1. Coverage is limited to IDs overlapping between:
   - `train.csv`
   - `Sentences_Oare_FirstWord_LinNum.csv`
2. Sentence boundaries are heuristic (marker search + fallback), not guaranteed perfect.
3. Alignment quality should be monitored using:
   - `first_word_match`
   - `boundary_method`

---

## 8) Suggested Next Improvements

1. Add confidence flags:
   - `is_high_confidence = (first_word_match == 1 and boundary_method == 'marker_search')`
2. Add stricter token matching variants for Akkadian markers.
3. Add dataset diagnostics report after each run:
   - row counts
   - match rates
   - per-document sentence count consistency
4. Add optional filtering mode:
   - high-confidence only
   - full dataset

---

## 9) Run Commands

From `scripts/` directory:

```bash
python data_layer.py
```

Optional custom paths:

```bash
python data_layer.py --input-dir "<your_input_dir>" --output-dir "<your_output_dir>"
```

Each run appends one summary row to `pipeline_run_history.csv`.

---

## 10) Quick Decision Rule

- If you want **safe first training**: use rows where `first_word_match == 1` and `boundary_method == marker_search`.
- If you want **more data quickly**: use full `sentence_level_pairs.csv`.

---

## 11) Open Problem To Revisit

Current observed issue:

- Source document-level dataset size is **1561** rows.
- Earlier generated sentence-level pairs were **396** rows (before alignment refactor).
- Current generated sentence-level pairs are substantially higher after robust fallback + boundary repair.

Why this is a problem:

- A full document->sentence conversion should usually produce **more** rows than document-level input.
- Current sentence output is a conservative subset, not full coverage.

Likely causes:

1. Limited overlap between `train.csv` ids and sentence-map ids.
2. Boundary matching misses due to token normalization/scribal notation differences.
3. Fallback boundaries may not always produce valid cuts.

Action for next iteration:

- Add stage-by-stage coverage report:
   - overlap docs count
   - marker-found count
   - fallback-used count
   - proportional-fallback count
   - repaired-boundary count
   - dropped-empty or invalid-slice count
- Improve marker matching and sentence-cut validation.

---

## 12) Publications Expansion (New)

A new stage-1 extractor script exists at `scripts/publications_extractor.py`.

Goal:

- Build deterministic candidate links from `publications.csv` (OCR pages) to tablets in `published_texts.csv`.
- Prepare LLM-ready rows for stage-2 translation extraction and cleanup.

Current matching (deterministic):

1. Extract CDLI ids (`P######`) from OCR `page_text` and `pdf_name`.
2. Match those IDs to `published_texts.csv` (`cdli_id` field).
3. Save candidate rows with confidence scores.

Outputs:

1. `publications_cdli_matches.csv`
   - deterministic page->tablet matches with confidence and context snippet
2. `publications_llm_alignment_input.csv`
   - same matches plus an `llm_task` field for stage-2 extraction prompts
3. `publications_extraction_summary.csv`
   - row counts and quick coverage stats

Run command:

```bash
python publications_extractor.py
```

Optional quick test on subset:

```bash
python publications_extractor.py --max-rows 5000
```

Planned stage-2 (LLM):

- For each deterministic candidate, ask LLM to extract the exact translation segment for that tablet from page text.
- If translation is not English, convert it to English.
- Return confidence score and keep only high-confidence outputs as silver training data.

Current full-run snapshot (with `has_akkadian=true` filter):

- `matches_rows`: 4994
- `unique_oare_ids`: 1312
- `unique_pdfs`: 198
- `llm_shortlist_rows`: 4902

Stage-2 template file:

- `scripts/llm_stage2_prompt_template.md`

Stage-2 runner script:

- `scripts/stage2_llm_runner.py`

Required env:

- `DO_AI_API_KEY` in `.env` (see `.env.example`)

Stage-2 outputs:

1. `data/processed/llm_stage2/llm_raw_responses.jsonl`
   - raw API outputs + parsed JSON + token usage + errors
2. `data/processed/llm_stage2/silver_sentence_pairs.csv`
   - one-to-many flattened sentence pairs
3. `data/processed/llm_stage2/llm_stage2_run_summary.csv`
   - run-level metrics

Run command (small test first):

```bash
python stage2_llm_runner.py --max-rows 20 --rpm 10 --tpm 50000
```

Full run:

```bash
python stage2_llm_runner.py --rpm 40 --tpm 200000
```

Cost/rate guidance:

- Start with low `--max-rows` for prompt validation.
- Keep `temperature=0` for stable JSON output.
- Use `--resume` (default true) to avoid re-paying for already processed rows.
- Tune `--rpm` and `--tpm` based on provider limits.
