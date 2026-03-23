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
2. Try to locate sentence starts via **marker search**:
   - compare normalized `first_word_transcription` with normalized document tokens
   - search sequentially through document token stream
3. If marker search fails, fallback to:
   - `first_word_obj_in_text` (word-index based boundary)
4. Slice document transliteration between consecutive boundaries.
5. Normalize sliced source sentence.
6. Pair with sentence translation from sentence map.

Quality helper columns are written for analysis:

- `first_word_match`
- `boundary_method` (`marker_search` or `first_word_obj_fallback`)

---

## 6) Outputs

Generated in output directory:

1. `canonical_train_pairs.csv`
   - document-level normalized pairs
2. `sentence_level_pairs.csv`
   - sentence-level pairs for seq2seq
3. `pipeline_run_history.csv`
   - appended run log (one row per script execution) with timestamp, settings, row counts, and match metrics

Recommended immediate seq2seq training file:

- **`sentence_level_pairs.csv`**
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
- Current generated sentence-level pairs are **396** rows.

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
