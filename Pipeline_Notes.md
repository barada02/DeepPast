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

---

## 10) Quick Decision Rule

- If you want **safe first training**: use rows where `first_word_match == 1` and `boundary_method == marker_search`.
- If you want **more data quickly**: use full `sentence_level_pairs.csv`.
