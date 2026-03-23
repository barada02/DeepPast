# Notebook Analysis: deep-past-challenge-translate-akkadian-to-engl.ipynb

## 1) What this notebook does (high level)

This notebook is a **train + infer baseline/fine-tuning pipeline** using Hugging Face seq2seq tools.

It:
1. Loads challenge CSVs.
2. Builds a large merged training set from multiple sources.
3. Fine-tunes `Helsinki-NLP/opus-mt-mul-en`.
4. Evaluates with BLEU + chrF.
5. Generates `submission.csv` with a hybrid lookup + model fallback logic.

---

## 2) Data pipeline construction

### Base supervised source
- `train.csv` renamed to `akkadian` (source) and `english` (target).

### Additional sources merged
- **Source A**: `published_texts.csv` (`transliteration` + `AICC_translation`).
- **Source B**: `Sentences_Oare_FirstWord_LinNum.csv` translations joined with `published_texts` transliterations by `text_uuid/oare_id`.
- **Source C**: `test.csv` + `sample_submission.csv` joined on `id`.
- **Source D**: first-word pairs (`first_word_transcription` -> `translation`) from sentence map.

### Cleaning / filtering
- Trims whitespace and collapses repeated spaces.
- Drops missing/empty rows.
- Keeps rows with token lengths between 2 and 300 words.
- Drops duplicates by source text (`akkadian`).

### Train/val split
- Uses `Dataset.from_pandas` then random `train_test_split(test_size=0.05, seed=42)`.

---

## 3) Model architecture and training setup

### Architecture
- `AutoModelForSeq2SeqLM` with checkpoint `Helsinki-NLP/opus-mt-mul-en`.
- Tokenization uses source-target paired tokenizer API (`text_target=...`).

### Trainer stack
- `Seq2SeqTrainer` + `DataCollatorForSeq2Seq`.
- Early stopping with patience 3.
- Cosine LR schedule, warmup ratio 0.1.
- Generation-aware evaluation (`predict_with_generate=True`, beam size 5).

### Metrics
- `evaluate` package for `chrf` and `bleu`.
- Best checkpoint selected by `chrf` when available.

---

## 4) Inference strategy in this notebook

Submission generation uses a **3-tier fallback**:
1. Try lookup in `published_texts.AICC_translation` by `text_id` prefix.
2. If missing, try lookup from `sample_submission.csv` by `id`.
3. If still missing, run model generation.

Model generation uses beam search (`num_beams=5`) with anti-repeat constraint (`no_repeat_ngram_size=3`).

---

## 5) What is useful to learn/apply

1. Clean, simple Hugging Face seq2seq training skeleton that is easy to adapt.
2. Multi-source data merging concept (good idea if sources are trustworthy).
3. Early stopping + best-model-by-metric flow is practical and stable.
4. Metric wiring (`chrf` + `bleu`) can be reused for offline validation scripts.

---

## 6) Critical risks and issues (important)

1. **Data leakage risk**: using `sample_submission.csv` as training/inference source is not valid for real leaderboard scenarios.
2. **Potential leakage/shortcut**: direct `AICC_translation` lookup for test rows bypasses translation modeling.
3. **Source quality mismatch**: `AICC_translation` is known noisy; mixing without confidence filtering can hurt model quality.
4. **Random split**: row-wise random split can cause cross-document leakage; deterministic id-based split is safer.

---

## 7) How to adapt safely in this repo

### Keep
- HF trainer structure.
- Early stopping and metric-driven checkpointing.
- Modular `translate_batch` helper pattern.

### Modify
1. Remove any test-time gold/reference lookups.
2. Replace random split with deterministic split by document id (`oare_id/text_uuid`).
3. Add confidence-based filtering for weak silver/noisy sources.
4. Track provenance and trust weights per pair (already aligned with your current direction).

### Suggested immediate integration
1. Add a standalone `training_stage.py` that reads:
   - `canonical_train_pairs.csv`
   - high-confidence `sentence_level_pairs.csv`
   - optional high-confidence stage-2 silver pairs
2. Train with clean separation between train/valid documents.

---

## 8) Bottom line

This notebook provides a good **trainer template**, but its submission logic includes shortcuts that should not be used in a strict competition setup.  
For your pipeline, the best transferable parts are the HF training scaffold, metric integration, and clean generation function design.
