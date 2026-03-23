# Notebook Analysis: deep-past-challenge-byt5-3-model-ensemble-mbr.ipynb

## 1) What this notebook does (high level)

This notebook is an **inference-only ensemble pipeline** for the Deep Past test set.

It:
1. Loads `test.csv`.
2. Applies custom transliteration preprocessing.
3. Runs **3 ByT5 seq2seq models**.
4. Generates multiple candidates per model (beam + sampling).
5. Uses **MBR (Minimum Bayes Risk)-style selection** with a geometric mean similarity score.
6. Postprocesses chosen outputs and writes `submission.csv`.

---

## 2) Data pipeline details

### Input
- `test.csv` with columns: `id`, `transliteration`.

### Preprocessing (`preprocess`)
- Converts ASCII transliteration variants to diacritics (`sz -> š`, `s, -> ṣ`, `t, -> ṭ`, vowel 2/3 to acute/grave).
- Collapses many gap notations into `<gap>`.
- Normalizes characters (`ḫ -> h`, subscript digits to normal digits).
- Canonicalizes decimal forms to a fixed compact representation.
- Adds heuristic commodity hints if specific OA logograms occur.
- Prefixes each source with task instruction: `translate Akkadian to English: ...`.

### Batching / throughput
- Uses a custom `BucketSampler` to group by token length for reduced padding waste.
- `DataLoader` uses model-specific tokenizer in a custom `collate` function.

### Candidate generation
For each model and batch:
- Beam search candidates (`num_beams=10`, `num_return_sequences=6`).
- Sampling candidates (`top_p=0.90`, `temperature=0.70`, `num_return_sequences=3`).
- Pools all candidates per `id` across all three models.

### Postprocessing (`postprocess`)
- Removes hint artifacts.
- Normalizes `<gap>` behavior.
- Removes forbidden symbols and repeated words/punctuation.
- Converts selected decimal strings to unicode fractions.

### Output
- Final `submission.csv` with `id,translation`.

---

## 3) Model architecture and decoding strategy

### Core architecture
- **Transformer seq2seq** (`AutoModelForSeq2SeqLM`) using **ByT5-family checkpoints**.
- Character/byte-friendly setup (consistent with noisy transliteration input).

### Ensemble design
- 3 independently trained checkpoints.
- Memory-safe loading/unloading per model.
- Mixed precision support with BF16 autocast when available.

### Selection objective
- Not simple averaging; uses candidate-level selection via `mbr_pick`.
- Candidate utility score is mean pairwise similarity to others.
- Similarity metric is geometric mean of:
  - sentence BLEU
  - character F-score (chr-like)

This favors candidates that are most “central” among generated hypotheses.

---

## 4) What is strong here

1. **Robust normalization** for real transliteration noise.
2. **Length bucketing** for efficient decoding.
3. **Hybrid decoding** (beam + nucleus sampling) improves diversity.
4. **MBR-style reranking** reduces brittle single-decoder choices.
5. Explicit OOM handling and model unload for Kaggle GPU constraints.

---

## 5) Risks / caveats

1. `OA_HINTS` text augmentation may inject bias or hallucination anchors.
2. `mbr_pick` uses pairwise candidate agreement, not ground-truth utility.
3. Score functions are sentence-level approximations (not corpus sacreBLEU/chrF++).
4. Aggressive postprocessing may remove meaningful symbols in edge cases.

---

## 6) Transfer ideas for this repo (data pipeline + modeling)

### Data pipeline ideas to apply
1. Add a **shared transliteration canonicalizer** module so `data_layer.py` and inference scripts use the same rules.
2. Expand gap/scribal normalization patterns in `data_layer.py` to include broader variants seen here.
3. Add optional **bucketed inference dataloader** utility for stage-2 model inference.

### Model/inference ideas to apply
1. Add optional **candidate-pool decoding** mode (beam + top-p sample).
2. Add lightweight **MBR reranking** over candidate pools before final output.
3. Keep deterministic mode (`temperature=0`) as baseline and make MBR mode toggleable.

### Practical integration order (recommended)
1. First: port normalization improvements only (low risk, immediate value).
2. Next: add candidate pooling in inference script.
3. Then: enable MBR reranking behind a flag and compare offline metrics.

---

## 7) Minimal actionable checklist for next commit

- [ ] Extract transliteration normalization into one reusable utility file.
- [ ] Add unit-like sanity checks for key normalization patterns (`[x]`, `...`, subscripts, `ḫ/Ḫ`).
- [ ] Add optional `--decode-mode {greedy,beam,mbr}` to inference runner.
- [ ] Add candidate-cache logging for debugging MBR decisions.

---

## 8) Bottom line

This notebook is primarily an **ensemble inference + reranking recipe**.  
The most reusable pieces for your current work are:
- stronger transliteration normalization,
- efficient bucketed batching,
- and optional MBR-style hypothesis selection.
