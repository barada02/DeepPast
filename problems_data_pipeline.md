Good — you’ve identified the **right bottleneck**. Let’s dissect this properly.

You’re getting:

* **~1561 document rows** ✅ (expected)
* **~396 sentence rows** ❌ (too low)

This is not random — it’s **systematically caused by how sentence boundaries are being recovered** in `build_sentence_level_pairs()`.

I’ll walk you through **exactly where things break** and why.

---

# 🧠 1. Pipeline Logic (What your code is trying to do)

From your script :

### Flow:

```text
train.csv (doc-level)
        +
sentence markers file
        ↓
Find sentence start positions
        ↓
Slice tokens
        ↓
Build sentence pairs
```

---

# 🚨 2. Where the Data is Getting Lost

There are **3 major failure points**:

---

# ⚠️ PROBLEM 1: Marker Matching is Too Fragile

### Code:

```python
marker = normalize_token_for_match(first_word_transcription_clean)

for j in range(search_cursor, len(doc_tokens_norm)):
    if token_norm == marker or token_norm.startswith(marker) or marker.startswith(token_norm):
        found_idx = j
```

---

## ❌ Why this fails:

### Akkadian issues:

* hyphenated tokens: `a-na`, `i-lu`
* determinatives: `{d}UTU`
* normalization removes symbols

👉 Result:

```text
marker = "ana"
token = "a-na"
→ mismatch ❌
```

---

## 🔥 Impact:

* `found_idx = None`
* fallback used OR row skipped

---

# ⚠️ PROBLEM 2: Fallback is Weak (and often wrong)

### Code:

```python
fallback = first_word_obj_in_text
fallback_idx = int(fallback) - 1
```

---

## ❌ Why this fails:

* This assumes:

  ```text
  word index == token index
  ```
* But:

  * transliteration formatting changes token count
  * gaps, symbols removed earlier

---

## 🔥 Impact:

* incorrect boundaries
* overlapping or invalid spans
* later dropped

---

# ⚠️ PROBLEM 3: Hard Drop When Not Found

### Code:

```python
if found_idx is None:
    continue
```

---

## ❌ This is the BIGGEST LOSS

👉 If marker + fallback both fail:

* ENTIRE sentence is dropped

---

### 💥 This explains your numbers:

| Stage          | Effect    |
| -------------- | --------- |
| marker fails   | many      |
| fallback fails | many      |
| continue       | data gone |

👉 Result: **1561 → 396**

---

# ⚠️ PROBLEM 4: Forced Truncation (Silent Data Loss)

### Code:

```python
if len(starts) != len(grp_sorted):
    grp_sorted = grp_sorted.iloc[: len(starts)]
```

---

## ❌ What this does:

* If some markers fail:

  * you **cut off remaining sentences**

---

### 💥 Example:

```text
Expected: 10 sentences
Found: 4 starts
→ you keep only first 4
→ lose 6 sentences ❌
```

---

# ⚠️ PROBLEM 5: Overlap / Cursor Issue

### Code:

```python
search_cursor = max(search_cursor, found_idx)
```

---

## ❌ Why risky:

* prevents backtracking
* if one match is slightly off → next matches fail

---

# ⚠️ PROBLEM 6: Boundary Collapse

### Code:

```python
if start_idx >= end_idx:
    continue
```

---

## ❌ Happens when:

* duplicate start positions
* incorrect fallback

👉 sentence dropped again

---

# 🧠 Root Cause Summary

Your pipeline assumes:

```text
marker → exact token match → clean boundaries
```

But reality is:

```text
marker ≠ token (due to normalization, hyphens, symbols)
```

---

# 🔍 What’s REALLY happening

---

## For each document:

Let’s say:

* 10 expected sentences

### Actual pipeline:

| Step            | Result     |
| --------------- | ---------- |
| marker matching | finds 4    |
| fallback works  | +2         |
| rest fail       | 4 lost     |
| truncation      | drops tail |
| overlap errors  | drop more  |

👉 Final:

```text
10 → 3 or 4 sentences
```

---

# 📊 Why ~396 specifically?

Rough intuition:

```text
1561 docs
avg ~3-4 sentences/doc
→ expected ~5000+

you get 396
→ ~75–90% loss
```

👉 consistent with:

* aggressive `continue`
* truncation
* weak matching

---

# 🧠 The REAL Problem (Conceptual)

You are trying to do:

```text
string matching → alignment
```

But this task needs:

```text
fuzzy + structural + positional alignment
```

---

# 🚀 What Needs to Change (Conceptually, no code yet)

---

## ✅ 1. Stop Dropping Data Aggressively

Replace mindset:

```text
"if not perfect → discard"
```

with:

```text
"if uncertain → approximate"
```

---

## ✅ 2. Decouple Marker Matching from Boundary Creation

Right now:

```text
marker → boundary
```

Better:

```text
use positions + smoothing + fallback segmentation
```

---

## ✅ 3. Do NOT truncate grp_sorted

This line is dangerous:

```python
grp_sorted = grp_sorted.iloc[: len(starts)]
```

👉 This is silently killing data

---

## ✅ 4. Use Relative Position Instead of Absolute Matching

Instead of:

```text
find exact token
```

Use:

```text
sentence positions → proportional slicing
```

---

## ✅ 5. Treat This as Alignment Problem (not matching)

Think like:

* sequence alignment
* not string equality

---

# 🔥 Key Insight (Important for your architecture)

This bug is not just a bug.

👉 It tells you:

> "Rule-based alignment is not enough"

---

# 🧠 This directly connects to your LLM idea

You can:

### Use LLM to:

* align sentences
* correct boundaries
* recover lost data

---

# 🎯 Final Diagnosis

### Why you get only 396:

✔ Marker matching fails (major)
✔ Fallback unreliable
✔ Aggressive dropping (`continue`)
✔ Truncation kills tail sentences
✔ Boundary errors collapse spans

---

