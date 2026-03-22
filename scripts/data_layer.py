"""Deep Past data pipelines (minimal version).

This script intentionally implements only the two pipelines requested:
1) Canonical text normalization on gold train pairs.
2) Sentence-level source-target pair generation from:
    - train.csv (document-level transliteration + translation)
    - Sentences_Oare_FirstWord_LinNum.csv (sentence markers + sentence translations)

It does NOT process test.csv or published_texts.csv in this version.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# -----------------------------------------------------------------------------
# Editable defaults
# -----------------------------------------------------------------------------
# Edit these two values directly if you want fixed paths without CLI flags.
DEFAULT_INPUT_DIR = "C:\\Users\\barad\\Desktop\\Kaggle\\DeepPast\\data\\deep-past-data"
DEFAULT_OUTPUT_DIR = "data/processed"

# -----------------------------------------------------------------------------
# Character normalization helpers
# -----------------------------------------------------------------------------
SUBSCRIPT_MAP = str.maketrans(
    {
        "₀": "0",
        "₁": "1",
        "₂": "2",
        "₃": "3",
        "₄": "4",
        "₅": "5",
        "₆": "6",
        "₇": "7",
        "₈": "8",
        "₉": "9",
        "ₓ": "x",
    }
)


def normalize_transliteration(text: str) -> str:
    """Apply canonical transliteration cleanup.

    Rules are intentionally simple and deterministic:
    - Ḫ/ḫ -> H/h
    - [x] -> <gap>
    - [...] or ... -> <big_gap>
    - subscript digits/letters -> plain
    - remove selected editorial symbols
    - normalize whitespace
    """
    if pd.isna(text):
        return ""

    value = str(text)
    value = value.replace("Ḫ", "H").replace("ḫ", "h")

    value = re.sub(r"\[x\]", " __GAP__ ", value)
    value = re.sub(r"\[\s*\.\.\.\s*\]", " __BIG_GAP__ ", value)
    value = re.sub(r"\.\.\.", " __BIG_GAP__ ", value)

    value = value.translate(SUBSCRIPT_MAP)

    value = re.sub(r"[!?/]", " ", value)
    value = value.replace(":", " ").replace(".", " ")
    value = value.replace("[", "").replace("]", "")
    value = value.replace("<", "").replace(">", "")

    value = value.replace("__GAP__", "<gap>")
    value = value.replace("__BIG_GAP__", "<big_gap>")

    return re.sub(r"\s+", " ", value).strip()


def normalize_translation(text: str) -> str:
    """Apply minimal translation cleanup.

    Keeps target text mostly intact and only removes angle brackets + extra spaces.
    """
    if pd.isna(text):
        return ""

    value = str(text)
    value = value.replace("<", "").replace(">", "")
    return re.sub(r"\s+", " ", value).strip()


def normalize_token_for_match(token: str) -> str:
    """Normalize a single token for boundary matching checks."""
    value = normalize_transliteration(token)
    value = value.lower()
    value = re.sub(r"[^a-z0-9šṣṭḫ\-]", "", value)
    return value


def deterministic_split_by_id(
    df: pd.DataFrame,
    id_col: str,
    valid_ratio: float,
    salt: str,
) -> pd.DataFrame:
    """Create deterministic train/valid split by stable hash of id_col."""
    cutoff = int(valid_ratio * 10_000)

    def is_valid(identifier: str) -> int:
        payload = f"{salt}:{identifier}".encode("utf-8")
        bucket = int(hashlib.md5(payload).hexdigest(), 16) % 10_000
        return int(bucket < cutoff)

    out = df.copy()
    out["is_valid"] = out[id_col].astype(str).map(is_valid)
    out["split"] = out["is_valid"].map({0: "train", 1: "valid"})
    return out


def build_canonical_train_pairs(train_df: pd.DataFrame, valid_ratio: float, split_salt: str) -> pd.DataFrame:
    """Pipeline 1: canonical train pairs from train.csv."""
    work = train_df.copy()
    work["transliteration_clean"] = work["transliteration"].map(normalize_transliteration)
    work["translation_clean"] = work["translation"].map(normalize_translation)

    split_col = "oare_id" if "oare_id" in work.columns else work.columns[0]
    work = deterministic_split_by_id(work, split_col, valid_ratio, split_salt)

    keep_cols = [
        c
        for c in [
            "oare_id",
            "transliteration",
            "transliteration_clean",
            "translation",
            "translation_clean",
            "split",
        ]
        if c in work.columns
    ]
    out = work[keep_cols].copy()
    out["provenance"] = "gold_train_doc"
    out["trust_weight"] = 1.0
    return out


def build_sentence_level_pairs(canonical_train_pairs: pd.DataFrame, sentence_df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline 2: build sentence-level source-target pairs.

    Strategy:
    - join sentence-map rows to overlapping train documents (text_uuid == oare_id)
    - use first_word_obj_in_text as sentence-start word index (1-based)
    - slice transliteration_clean token stream between consecutive starts
    - pair sliced source sentence with sentence translation from map file
    """
    doc_cols = [c for c in ["oare_id", "transliteration", "transliteration_clean", "split"] if c in canonical_train_pairs.columns]
    docs = canonical_train_pairs[doc_cols].copy().rename(columns={"oare_id": "text_uuid", "split": "doc_split"})

    work = sentence_df.copy()
    work["translation_clean"] = work["translation"].map(normalize_translation)
    work["first_word_transcription_clean"] = work["first_word_transcription"].map(normalize_transliteration)
    work = work.merge(docs, on="text_uuid", how="inner")

    rows: list[dict] = []
    for text_uuid, grp in work.groupby("text_uuid", sort=False):
        grp_sorted = grp.sort_values(["first_word_obj_in_text", "sentence_obj_in_text"], kind="stable").reset_index(drop=True)

        doc_translit_raw = str(grp_sorted.loc[0, "transliteration"])
        doc_tokens_raw = doc_translit_raw.split()
        doc_tokens_for_bounds = doc_tokens_raw
        if not doc_tokens_for_bounds:
            continue

        doc_tokens_norm = [normalize_token_for_match(token) for token in doc_tokens_raw]
        search_cursor = 0
        starts: list[int] = []
        start_method: list[str] = []

        for row in grp_sorted.itertuples(index=False):
            marker = normalize_token_for_match(getattr(row, "first_word_transcription_clean", ""))
            found_idx = None

            if marker:
                for j in range(search_cursor, len(doc_tokens_norm)):
                    token_norm = doc_tokens_norm[j]
                    if token_norm == marker or token_norm.startswith(marker) or marker.startswith(token_norm):
                        found_idx = j
                        break

            if found_idx is None:
                fallback = pd.to_numeric(getattr(row, "first_word_obj_in_text", None), errors="coerce")
                if pd.notna(fallback):
                    fallback_idx = int(fallback) - 1
                    if 0 <= fallback_idx < len(doc_tokens_norm):
                        found_idx = fallback_idx
                        start_method.append("first_word_obj_fallback")

            if found_idx is None:
                continue

            starts.append(found_idx + 1)
            if len(start_method) < len(starts):
                start_method.append("marker_search")
            search_cursor = max(search_cursor, found_idx)

        if not starts:
            continue

        if len(starts) != len(grp_sorted):
            grp_sorted = grp_sorted.iloc[: len(starts)].copy().reset_index(drop=True)

        for i, row in enumerate(grp_sorted.itertuples(index=False)):
            start_idx = starts[i] - 1
            end_idx = starts[i + 1] - 1 if i + 1 < len(starts) else len(doc_tokens_for_bounds)
            if start_idx >= end_idx:
                continue

            source_tokens_raw = doc_tokens_raw[start_idx:end_idx]
            source_sentence_raw = " ".join(source_tokens_raw).strip()
            source_sentence_clean = normalize_transliteration(source_sentence_raw)
            if not source_sentence_clean:
                continue

            marker = normalize_token_for_match(getattr(row, "first_word_transcription_clean", ""))
            source_first = normalize_token_for_match(source_tokens_raw[0]) if source_tokens_raw else ""
            marker_match = int(bool(marker) and (source_first == marker or source_first.startswith(marker) or marker.startswith(source_first)))

            rows.append(
                {
                    "text_uuid": text_uuid,
                    "sentence_uuid": getattr(row, "sentence_uuid", ""),
                    "sentence_obj_in_text": getattr(row, "sentence_obj_in_text", ""),
                    "line_number": getattr(row, "line_number", ""),
                    "source_sentence_raw": source_sentence_raw,
                    "source_sentence_clean": source_sentence_clean,
                    "target_sentence": getattr(row, "translation", ""),
                    "target_sentence_clean": getattr(row, "translation_clean", ""),
                    "source_start_word_idx": start_idx + 1,
                    "source_end_word_idx_exclusive": end_idx + 1,
                    "boundary_method": start_method[i],
                    "first_word_transcription": getattr(row, "first_word_transcription", ""),
                    "first_word_transcription_clean": getattr(row, "first_word_transcription_clean", ""),
                    "first_word_match": marker_match,
                    "split": getattr(row, "doc_split", "train"),
                    "provenance": "gold_sentence_pair_from_markers",
                    "trust_weight": 1.0,
                }
            )

    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    """Execute the minimal two-pipeline workflow and save outputs."""
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = input_dir / "train.csv"
    sentence_path = input_dir / "Sentences_Oare_FirstWord_LinNum.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing required file: {train_path}")
    if not sentence_path.exists():
        raise FileNotFoundError(f"Missing required file: {sentence_path}")

    train_df = pd.read_csv(train_path)
    sentence_df = pd.read_csv(sentence_path)

    canonical_train_pairs = build_canonical_train_pairs(train_df, args.valid_ratio, args.split_salt)
    sentence_level_pairs = build_sentence_level_pairs(canonical_train_pairs, sentence_df)

    canonical_train_pairs.to_csv(output_dir / "canonical_train_pairs.csv", index=False)
    sentence_level_pairs.to_csv(output_dir / "sentence_level_pairs.csv", index=False)

    summary = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "valid_ratio": args.valid_ratio,
        "split_salt": args.split_salt,
        "canonical_train_pairs_rows": int(len(canonical_train_pairs)),
        "sentence_level_pairs_rows": int(len(sentence_level_pairs)),
        "sentence_level_pairs_train_rows": int((sentence_level_pairs["split"] == "train").sum()) if "split" in sentence_level_pairs.columns else 0,
        "sentence_level_pairs_valid_rows": int((sentence_level_pairs["split"] == "valid").sum()) if "split" in sentence_level_pairs.columns else 0,
        "first_word_match_rate_percent": float(
            round((sentence_level_pairs["first_word_match"].astype(int).mean() * 100.0), 2)
        )
        if "first_word_match" in sentence_level_pairs.columns and len(sentence_level_pairs) > 0
        else 0.0,
        "marker_search_rows": int((sentence_level_pairs["boundary_method"] == "marker_search").sum()) if "boundary_method" in sentence_level_pairs.columns else 0,
        "first_word_obj_fallback_rows": int((sentence_level_pairs["boundary_method"] == "first_word_obj_fallback").sum()) if "boundary_method" in sentence_level_pairs.columns else 0,
    }

    summary_path = output_dir / "pipeline_run_history.csv"
    summary_df = pd.DataFrame([summary])
    if summary_path.exists():
        previous = pd.read_csv(summary_path)
        summary_df = pd.concat([previous, summary_df], ignore_index=True)
    summary_df.to_csv(summary_path, index=False)

    print("Pipelines completed (minimal scope).")
    print(f"Input dir: {input_dir}")
    print(f"Output dir: {output_dir}")
    print("Generated files:")
    print(" - canonical_train_pairs.csv")
    print(" - sentence_level_pairs.csv")
    print(" - pipeline_run_history.csv")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser.

    You can either:
    - edit DEFAULT_INPUT_DIR / DEFAULT_OUTPUT_DIR above, or
    - pass --input-dir / --output-dir from command line.
    """
    parser = argparse.ArgumentParser(description="Deep Past minimal data pipelines")
    parser.add_argument("--input-dir", type=str, default=DEFAULT_INPUT_DIR, help="Directory containing source CSV files")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory for generated CSV files")
    parser.add_argument("--valid-ratio", type=float, default=0.1, help="Validation ratio for deterministic split")
    parser.add_argument("--split-salt", type=str, default="deep_past_split_v1", help="Salt for deterministic split")
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    run(args)
