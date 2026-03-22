"""Deep Past data pipelines (minimal version).

This script intentionally implements only the two pipelines requested:
1) Canonical text normalization on gold train pairs.
2) Sentence-level preparation using train.csv + Sentences_Oare_FirstWord_LinNum.csv.

It does NOT process test.csv or published_texts.csv in this version.
"""

from __future__ import annotations

import argparse
import hashlib
import re
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


def build_sentence_targets(
    sentence_df: pd.DataFrame,
    valid_ratio: float,
    split_salt: str,
) -> pd.DataFrame:
    """Pipeline 2A: sentence target table from Sentences_Oare_FirstWord_LinNum.csv.

    Note: this file does not provide full source transliteration sentences.
    It provides sentence-level English targets plus first-word/line anchors.
    """
    work = sentence_df.copy()
    work["translation_clean"] = work["translation"].map(normalize_translation)
    work["first_word_transcription_clean"] = work["first_word_transcription"].map(normalize_transliteration)

    sentence_id_col = "text_uuid" if "text_uuid" in work.columns else work.columns[0]
    work = deterministic_split_by_id(work, sentence_id_col, valid_ratio, split_salt)

    keep_cols = [
        c
        for c in [
            "text_uuid",
            "sentence_uuid",
            "sentence_obj_in_text",
            "line_number",
            "first_word_transcription",
            "first_word_transcription_clean",
            "translation",
            "translation_clean",
            "split",
        ]
        if c in work.columns
    ]
    out = work[keep_cols].copy()
    out["provenance"] = "gold_sentence_targets"
    out["trust_weight"] = 1.0
    return out


def build_train_sentence_bridge(canonical_train_pairs: pd.DataFrame, sentence_targets: pd.DataFrame) -> pd.DataFrame:
    """Pipeline 2B: bridge sentence targets to train document text (when ids match).

    The merge uses text_uuid (sentence file) against oare_id (train file).
    This creates a practical intermediate table for later alignment work.
    """
    train_bridge_cols = [
        c
        for c in ["oare_id", "transliteration", "transliteration_clean", "translation", "translation_clean", "split"]
        if c in canonical_train_pairs.columns
    ]
    train_docs = canonical_train_pairs[train_bridge_cols].copy().rename(
        columns={"oare_id": "text_uuid", "split": "doc_split"}
    )

    bridge = sentence_targets.merge(train_docs, on="text_uuid", how="left")
    bridge["has_matching_train_doc"] = bridge["transliteration"].notna().astype(int)
    return bridge


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
    sentence_targets = build_sentence_targets(sentence_df, args.valid_ratio, args.split_salt)
    sentence_bridge = build_train_sentence_bridge(canonical_train_pairs, sentence_targets)

    canonical_train_pairs.to_csv(output_dir / "canonical_train_pairs.csv", index=False)
    sentence_targets.to_csv(output_dir / "sentence_translation_targets.csv", index=False)
    sentence_bridge.to_csv(output_dir / "train_sentence_bridge.csv", index=False)

    print("Pipelines completed (minimal scope).")
    print(f"Input dir: {input_dir}")
    print(f"Output dir: {output_dir}")
    print("Generated files:")
    print(" - canonical_train_pairs.csv")
    print(" - sentence_translation_targets.csv")
    print(" - train_sentence_bridge.csv")


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
