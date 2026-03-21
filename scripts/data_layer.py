from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import pandas as pd

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
    cutoff = int(valid_ratio * 10_000)

    def is_valid(identifier: str) -> int:
        payload = f"{salt}:{identifier}".encode("utf-8")
        bucket = int(hashlib.md5(payload).hexdigest(), 16) % 10_000
        return int(bucket < cutoff)

    out = df.copy()
    out["is_valid"] = out[id_col].astype(str).map(is_valid)
    out["split"] = out["is_valid"].map({0: "train", 1: "valid"})
    return out


def run(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = input_dir / "train.csv"
    test_path = input_dir / "test.csv"
    published_path = input_dir / "published_texts.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing required file: {train_path}")

    train_df = pd.read_csv(train_path)
    train_df["transliteration_clean"] = train_df["transliteration"].map(normalize_transliteration)
    train_df["translation_clean"] = train_df["translation"].map(normalize_translation)

    split_col = "oare_id" if "oare_id" in train_df.columns else train_df.columns[0]
    train_split_df = deterministic_split_by_id(train_df, split_col, args.valid_ratio, args.split_salt)

    keep_train_cols = [
        c
        for c in [
            "oare_id",
            "transliteration",
            "translation",
            "transliteration_clean",
            "translation_clean",
            "split",
        ]
        if c in train_split_df.columns
    ]

    train_ready = train_split_df[keep_train_cols].copy()
    train_ready.to_csv(output_dir / "train_ready.csv", index=False)
    train_ready.query("split == 'train'").to_csv(output_dir / "train_ready_train.csv", index=False)
    train_ready.query("split == 'valid'").to_csv(output_dir / "train_ready_valid.csv", index=False)

    if test_path.exists():
        test_df = pd.read_csv(test_path)
        test_df["transliteration_clean"] = test_df["transliteration"].map(normalize_transliteration)
        keep_test_cols = [
            c
            for c in ["id", "text_id", "line_start", "line_end", "transliteration", "transliteration_clean"]
            if c in test_df.columns
        ]
        test_df[keep_test_cols].to_csv(output_dir / "test_ready.csv", index=False)

    if published_path.exists():
        published_df = pd.read_csv(published_path)
        if "transliteration" in published_df.columns:
            published_df["transliteration_clean"] = published_df["transliteration"].map(normalize_transliteration)
            keep_pub_cols = [c for c in ["oare_id", "transliteration", "transliteration_clean"] if c in published_df.columns]
            published_df[keep_pub_cols].to_csv(output_dir / "published_ready.csv", index=False)

    print("Data pipeline completed.")
    print(f"Output directory: {output_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deep Past data-first preprocessing pipeline")
    parser.add_argument("--input-dir", type=str, default="data/raw", help="Directory containing Kaggle CSV files")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Directory for processed outputs")
    parser.add_argument("--valid-ratio", type=float, default=0.1, help="Validation ratio for deterministic split")
    parser.add_argument("--split-salt", type=str, default="deep_past_split_v1", help="Salt for deterministic split")
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    run(args)
