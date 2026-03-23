"""Publications-to-tablet candidate extractor (stage-1).

Purpose
- Link OCR pages in publications.csv to Old Assyrian tablets in published_texts.csv.
- Create high-confidence candidate rows for a later LLM extraction/alignment stage.

Current matching strategy (deterministic)
1) CDLI id matching (P######) from page_text and pdf_name.
2) Join matched CDLI ids to published_texts rows via cdli_id.

Why this first?
- It is fast, transparent, and low-noise.
- It creates a controlled candidate pool for LLM-based translation extraction.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

# -----------------------------------------------------------------------------
# Editable defaults
# -----------------------------------------------------------------------------
DEFAULT_INPUT_DIR = "C:\\Users\\barad\\Desktop\\Kaggle\\DeepPast\\data\\deep-past-data"
DEFAULT_OUTPUT_DIR = "data/processed"

DEFAULT_PUBLISHED_TEXTS = "published_texts.csv"
DEFAULT_PUBLICATIONS = "publications.csv"

CDLI_PATTERN = re.compile(r"\bP\d{5,7}\b", re.IGNORECASE)
WHITESPACE_PATTERN = re.compile(r"\s+")
NON_ALNUM_PATTERN = re.compile(r"[^A-Z0-9]+")
REF_PATTERN = re.compile(
    r"\b(?:BM\s*\d{3,8}|[A-Z]{2,8}\s*[IVXLC0-9]{0,4}\s*\d{1,6}[A-Z]{0,2})\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PublishedRow:
    """Small typed container for lookup rows from published_texts."""

    oare_id: str
    cdli_id: str
    label: str
    aliases: str
    publication_catalog: str
    transliteration: str


def split_multi_value(value: str) -> list[str]:
    """Split pipe-separated fields safely."""
    if value is None or pd.isna(value):
        return []
    return [part.strip() for part in str(value).split("|") if part.strip()]


def normalize_space(text: str) -> str:
    """Collapse whitespace for compact output fields."""
    if text is None or pd.isna(text):
        return ""
    return WHITESPACE_PATTERN.sub(" ", str(text)).strip()


def normalize_ref(text: str) -> str:
    """Canonical form for reference labels (aliases/catalog labels)."""
    if text is None or pd.isna(text):
        return ""
    raw = normalize_space(str(text)).upper()
    raw = NON_ALNUM_PATTERN.sub("", raw)
    return raw


def extract_cdli_ids(text: str) -> list[str]:
    """Extract unique CDLI ids from text."""
    if text is None or pd.isna(text):
        return []
    found = CDLI_PATTERN.findall(str(text))
    unique = sorted({item.upper() for item in found})
    return unique


def extract_publication_refs(text: str) -> list[str]:
    """Extract likely publication/museum reference strings from text."""
    if text is None or pd.isna(text):
        return []
    refs = REF_PATTERN.findall(str(text))
    norm = sorted({normalize_ref(item) for item in refs if normalize_ref(item)})
    return norm


def snippet_around_match(text: str, needle: str, radius: int = 240) -> str:
    """Return a compact snippet around first match occurrence."""
    content = normalize_space(text)
    if not content:
        return ""
    idx = content.upper().find(needle.upper())
    if idx < 0:
        return content[: min(len(content), radius * 2)]
    left = max(0, idx - radius)
    right = min(len(content), idx + len(needle) + radius)
    return content[left:right]


def build_cdli_lookup(
    published_texts_path: Path,
) -> tuple[dict[str, list[PublishedRow]], dict[str, list[PublishedRow]], pd.DataFrame]:
    """Build CDLI -> published_texts lookup.

    Returns:
    - cdli_lookup: CDLI id to candidate rows
    - published_df: compact dataframe used for downstream outputs
    """
    usecols = [
        "oare_id",
        "cdli_id",
        "label",
        "aliases",
        "publication_catalog",
        "transliteration",
    ]
    published_df = pd.read_csv(published_texts_path, usecols=usecols)

    lookup: dict[str, list[PublishedRow]] = {}
    ref_lookup: dict[str, list[PublishedRow]] = {}
    for row in published_df.itertuples(index=False):
        row_obj = PublishedRow(
            oare_id=str(getattr(row, "oare_id", "")),
            cdli_id=str(getattr(row, "cdli_id", "")),
            label=str(getattr(row, "label", "")),
            aliases=str(getattr(row, "aliases", "")),
            publication_catalog=str(getattr(row, "publication_catalog", "")),
            transliteration=str(getattr(row, "transliteration", "")),
        )
        cdli_values = split_multi_value(row_obj.cdli_id)
        for cdli in cdli_values:
            cdli_key = cdli.upper()
            lookup.setdefault(cdli_key, []).append(row_obj)

        ref_values = []
        ref_values.extend(split_multi_value(row_obj.label))
        ref_values.extend(split_multi_value(row_obj.aliases))
        ref_values.extend(split_multi_value(row_obj.publication_catalog))

        for ref in ref_values:
            ref_key = normalize_ref(ref)
            if not ref_key:
                continue
            if len(ref_key) < 4:
                continue
            ref_lookup.setdefault(ref_key, []).append(row_obj)

    return lookup, ref_lookup, published_df


def iter_publications_chunks(
    publications_path: Path,
    chunk_size: int,
    max_rows: int,
) -> Iterable[pd.DataFrame]:
    """Yield publications rows in chunks (memory safe for large file)."""
    usecols = ["pdf_name", "page", "page_text", "has_akkadian"]
    reader = pd.read_csv(publications_path, usecols=usecols, chunksize=chunk_size)

    seen = 0
    for chunk in reader:
        if max_rows > 0:
            remaining = max_rows - seen
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk.iloc[:remaining].copy()
        seen += len(chunk)
        yield chunk


def build_publications_matches(
    publications_path: Path,
    cdli_lookup: dict[str, list[PublishedRow]],
    ref_lookup: dict[str, list[PublishedRow]],
    chunk_size: int,
    max_rows: int,
) -> pd.DataFrame:
    """Create page-to-tablet matches using CDLI ids in OCR pages."""
    out_rows: list[dict] = []

    for chunk in iter_publications_chunks(publications_path, chunk_size, max_rows):
        for row in chunk.itertuples(index=False):
            pdf_name = str(getattr(row, "pdf_name", ""))
            page = getattr(row, "page", "")
            page_text = str(getattr(row, "page_text", ""))
            has_akkadian = getattr(row, "has_akkadian", "")

            page_cdli = extract_cdli_ids(page_text)
            name_cdli = extract_cdli_ids(pdf_name)
            page_refs = extract_publication_refs(page_text)
            name_refs = extract_publication_refs(pdf_name)

            # High confidence: IDs appearing in OCR page text.
            for cdli in page_cdli:
                for candidate in cdli_lookup.get(cdli, []):
                    out_rows.append(
                        {
                            "pdf_name": pdf_name,
                            "page": page,
                            "has_akkadian": has_akkadian,
                            "matched_cdli": cdli,
                            "match_source": "page_text_cdli",
                            "match_confidence": 0.95,
                            "oare_id": candidate.oare_id,
                            "label": candidate.label,
                            "aliases": candidate.aliases,
                            "publication_catalog": candidate.publication_catalog,
                            "transliteration": candidate.transliteration,
                            "page_text_snippet": snippet_around_match(page_text, cdli),
                            "page_text": normalize_space(page_text),
                        }
                    )

            # Medium confidence: IDs appearing only in filename/title.
            # Kept as weak candidates for optional LLM review.
            for cdli in name_cdli:
                for candidate in cdli_lookup.get(cdli, []):
                    out_rows.append(
                        {
                            "pdf_name": pdf_name,
                            "page": page,
                            "has_akkadian": has_akkadian,
                            "matched_cdli": cdli,
                            "match_source": "pdf_name_cdli",
                            "match_confidence": 0.70,
                            "oare_id": candidate.oare_id,
                            "label": candidate.label,
                            "aliases": candidate.aliases,
                            "publication_catalog": candidate.publication_catalog,
                            "transliteration": candidate.transliteration,
                            "page_text_snippet": snippet_around_match(page_text, cdli),
                            "page_text": normalize_space(page_text),
                        }
                    )

            for ref in page_refs:
                for candidate in ref_lookup.get(ref, []):
                    out_rows.append(
                        {
                            "pdf_name": pdf_name,
                            "page": page,
                            "has_akkadian": has_akkadian,
                            "matched_cdli": "",
                            "match_source": "page_text_ref",
                            "match_confidence": 0.82,
                            "oare_id": candidate.oare_id,
                            "label": candidate.label,
                            "aliases": candidate.aliases,
                            "publication_catalog": candidate.publication_catalog,
                            "transliteration": candidate.transliteration,
                            "page_text_snippet": snippet_around_match(page_text, ref),
                            "page_text": normalize_space(page_text),
                        }
                    )

            for ref in name_refs:
                for candidate in ref_lookup.get(ref, []):
                    out_rows.append(
                        {
                            "pdf_name": pdf_name,
                            "page": page,
                            "has_akkadian": has_akkadian,
                            "matched_cdli": "",
                            "match_source": "pdf_name_ref",
                            "match_confidence": 0.76,
                            "oare_id": candidate.oare_id,
                            "label": candidate.label,
                            "aliases": candidate.aliases,
                            "publication_catalog": candidate.publication_catalog,
                            "transliteration": candidate.transliteration,
                            "page_text_snippet": snippet_around_match(page_text, ref),
                            "page_text": normalize_space(page_text),
                        }
                    )

    if not out_rows:
        return pd.DataFrame(
            columns=[
                "pdf_name",
                "page",
                "has_akkadian",
                "matched_cdli",
                "match_source",
                "match_confidence",
                "oare_id",
                "label",
                "aliases",
                "publication_catalog",
                "transliteration",
                "page_text_snippet",
                "page_text",
            ]
        )

    matches = pd.DataFrame(out_rows)
    matches = matches.drop_duplicates(subset=["pdf_name", "page", "oare_id", "matched_cdli", "match_source"]).reset_index(drop=True)
    return matches


def build_llm_input(matches: pd.DataFrame) -> pd.DataFrame:
    """Create LLM-ready alignment rows from deterministic matches.

    LLM stage idea:
    - Given transliteration + page_text, extract exact translation segment for this tablet.
    - If non-English, translate to English.
    - Return confidence + rationale.
    """
    if matches.empty:
        return matches.copy()

    llm_df = matches.copy()
    llm_df["llm_task"] = (
        "Find the translation segment in page_text that corresponds to the given transliteration/tablet; "
        "if translation is not English, translate to English; return a single clean translation text and confidence."
    )
    llm_df["llm_ready"] = 1
    return llm_df


def run(args: argparse.Namespace) -> None:
    """Run deterministic extraction and build LLM-ready candidate tables."""
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    published_texts_path = input_dir / args.published_texts_file
    publications_path = input_dir / args.publications_file

    if not published_texts_path.exists():
        raise FileNotFoundError(f"Missing file: {published_texts_path}")
    if not publications_path.exists():
        raise FileNotFoundError(f"Missing file: {publications_path}")

    cdli_lookup, ref_lookup, _ = build_cdli_lookup(published_texts_path)
    matches = build_publications_matches(
        publications_path=publications_path,
        cdli_lookup=cdli_lookup,
        ref_lookup=ref_lookup,
        chunk_size=args.chunk_size,
        max_rows=args.max_rows,
    )
    llm_input = build_llm_input(matches)

    matches_path = output_dir / "publications_cdli_matches.csv"
    llm_input_path = output_dir / "publications_llm_alignment_input.csv"
    summary_path = output_dir / "publications_extraction_summary.csv"

    matches.to_csv(matches_path, index=False)
    llm_input.to_csv(llm_input_path, index=False)

    summary = {
        "input_dir": str(input_dir),
        "published_texts_rows": "",
        "publications_max_rows": args.max_rows,
        "chunk_size": args.chunk_size,
        "matches_rows": int(len(matches)),
        "high_confidence_rows": int((matches["match_confidence"] >= 0.9).sum()) if not matches.empty else 0,
        "unique_oare_ids": int(matches["oare_id"].nunique()) if not matches.empty else 0,
        "unique_pdfs": int(matches["pdf_name"].nunique()) if not matches.empty else 0,
    }
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    print("Publications extraction completed.")
    print(f"Input dir: {input_dir}")
    print(f"Output dir: {output_dir}")
    print("Generated files:")
    print(" - publications_cdli_matches.csv")
    print(" - publications_llm_alignment_input.csv")
    print(" - publications_extraction_summary.csv")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI arguments for stage-1 extraction."""
    parser = argparse.ArgumentParser(description="Deep Past publications extractor (stage-1 deterministic)")
    parser.add_argument("--input-dir", type=str, default=DEFAULT_INPUT_DIR, help="Directory containing source CSV files")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory for generated CSV files")
    parser.add_argument("--published-texts-file", type=str, default=DEFAULT_PUBLISHED_TEXTS, help="Published texts CSV filename")
    parser.add_argument("--publications-file", type=str, default=DEFAULT_PUBLICATIONS, help="Publications CSV filename")
    parser.add_argument("--chunk-size", type=int, default=2500, help="Chunk size for streaming large publications CSV")
    parser.add_argument("--max-rows", type=int, default=0, help="Limit processed publication rows (0 = all rows)")
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    run(args)
