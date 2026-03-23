"""Stage-2 LLM runner for publications alignment.

Reads shortlist rows from stage-1 extraction, calls an OpenAI-compatible endpoint
(e.g., DigitalOcean serverless), and writes:

1) Raw audit log (JSONL)
2) Flattened silver sentence-pair CSV (one-to-many from each input row)
3) Run summary CSV

Non-streaming API is used.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI

DEFAULT_INPUT_CSV = "data/processed/publications_llm_alignment_shortlist.csv"
DEFAULT_OUTPUT_DIR = "data/processed/llm_stage2"
DEFAULT_BASE_URL = "https://inference.do-ai.run/v1"
DEFAULT_MODEL = "gpt-oss-20b"

JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


def load_env_file() -> None:
    """Load .env values (simple parser) from scripts/.env or project-root/.env."""
    candidates = [Path(".env"), Path("..").resolve() / ".env"]
    for path in candidates:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def estimate_tokens_from_text(text: str) -> int:
    """Very rough token estimate for pre-call throttling."""
    if not text:
        return 0
    return max(1, len(text) // 4)


@dataclass
class WindowLimiter:
    """Simple rolling-window limiter for requests/min and tokens/min."""

    rpm: int
    tpm: int

    def __post_init__(self) -> None:
        self.req_times: deque[float] = deque()
        self.token_events: deque[tuple[float, int]] = deque()

    def _cleanup(self, now_ts: float) -> None:
        cutoff = now_ts - 60.0
        while self.req_times and self.req_times[0] < cutoff:
            self.req_times.popleft()
        while self.token_events and self.token_events[0][0] < cutoff:
            self.token_events.popleft()

    def _tokens_in_window(self) -> int:
        return sum(tokens for _, tokens in self.token_events)

    def wait_for_slot(self, estimated_tokens: int) -> None:
        while True:
            now_ts = time.time()
            self._cleanup(now_ts)

            req_ok = len(self.req_times) < self.rpm if self.rpm > 0 else True
            tok_ok = (self._tokens_in_window() + estimated_tokens) <= self.tpm if self.tpm > 0 else True

            if req_ok and tok_ok:
                self.req_times.append(now_ts)
                return
            time.sleep(0.5)

    def record_tokens(self, used_tokens: int) -> None:
        if used_tokens <= 0:
            return
        self.token_events.append((time.time(), int(used_tokens)))


def parse_json_from_text(content: str) -> dict[str, Any]:
    """Parse first JSON object in model text output."""
    if not content:
        return {}

    content = content.strip()

    # Direct parse first.
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # Fallback: find first object-like span.
    match = JSON_OBJECT_PATTERN.search(content)
    if not match:
        return {}

    try:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return {}

    return {}


def make_input_key(row: pd.Series) -> str:
    return f"{row.get('pdf_name','')}||{row.get('page','')}||{row.get('oare_id','')}"


def build_prompt(row: pd.Series) -> list[dict[str, str]]:
    transliteration = str(row.get("transliteration", ""))
    page_text = str(row.get("page_text", ""))
    page_snippet = str(row.get("page_text_snippet", ""))

    developer_content = (
        "You are an expert data-alignment assistant for Old Assyrian transliteration and OCR publication text. "
        "Return strict JSON only. No markdown, no code fences, no extra commentary."
    )

    user_content = f"""
Given:
- tablet id: {row.get('oare_id','')}
- publication: {row.get('pdf_name','')}
- page: {row.get('page','')}
- transliteration: {transliteration}
- stage1 match source: {row.get('match_source','')}
- stage1 match confidence: {row.get('match_confidence','')}
- OCR page snippet: {page_snippet}
- OCR page text: {page_text}

Tasks:
1) Determine if this page contains a translation for this tablet.
2) Extract the most likely matching translation segment.
3) If not English, translate to English.
4) Split into sentence-level pairs when possible.
5) Return confidence and short evidence.

Return strict JSON with this schema:
{{
  "found": true,
  "oare_id": "...",
  "source_pdf": "...",
  "source_page": "...",
  "language_detected": "en|de|fr|tr|other",
  "confidence": 0.0,
  "translation_extracted": "...",
  "translation_english": "...",
  "pairs": [
    {{"source_sentence": "...", "target_sentence_english": "..."}}
  ],
  "evidence": "...",
  "notes": "..."
}}

If not reliable, return:
{{
  "found": false,
  "oare_id": "...",
  "source_pdf": "...",
  "source_page": "...",
  "confidence": 0.0,
  "evidence": "...",
  "notes": "..."
}}
""".strip()

    return [
        {"role": "developer", "content": developer_content},
        {"role": "user", "content": user_content},
    ]


def load_processed_keys(raw_jsonl_path: Path) -> set[str]:
    keys: set[str] = set()
    if not raw_jsonl_path.exists():
        return keys

    with raw_jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                key = obj.get("input_key")
                if key:
                    keys.add(str(key))
            except Exception:
                continue
    return keys


def flatten_silver_pairs(df_input: pd.DataFrame, raw_jsonl_path: Path, min_conf: float) -> pd.DataFrame:
    """Convert raw JSONL outputs to one-to-many sentence pairs."""
    if not raw_jsonl_path.exists():
        return pd.DataFrame()

    # Build quick lookup by input_key to recover stage-1 fields.
    idx = df_input.copy()
    idx["input_key"] = idx.apply(make_input_key, axis=1)
    by_key = {str(k): row for k, row in idx.set_index("input_key").iterrows()}

    out_rows: list[dict[str, Any]] = []

    with raw_jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            input_key = str(rec.get("input_key", ""))
            parsed = rec.get("parsed", {}) or {}
            stage1 = by_key.get(input_key)
            if stage1 is None:
                continue

            found = bool(parsed.get("found", False))
            conf = float(parsed.get("confidence", 0.0) or 0.0)
            if not found or conf < min_conf:
                continue

            pairs = parsed.get("pairs", [])
            if isinstance(pairs, list) and pairs:
                for i, pair in enumerate(pairs):
                    source_sentence = str(pair.get("source_sentence", "")).strip()
                    target_sentence = str(pair.get("target_sentence_english", "")).strip()
                    if not target_sentence:
                        continue
                    out_rows.append(
                        {
                            "oare_id": stage1.get("oare_id", ""),
                            "source_transliteration": source_sentence if source_sentence else stage1.get("transliteration", ""),
                            "target_translation_english": target_sentence,
                            "source_pdf": stage1.get("pdf_name", ""),
                            "source_page": stage1.get("page", ""),
                            "stage1_match_source": stage1.get("match_source", ""),
                            "stage1_match_confidence": stage1.get("match_confidence", ""),
                            "stage2_confidence": conf,
                            "language_detected": parsed.get("language_detected", "other"),
                            "evidence": parsed.get("evidence", ""),
                            "notes": parsed.get("notes", ""),
                            "pair_index": i,
                            "input_key": input_key,
                        }
                    )
            else:
                fallback_target = str(parsed.get("translation_english", "")).strip()
                if fallback_target:
                    out_rows.append(
                        {
                            "oare_id": stage1.get("oare_id", ""),
                            "source_transliteration": stage1.get("transliteration", ""),
                            "target_translation_english": fallback_target,
                            "source_pdf": stage1.get("pdf_name", ""),
                            "source_page": stage1.get("page", ""),
                            "stage1_match_source": stage1.get("match_source", ""),
                            "stage1_match_confidence": stage1.get("match_confidence", ""),
                            "stage2_confidence": conf,
                            "language_detected": parsed.get("language_detected", "other"),
                            "evidence": parsed.get("evidence", ""),
                            "notes": parsed.get("notes", ""),
                            "pair_index": 0,
                            "input_key": input_key,
                        }
                    )

    if not out_rows:
        return pd.DataFrame()

    silver = pd.DataFrame(out_rows)
    silver = silver.drop_duplicates(
        subset=["oare_id", "source_pdf", "source_page", "source_transliteration", "target_translation_english"]
    ).reset_index(drop=True)
    return silver


def run(args: argparse.Namespace) -> None:
    load_env_file()

    api_key = os.getenv(args.api_key_env, "")
    if not api_key:
        raise RuntimeError(
            f"API key not found in env var '{args.api_key_env}'. Add it to .env or environment variables."
        )

    client = OpenAI(api_key=api_key, base_url=args.base_url)

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_csv.exists():
        raise FileNotFoundError(f"Missing input CSV: {input_csv}")

    df = pd.read_csv(input_csv)
    if args.max_rows > 0:
        df = df.head(args.max_rows).copy()

    required_cols = ["oare_id", "pdf_name", "page", "transliteration", "page_text", "page_text_snippet"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")

    raw_jsonl_path = output_dir / "llm_raw_responses.jsonl"
    silver_csv_path = output_dir / "silver_sentence_pairs.csv"
    summary_csv_path = output_dir / "llm_stage2_run_summary.csv"

    processed_keys = load_processed_keys(raw_jsonl_path) if args.resume else set()

    limiter = WindowLimiter(rpm=args.rpm, tpm=args.tpm)

    total_rows = len(df)
    sent = 0
    skipped = 0
    parse_fail = 0

    with raw_jsonl_path.open("a", encoding="utf-8") as raw_f:
        for row in df.itertuples(index=False):
            row_series = pd.Series(row._asdict())
            input_key = make_input_key(row_series)
            if input_key in processed_keys:
                skipped += 1
                continue

            messages = build_prompt(row_series)
            estimated_tokens = estimate_tokens_from_text(messages[1]["content"])
            limiter.wait_for_slot(estimated_tokens=estimated_tokens)

            call_started = time.time()
            response_text = ""
            parsed: dict[str, Any] = {}
            error_text = ""
            usage_prompt_tokens = 0
            usage_completion_tokens = 0
            usage_total_tokens = 0

            for attempt in range(max(1, args.max_retries)):
                try:
                    resp = client.chat.completions.create(
                        model=args.model,
                        messages=messages,
                        temperature=args.temperature,
                        max_completion_tokens=args.max_completion_tokens,
                    )
                    response_text = (resp.choices[0].message.content or "").strip()
                    parsed = parse_json_from_text(response_text)

                    usage = getattr(resp, "usage", None)
                    if usage is not None:
                        usage_prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                        usage_completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
                        usage_total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
                    limiter.record_tokens(usage_total_tokens)
                    break
                except Exception as exc:
                    error_text = str(exc)
                    if attempt + 1 >= max(1, args.max_retries):
                        break
                    time.sleep(min(2 ** attempt, 8))

            if response_text and not parsed:
                parse_fail += 1

            elapsed_ms = int((time.time() - call_started) * 1000)

            record = {
                "run_timestamp_utc": now_utc_iso(),
                "input_key": input_key,
                "oare_id": row_series.get("oare_id", ""),
                "pdf_name": row_series.get("pdf_name", ""),
                "page": row_series.get("page", ""),
                "match_source": row_series.get("match_source", ""),
                "match_confidence": row_series.get("match_confidence", ""),
                "elapsed_ms": elapsed_ms,
                "usage_prompt_tokens": usage_prompt_tokens,
                "usage_completion_tokens": usage_completion_tokens,
                "usage_total_tokens": usage_total_tokens,
                "error": error_text,
                "raw_text": response_text,
                "parsed": parsed,
            }
            raw_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            raw_f.flush()

            sent += 1
            if sent % 50 == 0:
                print(f"Processed {sent}/{total_rows} rows (skipped={skipped}, parse_fail={parse_fail})")

    silver_df = flatten_silver_pairs(df_input=df, raw_jsonl_path=raw_jsonl_path, min_conf=args.min_confidence)
    silver_df.to_csv(silver_csv_path, index=False)

    summary = {
        "run_timestamp_utc": now_utc_iso(),
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "model": args.model,
        "base_url": args.base_url,
        "total_input_rows": int(total_rows),
        "processed_rows": int(sent),
        "skipped_rows": int(skipped),
        "parse_fail_rows": int(parse_fail),
        "silver_rows": int(len(silver_df)),
        "silver_unique_oare_ids": int(silver_df["oare_id"].nunique()) if not silver_df.empty else 0,
        "rpm": args.rpm,
        "tpm": args.tpm,
        "max_completion_tokens": args.max_completion_tokens,
        "min_confidence": args.min_confidence,
        "resume": args.resume,
    }
    pd.DataFrame([summary]).to_csv(summary_csv_path, index=False)

    print("Stage-2 LLM run completed.")
    print(f"Input: {input_csv}")
    print(f"Output directory: {output_dir}")
    print("Generated files:")
    print(" - llm_raw_responses.jsonl")
    print(" - silver_sentence_pairs.csv")
    print(" - llm_stage2_run_summary.csv")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage-2 LLM runner for publications alignment")
    parser.add_argument("--input-csv", type=str, default=DEFAULT_INPUT_CSV, help="LLM shortlist input CSV")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory")

    parser.add_argument("--api-key-env", type=str, default="DO_AI_API_KEY", help="Environment variable name holding API key")
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL, help="OpenAI-compatible base URL")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model name")

    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature")
    parser.add_argument("--max-completion-tokens", type=int, default=600, help="Max completion tokens")

    parser.add_argument("--rpm", type=int, default=40, help="Requests per minute limit (0 to disable)")
    parser.add_argument("--tpm", type=int, default=200000, help="Tokens per minute limit (0 to disable)")

    parser.add_argument("--max-rows", type=int, default=0, help="Process only first N rows (0 = all)")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries per row on API failure")
    parser.add_argument("--min-confidence", type=float, default=0.80, help="Min confidence for silver output")
    parser.add_argument("--resume", action="store_true", default=True, help="Skip rows already present in raw JSONL")
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    run(args)
