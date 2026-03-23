"""LLM-based sentence alignment pipeline for train.csv + Sentences_Oare_FirstWord_LinNum.csv.

What this script does (simple and explicit):
1) Match documents by id between train.csv and sentence map CSV.
2) Track matched, unmatched, and processed document status in separate CSV files.
3) Send document + sentence clues to an OpenAI-compatible endpoint for alignment.
4) Build cleaned sentence-level training dataset: train_dataset_sl_clean.csv.
5) Print continuous progress messages in console.

Environment variables (.env supported):
- DO_AI_API_KEY
- DO_AI_BASE_URL (default: https://inference.do-ai.run/v1)
- DO_AI_MODEL (default: openai-gpt-oss-20b)

Run example:
    python scripts/llm_alignment_pipeline.py --max-docs 50
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI

DEFAULT_INPUT_DIR = "C:\\Users\\barad\\Desktop\\Kaggle\\DeepPast\\data\\deep-past-data"
DEFAULT_OUTPUT_DIR = "data/processed/llm_alignment"


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def load_env_file() -> None:
    env_path = Path(".env")
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def pick_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    if required:
        raise KeyError(f"Missing required column. Tried: {candidates}")
    return None


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
    value = re.sub(r"\[x\]", " <gap> ", value)
    value = re.sub(r"\[\s*\.\.\.\s*\]", " <big_gap> ", value)
    value = re.sub(r"\.\.\.", " <big_gap> ", value)
    value = value.translate(SUBSCRIPT_MAP)
    value = re.sub(r"[!?/]", " ", value)
    value = value.replace(":", " ").replace(".", " ")
    value = value.replace("[", "").replace("]", "")
    value = value.replace("<", "").replace(">", "")
    return re.sub(r"\s+", " ", value).strip()


def normalize_translation(text: str) -> str:
    if pd.isna(text):
        return ""
    value = str(text).replace("<", "").replace(">", "")
    return re.sub(r"\s+", " ", value).strip()


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def parse_json_safely(content: str) -> dict[str, Any] | None:
    if not content:
        return None

    raw = content.strip()
    raw = re.sub(r"^```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
    raw = re.sub(r"```$", "", raw).strip()

    try:
        return json.loads(raw)
    except Exception:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    snippet = raw[start : end + 1]
    try:
        return json.loads(snippet)
    except Exception:
        return None


def build_prompt_payload(
    oare_id: str,
    doc_transliteration: str,
    doc_translation: str,
    clue_rows: list[dict[str, Any]],
) -> str:
    schema = {
        "oare_id": "string",
        "pairs": [
            {
                "source_sentence": "string",
                "target_sentence": "string",
                "source_method": "clue|inferred",
                "target_method": "clue|inferred",
                "confidence": 0.0,
                "note": "string",
            }
        ],
        "leftover_source_text": "string",
        "leftover_target_text": "string",
    }

    instructions = {
        "task": "Align sentence pairs between full transliteration and full English translation using clues.",
        "hard_rules": [
            "Do NOT hallucinate facts, words, or meanings not present in provided text.",
            "Do NOT add commentary outside JSON.",
            "Use only material from the given transliteration/translation.",
            "source_sentence and target_sentence must be exact substrings copied from the provided document text.",
            "Do not rewrite, paraphrase, normalize, or fix spelling in extracted sentences.",
            "If clue coverage is incomplete, infer additional sentence boundaries conservatively to maximize coverage.",
            "If uncertain, still provide best local alignment with lower confidence instead of inventing content.",
            "Keep source/target language as provided: source Akkadian transliteration, target English translation.",
        ],
        "goal": "Produce as many valid aligned sentence pairs as possible from this document.",
        "output_json_schema": schema,
    }

    payload = {
        "instructions": instructions,
        "document": {
            "oare_id": oare_id,
            "full_transliteration": doc_transliteration,
            "full_translation": doc_translation,
            "sentence_clues": clue_rows,
        },
    }
    return json.dumps(payload, ensure_ascii=False)


def build_second_pass_payload(
    oare_id: str,
    leftover_source_text: str,
    leftover_target_text: str,
    existing_pairs: list[dict[str, Any]],
) -> str:
    schema = {
        "oare_id": "string",
        "pairs": [
            {
                "source_sentence": "string",
                "target_sentence": "string",
                "source_method": "leftover_inferred",
                "target_method": "leftover_inferred",
                "confidence": 0.0,
                "note": "string",
            }
        ],
        "leftover_source_text": "string",
        "leftover_target_text": "string",
    }

    instructions = {
        "task": "Second-pass alignment over leftovers only.",
        "hard_rules": [
            "Do NOT hallucinate.",
            "Return strict JSON only.",
            "Extract source/target as exact substrings from the leftover texts.",
            "Do not duplicate any existing aligned pair.",
            "Do not rewrite or paraphrase.",
        ],
        "goal": "Recover additional valid sentence pairs from leftovers.",
        "output_json_schema": schema,
    }

    payload = {
        "instructions": instructions,
        "document": {
            "oare_id": oare_id,
            "leftover_source_text": leftover_source_text,
            "leftover_target_text": leftover_target_text,
            "existing_pairs": existing_pairs,
        },
    }
    return json.dumps(payload, ensure_ascii=False)


def _contains_exact_substring(container_text: str, candidate: str) -> bool:
    if not candidate:
        return False
    return candidate in container_text


def _find_position(container_text: str, candidate: str, start_pos: int = 0) -> int:
    if not candidate:
        return -1
    return container_text.find(candidate, max(0, start_pos))


def _token_set(text: str) -> set[str]:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip().lower()
    if not cleaned:
        return set()
    return set(cleaned.split())


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    if union == 0:
        return 0.0
    return inter / union


def validate_aligned_pairs(
    pairs: list[dict[str, Any]],
    doc_transliteration: str,
    doc_translation: str,
    existing_pair_keys: set[tuple[str, str]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Validate LLM output pairs with strict checks for training safety."""
    accepted: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []

    pair_keys = set(existing_pair_keys or set())
    last_src_pos = -1
    last_tgt_pos = -1
    last_src_tokens: set[str] = set()

    stats = {
        "input_pairs": len(pairs),
        "accepted_pairs": 0,
        "rejected_empty": 0,
        "rejected_not_substring": 0,
        "rejected_duplicate": 0,
        "rejected_order": 0,
        "rejected_overlap": 0,
    }

    for idx, pair in enumerate(pairs, start=1):
        if not isinstance(pair, dict):
            issues.append({"pair_index": idx, "reason": "not_dict"})
            continue

        src_raw = clean_text(pair.get("source_sentence", ""))
        tgt_raw = clean_text(pair.get("target_sentence", ""))
        if not src_raw or not tgt_raw:
            stats["rejected_empty"] += 1
            issues.append({"pair_index": idx, "reason": "empty_source_or_target"})
            continue

        if not _contains_exact_substring(doc_transliteration, src_raw) or not _contains_exact_substring(doc_translation, tgt_raw):
            stats["rejected_not_substring"] += 1
            issues.append({"pair_index": idx, "reason": "not_exact_substring", "source": src_raw, "target": tgt_raw})
            continue

        pair_key = (src_raw, tgt_raw)
        if pair_key in pair_keys:
            stats["rejected_duplicate"] += 1
            issues.append({"pair_index": idx, "reason": "duplicate_pair", "source": src_raw, "target": tgt_raw})
            continue

        src_pos = _find_position(doc_transliteration, src_raw, last_src_pos + 1)
        tgt_pos = _find_position(doc_translation, tgt_raw, last_tgt_pos + 1)
        if src_pos == -1 or tgt_pos == -1:
            stats["rejected_order"] += 1
            issues.append({"pair_index": idx, "reason": "order_position_not_found", "source": src_raw, "target": tgt_raw})
            continue

        if src_pos < last_src_pos or tgt_pos < last_tgt_pos:
            stats["rejected_order"] += 1
            issues.append({"pair_index": idx, "reason": "non_monotonic_order", "source": src_raw, "target": tgt_raw})
            continue

        src_tokens = _token_set(src_raw)
        if last_src_tokens and _jaccard(src_tokens, last_src_tokens) >= 0.95:
            stats["rejected_overlap"] += 1
            issues.append({"pair_index": idx, "reason": "near_duplicate_overlap", "source": src_raw, "target": tgt_raw})
            continue

        accepted.append(
            {
                "pair_index": idx,
                "source_sentence_raw": src_raw,
                "target_sentence_raw": tgt_raw,
                "source_method": clean_text(pair.get("source_method", "")) or "inferred",
                "target_method": clean_text(pair.get("target_method", "")) or "inferred",
                "alignment_confidence": pair.get("confidence", None),
                "alignment_note": clean_text(pair.get("note", "")),
                "source_pos": src_pos,
                "target_pos": tgt_pos,
            }
        )

        pair_keys.add(pair_key)
        last_src_pos = src_pos
        last_tgt_pos = tgt_pos
        last_src_tokens = src_tokens

    stats["accepted_pairs"] = len(accepted)
    return accepted, issues, stats


def call_alignment_llm(
    client: OpenAI,
    model: str,
    prompt_payload: str,
    max_completion_tokens: int,
    retries: int,
    sleep_seconds: float,
) -> tuple[dict[str, Any] | None, str, str]:
    last_error = ""
    raw_content = ""

    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "developer",
                        "content": (
                            "You are a strict alignment engine. "
                            "Return JSON only. Never add extra prose. "
                            "Never hallucinate or invent text beyond provided inputs."
                        ),
                    },
                    {"role": "user", "content": prompt_payload},
                ],
                temperature=0,
                max_completion_tokens=max_completion_tokens,
            )
            raw_content = clean_text(response.choices[0].message.content or "")
            parsed = parse_json_safely(raw_content)
            if parsed is not None:
                return parsed, raw_content, ""
            last_error = "Model response was not valid JSON"
        except Exception as exc:
            last_error = str(exc)

        log(f"LLM retry {attempt}/{retries} failed: {last_error}")
        if attempt < retries:
            time.sleep(sleep_seconds)

    return None, raw_content, last_error


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def run(args: argparse.Namespace) -> int:
    load_env_file()

    api_key = os.getenv("DO_AI_API_KEY", "")
    base_url = os.getenv("DO_AI_BASE_URL", "https://inference.do-ai.run/v1")
    model = args.model or os.getenv("DO_AI_MODEL", "openai-gpt-oss-20b")

    if not api_key:
        log("ERROR: Missing DO_AI_API_KEY in environment or .env")
        return 1

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = input_dir / "train.csv"
    sentence_path = input_dir / "Sentences_Oare_FirstWord_LinNum.csv"

    if not train_path.exists():
        log(f"ERROR: Missing {train_path}")
        return 1
    if not sentence_path.exists():
        log(f"ERROR: Missing {sentence_path}")
        return 1

    log("Loading CSV files...")
    train_df = pd.read_csv(train_path)
    sent_df = pd.read_csv(sentence_path)

    train_id_col = pick_col(train_df, ["oare_id"])
    train_src_col = pick_col(train_df, ["transliteration"])
    train_tgt_col = pick_col(train_df, ["translation"])

    sent_id_col = pick_col(sent_df, ["text_uuid", "oare_id"])
    sent_first_word_col = pick_col(sent_df, ["first_word_transcription"], required=False)
    sent_translation_col = pick_col(sent_df, ["translation"], required=False)
    sent_line_col = pick_col(sent_df, ["line_number"], required=False)
    sent_order_col = pick_col(sent_df, ["sentence_obj_in_text", "first_word_obj_in_text"], required=False)

    train_df[train_id_col] = train_df[train_id_col].astype(str)
    sent_df[sent_id_col] = sent_df[sent_id_col].astype(str)

    train_ids = set(train_df[train_id_col].unique())
    sent_ids = set(sent_df[sent_id_col].unique())
    matched_ids = sorted(train_ids & sent_ids)
    train_only_ids = sorted(train_ids - sent_ids)
    sentence_only_ids = sorted(sent_ids - train_ids)

    log(
        "ID summary: "
        f"train_docs={len(train_ids)} | sentence_map_docs={len(sent_ids)} | "
        f"matched_docs={len(matched_ids)} | train_only={len(train_only_ids)} | sentence_only={len(sentence_only_ids)}"
    )

    unmatched_rows = []
    unmatched_rows.extend(
        [{"document_id": doc_id, "unmatched_source": "train_only", "reason": "no_sentence_map_id_match"} for doc_id in train_only_ids]
    )
    unmatched_rows.extend(
        [{"document_id": doc_id, "unmatched_source": "sentence_map_only", "reason": "no_train_id_match"} for doc_id in sentence_only_ids]
    )
    unmatched_df = pd.DataFrame(unmatched_rows)
    unmatched_path = output_dir / "alignment_unmatched_documents.csv"
    save_csv(unmatched_df, unmatched_path)

    doc_status_path = output_dir / "alignment_document_status.csv"
    raw_jsonl_path = output_dir / "alignment_raw_responses.jsonl"
    dataset_path = output_dir / "train_dataset_sl_clean.csv"
    leftover_debug_path = output_dir / "alignment_leftovers_debug.csv"
    validation_issues_path = output_dir / "alignment_validation_issues.csv"

    existing_status_df = pd.DataFrame()
    processed_completed_ids: set[str] = set()
    if args.resume and doc_status_path.exists():
        existing_status_df = pd.read_csv(doc_status_path)
        if "oare_id" in existing_status_df.columns and "status" in existing_status_df.columns:
            processed_completed_ids = set(
                existing_status_df.loc[existing_status_df["status"] == "completed", "oare_id"].astype(str).tolist()
            )
        log(f"Resume enabled: found {len(processed_completed_ids)} completed docs in status file.")

    if args.max_docs > 0:
        matched_ids = matched_ids[: args.max_docs]

    if args.shuffle:
        matched_ids = sorted(matched_ids)

    client = OpenAI(api_key=api_key, base_url=base_url)

    train_lookup = train_df.set_index(train_id_col, drop=False)

    result_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    leftover_debug_rows: list[dict[str, Any]] = []
    validation_issue_rows: list[dict[str, Any]] = []

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log(f"Starting LLM alignment run_id={run_id} | model={model} | base_url={base_url}")

    raw_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    raw_handle = raw_jsonl_path.open("a", encoding="utf-8")

    total_docs = len(matched_ids)
    for index, doc_id in enumerate(matched_ids, start=1):
        if args.resume and doc_id in processed_completed_ids:
            log(f"[{index}/{total_docs}] SKIP already completed doc {doc_id}")
            continue

        if doc_id not in train_lookup.index:
            status_rows.append(
                {
                    "run_id": run_id,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "oare_id": doc_id,
                    "status": "skipped",
                    "reason": "missing_train_row",
                    "expected_clue_rows": 0,
                    "produced_pairs": 0,
                }
            )
            continue

        train_row = train_lookup.loc[doc_id]
        doc_transliteration = clean_text(train_row[train_src_col])
        doc_translation = clean_text(train_row[train_tgt_col])

        doc_sentence_rows = sent_df[sent_df[sent_id_col] == doc_id].copy()
        if sent_order_col and sent_order_col in doc_sentence_rows.columns:
            doc_sentence_rows = doc_sentence_rows.sort_values(sent_order_col, kind="stable")

        clue_rows = []
        for _, row in doc_sentence_rows.iterrows():
            clue_rows.append(
                {
                    "sentence_obj_in_text": row.get("sentence_obj_in_text", None),
                    "first_word_obj_in_text": row.get("first_word_obj_in_text", None),
                    "line_number": row.get(sent_line_col, None) if sent_line_col else None,
                    "first_word_transcription": row.get(sent_first_word_col, None) if sent_first_word_col else None,
                    "sentence_translation_hint": row.get(sent_translation_col, None) if sent_translation_col else None,
                }
            )

        if args.max_doc_chars > 0:
            doc_transliteration = doc_transliteration[: args.max_doc_chars]
            doc_translation = doc_translation[: args.max_doc_chars]

        log(
            f"[{index}/{total_docs}] Processing {doc_id} | clue_rows={len(clue_rows)} | "
            f"src_chars={len(doc_transliteration)} | tgt_chars={len(doc_translation)}"
        )

        payload = build_prompt_payload(
            oare_id=doc_id,
            doc_transliteration=doc_transliteration,
            doc_translation=doc_translation,
            clue_rows=clue_rows,
        )

        parsed, raw_content, error = call_alignment_llm(
            client=client,
            model=model,
            prompt_payload=payload,
            max_completion_tokens=args.max_completion_tokens,
            retries=args.retries,
            sleep_seconds=args.retry_sleep_seconds,
        )

        raw_handle.write(
            json.dumps(
                {
                    "run_id": run_id,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "oare_id": doc_id,
                    "raw_response": raw_content,
                    "parse_ok": parsed is not None,
                    "error": error,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        raw_handle.flush()

        if parsed is None:
            log(f"[{index}/{total_docs}] ERROR {doc_id}: {error}")
            status_rows.append(
                {
                    "run_id": run_id,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "oare_id": doc_id,
                    "status": "llm_error",
                    "reason": error,
                    "expected_clue_rows": len(clue_rows),
                    "produced_pairs": 0,
                }
            )
            continue

        pairs = parsed.get("pairs", []) if isinstance(parsed, dict) else []

        accepted_pairs, pair_issues, pair_stats = validate_aligned_pairs(
            pairs=pairs,
            doc_transliteration=doc_transliteration,
            doc_translation=doc_translation,
        )

        for issue in pair_issues:
            validation_issue_rows.append(
                {
                    "run_id": run_id,
                    "oare_id": doc_id,
                    "pass": "first_pass",
                    **issue,
                }
            )

        produced = 0
        existing_pair_keys = set()
        for accepted in accepted_pairs:
            src_raw = accepted["source_sentence_raw"]
            tgt_raw = accepted["target_sentence_raw"]
            existing_pair_keys.add((src_raw, tgt_raw))
            result_rows.append(
                {
                    "oare_id": doc_id,
                    "pair_index": accepted["pair_index"],
                    "alignment_pass": "first_pass",
                    "source_sentence_raw": src_raw,
                    "target_sentence_raw": tgt_raw,
                    "source_sentence_clean": normalize_transliteration(src_raw),
                    "target_sentence_clean": normalize_translation(tgt_raw),
                    "source_method": accepted["source_method"],
                    "target_method": accepted["target_method"],
                    "alignment_confidence": accepted["alignment_confidence"],
                    "alignment_note": accepted["alignment_note"],
                    "source_pos": accepted["source_pos"],
                    "target_pos": accepted["target_pos"],
                    "doc_translation": doc_translation,
                    "doc_transliteration": doc_transliteration,
                    "run_id": run_id,
                }
            )
            produced += 1

        leftover_source = clean_text(parsed.get("leftover_source_text", ""))
        leftover_target = clean_text(parsed.get("leftover_target_text", ""))

        second_pass_added = 0
        second_pass_error = ""
        if leftover_source and leftover_target and args.enable_second_pass:
            second_payload = build_second_pass_payload(
                oare_id=doc_id,
                leftover_source_text=leftover_source,
                leftover_target_text=leftover_target,
                existing_pairs=[{"source_sentence": src, "target_sentence": tgt} for src, tgt in existing_pair_keys],
            )
            parsed_second, raw_content_second, second_pass_error = call_alignment_llm(
                client=client,
                model=model,
                prompt_payload=second_payload,
                max_completion_tokens=args.max_completion_tokens,
                retries=args.retries,
                sleep_seconds=args.retry_sleep_seconds,
            )

            raw_handle.write(
                json.dumps(
                    {
                        "run_id": run_id,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "oare_id": doc_id,
                        "raw_response": raw_content_second,
                        "parse_ok": parsed_second is not None,
                        "error": second_pass_error,
                        "pass": "second_pass",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            raw_handle.flush()

            if parsed_second is not None:
                pairs_second = parsed_second.get("pairs", []) if isinstance(parsed_second, dict) else []
                accepted_second, issues_second, _ = validate_aligned_pairs(
                    pairs=pairs_second,
                    doc_transliteration=doc_transliteration,
                    doc_translation=doc_translation,
                    existing_pair_keys=existing_pair_keys,
                )

                for issue in issues_second:
                    validation_issue_rows.append(
                        {
                            "run_id": run_id,
                            "oare_id": doc_id,
                            "pass": "second_pass",
                            **issue,
                        }
                    )

                for accepted in accepted_second:
                    src_raw = accepted["source_sentence_raw"]
                    tgt_raw = accepted["target_sentence_raw"]
                    result_rows.append(
                        {
                            "oare_id": doc_id,
                            "pair_index": int(accepted["pair_index"]),
                            "alignment_pass": "second_pass",
                            "source_sentence_raw": src_raw,
                            "target_sentence_raw": tgt_raw,
                            "source_sentence_clean": normalize_transliteration(src_raw),
                            "target_sentence_clean": normalize_translation(tgt_raw),
                            "source_method": accepted["source_method"] or "leftover_inferred",
                            "target_method": accepted["target_method"] or "leftover_inferred",
                            "alignment_confidence": accepted["alignment_confidence"],
                            "alignment_note": accepted["alignment_note"],
                            "source_pos": accepted["source_pos"],
                            "target_pos": accepted["target_pos"],
                            "doc_translation": doc_translation,
                            "doc_transliteration": doc_transliteration,
                            "run_id": run_id,
                        }
                    )
                    second_pass_added += 1
                    produced += 1

        status_rows.append(
            {
                "run_id": run_id,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "oare_id": doc_id,
                "status": "completed",
                "reason": "ok",
                "expected_clue_rows": len(clue_rows),
                "produced_pairs": produced,
                "accepted_first_pass_pairs": pair_stats["accepted_pairs"],
                "rejected_not_substring": pair_stats["rejected_not_substring"],
                "rejected_duplicate": pair_stats["rejected_duplicate"],
                "rejected_order": pair_stats["rejected_order"],
                "rejected_overlap": pair_stats["rejected_overlap"],
                "second_pass_added_pairs": second_pass_added,
                "second_pass_error": second_pass_error,
                "leftover_source_text": leftover_source,
                "leftover_target_text": leftover_target,
            }
        )

        leftover_debug_rows.append(
            {
                "run_id": run_id,
                "oare_id": doc_id,
                "leftover_source_text": leftover_source,
                "leftover_target_text": leftover_target,
                "leftover_source_chars": len(leftover_source),
                "leftover_target_chars": len(leftover_target),
                "first_pass_accepted_pairs": pair_stats["accepted_pairs"],
                "second_pass_added_pairs": second_pass_added,
            }
        )

        log(f"[{index}/{total_docs}] DONE {doc_id} | produced_pairs={produced}")

        if index % max(1, args.checkpoint_every) == 0:
            log(f"Checkpoint save at {index}/{total_docs} docs...")
            result_df_ckpt = pd.DataFrame(result_rows)
            if not result_df_ckpt.empty:
                save_csv(result_df_ckpt, dataset_path)

            new_status_df_ckpt = pd.DataFrame(status_rows)
            if not existing_status_df.empty:
                merged = pd.concat([existing_status_df, new_status_df_ckpt], ignore_index=True)
            else:
                merged = new_status_df_ckpt
            save_csv(merged, doc_status_path)

            save_csv(pd.DataFrame(leftover_debug_rows), leftover_debug_path)
            save_csv(pd.DataFrame(validation_issue_rows), validation_issues_path)

    raw_handle.close()

    result_df = pd.DataFrame(result_rows)
    if not result_df.empty:
        result_df["source_token_count"] = result_df["source_sentence_clean"].map(lambda x: len(str(x).split()))
        result_df["target_token_count"] = result_df["target_sentence_clean"].map(lambda x: len(str(x).split()))
        result_df = result_df[(result_df["source_token_count"] > 0) & (result_df["target_token_count"] > 0)].copy()
        result_df = result_df.reset_index(drop=True)
    save_csv(result_df, dataset_path)

    new_status_df = pd.DataFrame(status_rows)
    if not existing_status_df.empty:
        final_status_df = pd.concat([existing_status_df, new_status_df], ignore_index=True)
    else:
        final_status_df = new_status_df
    save_csv(final_status_df, doc_status_path)
    save_csv(pd.DataFrame(leftover_debug_rows), leftover_debug_path)
    save_csv(pd.DataFrame(validation_issue_rows), validation_issues_path)

    completed_count = int((final_status_df["status"] == "completed").sum()) if not final_status_df.empty else 0
    error_count = int((final_status_df["status"] == "llm_error").sum()) if not final_status_df.empty else 0

    log("Run completed.")
    log(f"Output dataset: {dataset_path} | rows={len(result_df)}")
    log(f"Document status: {doc_status_path} | completed={completed_count} | llm_error={error_count}")
    log(f"Unmatched docs: {unmatched_path} | rows={len(unmatched_df)}")
    log(f"Leftover debug: {leftover_debug_path}")
    log(f"Validation issues: {validation_issues_path}")
    log(f"Raw responses: {raw_jsonl_path}")

    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM alignment pipeline for train sentence-level dataset generation")
    parser.add_argument("--input-dir", type=str, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--max-docs", type=int, default=0, help="0 means all matched docs")
    parser.add_argument("--max-doc-chars", type=int, default=12000)
    parser.add_argument("--max-completion-tokens", type=int, default=3000)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep-seconds", type=float, default=2.0)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--enable-second-pass", action="store_true", default=True)
    parser.add_argument("--disable-second-pass", dest="enable_second_pass", action="store_false")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--shuffle", action="store_true", default=False)
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    parsed_args = parser.parse_args()
    sys.exit(run(parsed_args))
