# Stage-2 LLM Extraction Template

Use this template for each row from:

- `data/processed/publications_llm_alignment_shortlist.csv`

## Input fields

- `oare_id`
- `transliteration`
- `pdf_name`
- `page`
- `page_text`
- `page_text_snippet`
- `match_source`
- `match_confidence`

## Prompt Template

You are aligning OCR publication text with an Old Assyrian tablet transliteration.

Given:
- tablet id: {oare_id}
- transliteration: {transliteration}
- publication: {pdf_name}
- page: {page}
- OCR page text: {page_text}
- optional snippet: {page_text_snippet}

Tasks:
1. Find the most likely translation segment in the OCR text that corresponds to this tablet.
2. If the translation is not English, translate it to English.
3. Return a clean single translation text (preserve meaning, no commentary).
4. Return confidence (0.0-1.0).
5. Return a short evidence note (why this segment was selected).
6. If no reliable translation is present, return `found=false`.

Return strict JSON only in this schema:

```json
{
  "found": true,
  "oare_id": "...",
  "source_pdf": "...",
  "source_page": "...",
  "translation_extracted": "...",
  "translation_english": "...",
  "language_detected": "en|de|fr|tr|other",
  "confidence": 0.0,
  "evidence": "...",
  "notes": "..."
}
```

## Acceptance filter (recommended)

Keep as silver pair only if:

- `found == true`
- `confidence >= 0.80`
- `translation_english` length >= 15 chars

## Final silver pair columns

- `oare_id`
- `source_transliteration`
- `target_translation_english`
- `source_pdf`
- `source_page`
- `stage1_match_source`
- `stage1_match_confidence`
- `stage2_confidence`
- `language_detected`
- `evidence`
