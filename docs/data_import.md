# Data Import Guide

This guide explains how to load the initial reference data (roles, diagnosis terms + synonyms), cases, full AI probability vectors, and top‑3 AI outputs into the database.

## Overview
Data Sources:
1. `data/derm_dictionary.csv` – canonical diagnosis terms (id,name) with optional synonym / abbreviation rows.
2. `data/ai_prediction_by_id.csv` – per‑case model probability distribution. Columns:
   - `case_id` – numeric id (will be used directly as primary key for the case).
   - `image_path` – relative path to the image file.
   - `gt` – ground truth diagnosis term id (matches `derm_dictionary.csv` id).
   - Remaining columns – one numeric probability column per diagnosis term id (header value is the numeric term id, e.g. `130,54,87,...`).
3. `data/ai_prediction_header_map_ids.json` (optional helper) – maps verbose model probability names to numeric term ids (already aligned with the numeric columns in the CSV). Not required by the import script because the CSV header already exposes the numeric ids.

Target Tables:
| Table | Purpose |
|-------|---------|
| `roles` | Insert study roles (GP, Nurse, Other). |
| `diagnosis_terms` | Canonical dermatologic diagnoses with fixed ids (0..133 etc.). |
| `diagnosis_synonyms` | Synonyms / abbreviations derived from CSV rows where `type` is `synonym` or `abbreviation`. |
| `cases` | One row per case (id matches CSV). Stores `ground_truth_diagnosis_id` and full probability vector in `ai_predictions_json` (dict: term_id -> probability). |
| `images` | One image per case, using `image_path`. |
| `ai_outputs` | Top‑3 model predictions per case (rank 1..3) derived from the full probability vector. |

## Canonical Term & Synonym Import Logic
`derm_dictionary.csv` columns: `id,canonical,type,alias`

Rules:
* Rows with empty `type` are canonical definitions; ensure a `diagnosis_terms` row with that `id` and `canonical` as `name`.
* Rows where `type` in (`synonym`, `abbreviation`) create a `diagnosis_synonyms` row with `diagnosis_term_id = id` and `synonym = alias`.
* Duplicate synonyms (case‑insensitive) are skipped.
* Script is idempotent – existing terms or synonyms are left unchanged.

## Case + AI Data Import Logic
For each row of `ai_prediction_by_id.csv`:
1. Create `cases` row (skip if case already exists) with:
   * `id = case_id`
   * `ground_truth_diagnosis_id = gt`
   * `ai_predictions_json = { term_id(int): probability(float), ... }` built from the numeric probability columns.
2. Create matching `images` row with `image_url = image_path` (only if not already present).
3. Determine Top‑3 predictions: sort probability items descending; take top three distinct term ids.
4. Insert `ai_outputs` rows rank 1..3 (skip rank if already exists for the case) storing `prediction_id` and `confidence_score`.

Edge Cases & Safeguards:
* Missing canonical term id referenced in ground truth or probability columns → logged as warning; that row is skipped.
* If fewer than 3 distinct probabilities (extremely unlikely) only available ones are stored.
* Floating point scientific notation is parsed with `float()`.

## Roles
The script ensures (case‑insensitive) presence of:
* GP
* Dermatology Specialist
* Nurse

Existing different‑cased variants are reused; otherwise new rows inserted.

## Idempotency
You can run the script multiple times; it will:
* Skip existing roles (matching by lowercase name).
* Skip existing diagnosis terms (matching by id).
* Skip existing synonyms (matching by lowercase synonym text).
* Skip existing cases / images / ai outputs (matching by primary key or unique constraints).

## Running the Import
Activate your virtual environment, then execute:
```bash
python -m scripts.import_initial_data \
  --terms data/derm_dictionary.csv \
  --cases data/ai_prediction_by_id.csv
```

Optional flags:
* `--commit-batch-size N` (default 200) – commit every N cases to keep memory small.
* `--max-cases N` – limit number of case rows imported (debug / sampling).
* `--dry-run` – parse and report counts without writing.

## Verification Checklist After Import
| Check | Query |
|-------|-------|
| Roles inserted | `SELECT name FROM roles;` |
| Term count | `SELECT COUNT(*) FROM diagnosis_terms;` (expect ~134) |
| Synonym count | `SELECT COUNT(*) FROM diagnosis_synonyms;` |
| Case count | `SELECT COUNT(*) FROM cases;` |
| Image count | `SELECT COUNT(*) FROM images;` (should equal case count) |
| AI output rows | `SELECT COUNT(*) FROM ai_outputs;` (≈ 3 * case count) |

## Troubleshooting
| Symptom | Cause | Resolution |
|---------|-------|-----------|
| Missing term id warning | Term id in CSV not in dictionary | Add term to dictionary & re-run terms import. |
| UNIQUE constraint failed (roles.name) | Re-run with different casing | Safe to ignore – script catches after first attempt; subsequent runs skip. |
| JSON serialization error | Non-float probability value | Inspect offending row; ensure numeric tokens only. |

## Next Steps
* Add unit test that loads a small 2‑row slice to validate parser.
* Extend script with progress bar (`tqdm`) if desired (not added to avoid extra dependency).

---
Script implementation lives in `scripts/import_initial_data.py`.
