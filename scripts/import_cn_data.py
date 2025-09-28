"""Import CN terminology (Chinese) and SNU84 sample cases + AI predictions.

Sources (relative to backend/):
    - data/term_synonom-cn.json  (list of { id, canonical_en, canonical_zh, synonyms_zh[], abbreviations_zh[] })
    - data/SNU84_sample_10_stratified_cleaned.csv (CSV with image_path, gt, gt_disease, prob_class_*)

Behavior:
    - Upserts DiagnosisTerm rows using CN canonical name (canonical_zh) as the display name
    - Adds Chinese synonyms/abbreviations into diagnosis_synonyms (unique by text)
    - Deletes old case-related data (cases/images/AI outputs/assignments/assessments/entries/block feedback/metadata)
    - Imports cases/images and top-3 AIOutput per case from SNU84 CSV
    - Stores full probability vector into Case.ai_predictions_json (term_id -> score)

Idempotency:
    - Terms: update name if term id exists; create if missing
    - Synonyms: skip if synonym already exists
    - Import always starts from a clean slate for case-related data

Usage:
    python scripts/import_cn_data.py

Notes:
    - Matches probability headers to canonical_en from the CN JSON file.
    - Accepts both "prob_" and "prob_class_" prefixes in probability columns.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple

from sqlalchemy import select, delete
from sqlalchemy.orm import Session

BASE_DIR = Path(__file__).resolve().parent.parent  # backend/
# Ensure project root is importable when running as a file: `python scripts/import_cn_data.py`
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.db.session import SessionLocal
from app.models import models as m  # ensure models imported
DATA_DIR = BASE_DIR / "data"
CN_TERMS_PATH = DATA_DIR / "term_synonom-cn.json"
SNU84_CSV_PATH = DATA_DIR / "SNU84_sample_10_stratified_cleaned.csv"


def log(msg: str) -> None:
    print(f"[import_cn] {msg}")


def ensure_roles(db: Session, role_names: List[str]) -> int:
    roles = db.execute(select(m.Role)).scalars().all()
    existing_lc = {r.name.lower(): r for r in roles}
    created = 0
    for name in role_names:
        if name.lower() not in existing_lc:
            db.add(m.Role(name=name))
            created += 1
    return created


# ---------- Reset old data (cases and related) ----------

def reset_old_case_data(db: Session) -> Dict[str, int]:
    """Delete old case-related data in dependency-safe order.

    Keeps terminology (diagnosis_terms and diagnosis_synonyms) intact.
    Returns a dict of deleted row counts by table label.
    """
    results: Dict[str, int] = {}
    # Aggregates
    results["block_feedback"] = db.execute(delete(m.BlockFeedback)).rowcount or 0
    # Assessment tree
    results["diagnosis_entries"] = db.execute(delete(m.DiagnosisEntry)).rowcount or 0
    results["assessments"] = db.execute(delete(m.Assessment)).rowcount or 0
    results["assignments"] = db.execute(delete(m.ReaderCaseAssignment)).rowcount or 0
    # Case children
    results["ai_outputs"] = db.execute(delete(m.AIOutput)).rowcount or 0
    results["images"] = db.execute(delete(m.Image)).rowcount or 0
    results["case_metadata"] = db.execute(delete(m.CaseMetaData)).rowcount or 0
    # Cases
    results["cases"] = db.execute(delete(m.Case)).rowcount or 0
    db.commit()
    return results


# ---------- Chinese Terms & Synonyms ----------

def load_cn_terms(path: Path) -> List[dict]:
    if not path.exists():
        raise SystemExit(f"Missing CN terms JSON: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("JSON root must be a list")
        # Filter out empty/invalid entries gracefully
        cleaned: List[dict] = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            # must have id and canonical_zh at minimum
            if "id" not in item or "canonical_zh" not in item:
                continue
            cleaned.append(item)
        if not cleaned:
            raise ValueError("No valid entries found in CN terms JSON")
        return cleaned
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"Failed to parse {path}: {e}")


def ensure_cn_terms_and_synonyms(db: Session, entries: List[dict]) -> Tuple[int, int]:
    # Build existing synonyms set for quick lookups
    existing_synonyms = {s.synonym for s in db.execute(select(m.DiagnosisSynonym)).scalars().all()}
    terms_created = 0
    synonyms_created = 0

    for item in entries:
        try:
            tid = int(item.get("id"))
        except Exception:
            continue
        name_zh = (item.get("canonical_zh") or "").strip()
        name_en = (item.get("canonical_en") or "").strip()
        if not name_zh:
            continue

        term = db.get(m.DiagnosisTerm, tid)
        if term is None:
            db.add(m.DiagnosisTerm(id=tid, name=name_zh))
            terms_created += 1
        else:
            # Update display name to Chinese if different
            if (term.name or "").strip() != name_zh:
                term.name = name_zh

        # Add synonyms (Chinese) if any
        syns = []
        if isinstance(item.get("synonyms_zh"), list):
            syns.extend([str(s).strip() for s in item["synonyms_zh"] if str(s).strip()])
        if isinstance(item.get("abbreviations_zh"), list):
            syns.extend([str(s).strip() for s in item["abbreviations_zh"] if str(s).strip()])
        # Optionally include canonical_zh as a synonym too (usually redundant)
        for s in syns:
            if s in existing_synonyms:
                continue
            db.add(m.DiagnosisSynonym(diagnosis_term_id=tid, synonym=s))
            existing_synonyms.add(s)
            synonyms_created += 1

    return terms_created, synonyms_created


# ---------- R1 Cases & AI Outputs ----------

def header_prob_to_name(col: str) -> str | None:
    """Map probability column header to canonical English name.

    Supports prefixes like:
      - prob_<name>
      - prob_class_<name>
    """
    if col.startswith("prob_class_"):
        core = col[len("prob_class_"):]
    elif col.startswith("prob_"):
        core = col[len("prob_"):]
    else:
        return None
    return core.replace("_", " ")


def build_en_to_id(entries: List[dict]) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for item in entries:
        try:
            tid = int(item.get("id"))
        except Exception:
            continue
        en = (item.get("canonical_en") or "").strip().lower()
        if not en:
            continue
        mapping[en] = tid
    return mapping


def build_prob_header_index(reader: csv.DictReader, en_to_id: Dict[str, int]) -> List[Tuple[int, str]]:
    terms: List[Tuple[int, str]] = []
    for col in reader.fieldnames or []:
        dn = header_prob_to_name(col)
        if dn is None:
            continue
        tid = en_to_id.get(dn.lower())
        if tid is not None:
            terms.append((tid, col))
    if not terms:
        raise SystemExit("No probability columns matched canonical_en names from CN JSON")
    return terms


def parse_float_safe(v: str) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


def ensure_case_and_image(db: Session, image_path: str, gt_id: int | None, probs: Dict[int, float], next_case_id: int) -> Tuple[int, bool]:
    existing_img = db.execute(select(m.Image).where(m.Image.image_url == image_path)).scalar_one_or_none()
    if existing_img:
        return existing_img.case_id, False
    case = m.Case(id=next_case_id, ground_truth_diagnosis_id=gt_id, ai_predictions_json=probs)
    db.add(case)
    db.flush()
    db.add(m.Image(case_id=case.id, image_url=image_path))
    return case.id, True


def ensure_top3_ai_outputs(db: Session, case_id: int, probs: Dict[int, float]) -> int:
    top3 = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:3]
    existing_ranks = {o.rank for o in db.execute(select(m.AIOutput).where(m.AIOutput.case_id == case_id)).scalars()}
    inserted = 0
    for rank, (term_id, score) in enumerate(top3, start=1):
        if rank in existing_ranks:
            continue
        db.add(m.AIOutput(case_id=case_id, rank=rank, prediction_id=term_id, confidence_score=score))
        inserted += 1
    return inserted


def import_cn(db: Session) -> None:
    # 0) Reset old case-related data
    deleted = reset_old_case_data(db)

    # 1) Roles
    # Seed roles with Chinese labels for CN site
    roles_added = ensure_roles(db, ["医生", "护士", "其他"])  # harmless if already present

    # 2) CN terms & synonyms
    cn_entries = load_cn_terms(CN_TERMS_PATH)
    terms_added, syns_added = ensure_cn_terms_and_synonyms(db, cn_entries)

    # Build en->id map for reading R1 CSV probability columns
    en_to_id = build_en_to_id(cn_entries)

    # 3) Cases from SNU84 CSV
    if not SNU84_CSV_PATH.exists():
        raise SystemExit(f"Missing SNU84 CSV file: {SNU84_CSV_PATH}")
    lines = SNU84_CSV_PATH.read_text(encoding="utf-8").splitlines()
    reader = csv.DictReader(lines)

    prob_cols = build_prob_header_index(reader, en_to_id)
    log(f"Matched probability columns: {len(prob_cols)}")

    cases_new = images_new = ai_rows = 0
    next_case_id = 1
    existing_case_ids = [c.id for c in db.execute(select(m.Case)).scalars()]
    if existing_case_ids:
        next_case_id = max(existing_case_ids) + 1

    for idx, row in enumerate(reader, start=1):
        image_path = (row.get("image_path") or "").strip()
        if not image_path:
            log(f"Row {idx}: missing image_path, skipping")
            continue

        gt_id: int | None = None
        gt_raw = (row.get("gt") or "").strip()
        if gt_raw != "":
            try:
                gt_id = int(gt_raw)
            except Exception:
                gt_id = None
        if gt_id is None:
            # fallback: map gt_disease (english) via en_to_id
            gt_name_en = (row.get("gt_disease") or "").strip().lower()
            if gt_name_en:
                gt_id = en_to_id.get(gt_name_en)

        probs: Dict[int, float] = {}
        for term_id, col in prob_cols:
            v = parse_float_safe(row.get(col, "") or "0")
            probs[term_id] = v

        case_id, created = ensure_case_and_image(db, image_path, gt_id, probs, next_case_id)
        if created:
            cases_new += 1
            images_new += 1
            next_case_id += 1
        ai_rows += ensure_top3_ai_outputs(db, case_id, probs)

    db.commit()
    log("Import CN complete:")
    log("  Old data deleted:")
    for k, v in deleted.items():
        log(f"    {k}: {v}")
    log(f"  Roles inserted: {roles_added}")
    log(f"  Terms created (or updated): {terms_added}")
    log(f"  Synonyms inserted: {syns_added}")
    log(f"  Cases inserted: {cases_new}")
    log(f"  Images inserted: {images_new}")
    log(f"  AIOutput rows inserted: {ai_rows}")


def main() -> None:
    db: Session = SessionLocal()
    try:
        import_cn(db)
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        db.rollback()
        log(f"Error: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":  # pragma: no cover
    main()
