"""Import terms from class_mapping and cases/images/top-3 AI outputs from R1 CSV.

Sources (relative to backend/):
  - docs/class_mapping.txt (CSV: class_id,disease_condition)
  - docs/R1_Data_with_disease.csv (CSV with image_path, gt, gt_disease, prob_*)

Behavior:
  - Ensures roles: GP, Nurse, Other (removes legacy 'Dermatology Specialist' if present)
  - Creates DiagnosisTerm rows using class_id as id and disease_condition as canonical name
  - Creates Case rows (idempotent by image_path), Image rows, and AIOutput top-3 per case
  - Stores full probability vector in Case.ai_predictions_json (term_id -> score)

Idempotency:
  - Terms: skip if id already exists
  - Cases: skip creating Case/Image if an Image with same image_url exists
  - AIOutput: unique by (case_id, rank); skip if that rank exists for the case

Usage:
  python scripts/import_r1_data.py
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.models import models as m  # ensure models imported


BASE_DIR = Path(__file__).resolve().parent.parent  # backend/
DOCS_DIR = BASE_DIR / "docs"
CLASS_MAPPING_PATH = DOCS_DIR / "class_mapping.txt"
R1_CSV_PATH = DOCS_DIR / "R1_Data_with_disease.csv"


def log(msg: str) -> None:
    print(f"[import_r1] {msg}")


def ensure_roles(db: Session, role_names: List[str]) -> int:
    roles = db.execute(select(m.Role)).scalars().all()
    existing_lc = {r.name.lower(): r for r in roles}
    # remove deprecated
    legacy = existing_lc.get("dermatology specialist")
    if legacy:
        db.delete(legacy)
        log("Removed legacy role 'Dermatology Specialist'")
    created = 0
    for name in role_names:
        if name.lower() not in existing_lc:
            db.add(m.Role(name=name))
            created += 1
    return created


def load_class_mapping(path: Path) -> Dict[int, str]:
    """Return mapping class_id -> disease_condition (canonical name)."""
    if not path.exists():
        raise SystemExit(f"Missing class mapping file: {path}")
    mapping: Dict[int, str] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    reader = csv.DictReader(lines)
    for row in reader:
        try:
            cid = int(row["class_id"])  # type: ignore[arg-type]
        except Exception as e:  # noqa: BLE001
            raise SystemExit(f"Invalid class_id in {path}: {e}")
        name = (row.get("disease_condition") or "").strip()
        if not name:
            raise SystemExit(f"Empty disease_condition for class_id={cid}")
        mapping[cid] = name
    if not mapping:
        raise SystemExit("No rows found in class_mapping.txt")
    return mapping


def ensure_terms(db: Session, mapping: Dict[int, str]) -> int:
    existing_ids = {t.id for t in db.execute(select(m.DiagnosisTerm)).scalars()}
    created = 0
    for cid, name in mapping.items():
        if cid in existing_ids:
            continue
        db.add(m.DiagnosisTerm(id=cid, name=name))
        created += 1
    return created


def header_prob_to_name(col: str) -> str | None:
    """Map 'prob_xxx_yyy' -> 'xxx yyy'. Returns None if not a prob column."""
    if not col.startswith("prob_"):
        return None
    core = col[len("prob_"):]
    return core.replace("_", " ")


def build_prob_header_index(reader: csv.DictReader, name_to_id: Dict[str, int]) -> List[Tuple[int, str]]:
    """Return list of (term_id, header_name) for prob columns present and mappable."""
    terms: List[Tuple[int, str]] = []
    for col in reader.fieldnames or []:
        dn = header_prob_to_name(col)
        if dn is None:
            continue
        # disease names in mapping are lowercased for matching
        tid = name_to_id.get(dn.lower())
        if tid is not None:
            terms.append((tid, col))
    if not terms:
        raise SystemExit("No probability columns matched class_mapping names")
    return terms


def parse_float_safe(v: str) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


def ensure_case_and_image(db: Session, image_path: str, gt_id: int, probs: Dict[int, float], next_case_id: int) -> Tuple[int, bool]:
    """Create Case and Image if not existing by image_path. Return (case_id, created?)."""
    existing_img = db.execute(select(m.Image).where(m.Image.image_url == image_path)).scalar_one_or_none()
    if existing_img:
        case_id = existing_img.case_id
        return case_id, False
    # create new case with deterministic id from sequence given
    case = m.Case(id=next_case_id, ground_truth_diagnosis_id=gt_id, ai_predictions_json=probs)
    db.add(case)
    db.flush()  # ensure id
    img = m.Image(case_id=case.id, image_url=image_path)
    db.add(img)
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


def import_r1(db: Session) -> None:
    # 1) Roles
    roles_added = ensure_roles(db, ["GP", "Nurse", "Other"])

    # 2) Terms
    class_map = load_class_mapping(CLASS_MAPPING_PATH)
    terms_added = ensure_terms(db, class_map)

    # name->id map for prob header matching
    name_to_id = {v.lower(): k for k, v in class_map.items()}

    # 3) Cases from R1 CSV
    if not R1_CSV_PATH.exists():
        raise SystemExit(f"Missing R1 CSV file: {R1_CSV_PATH}")
    lines = R1_CSV_PATH.read_text(encoding="utf-8").splitlines()
    reader = csv.DictReader(lines)
    # Map probability columns to term IDs
    prob_cols = build_prob_header_index(reader, name_to_id)

    cases_new = images_new = ai_rows = 0
    next_case_id = 1
    # If there are existing cases, continue IDs after max
    existing_case_ids = [c.id for c in db.execute(select(m.Case)).scalars()]
    if existing_case_ids:
        next_case_id = max(existing_case_ids) + 1

    for idx, row in enumerate(reader, start=1):
        image_path = (row.get("image_path") or "").strip()
        if not image_path:
            log(f"Row {idx}: missing image_path, skipping")
            continue
        try:
            gt_id = int(row.get("gt") or "")
        except Exception:
            log(f"Row {idx}: invalid gt value '{row.get('gt')}', skipping")
            continue

        # Build full prob vector term_id -> score
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
    log("Import complete:")
    log(f"  Roles inserted: {roles_added}")
    log(f"  Terms inserted: {terms_added}")
    log(f"  Cases inserted: {cases_new}")
    log(f"  Images inserted: {images_new}")
    log(f"  AIOutput rows inserted: {ai_rows}")


def main() -> None:
    db: Session = SessionLocal()
    try:
        import_r1(db)
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
