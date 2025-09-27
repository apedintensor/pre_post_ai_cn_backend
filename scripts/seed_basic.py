"""Simplified production seeding script (no CLI args).

Intended for Fly.io one-shot initialization AFTER the database schema
already exists (e.g. via migrations or metadata.create_all elsewhere).

Data sources (hard‑coded relative paths):
  * data/derm_dictionary.json         (diagnosis vocabulary)
  * data/ai_prediction_by_id.csv      (cases + full probability vectors)

What it loads (idempotent):
    1. Roles: GP, Nurse, Other. (Removes legacy 'Dermatology Specialist')
  2. DiagnosisTerm rows from each JSON entry (id -> canonical name).
  3. DiagnosisSynonym rows for every synonym AND abbreviation.
  4. Case rows (id, ground_truth_diagnosis_id, ai_predictions_json).
  5. Image (one per case) referencing image_path column in CSV.
  6. AIOutput top‑3 predictions per case (ranked 1..3).

Safe to re-run: existing rows are skipped (unique constraints / lookups).

Usage (locally):
  python scripts/seed_basic.py

Usage (Fly after deploy):
  fly ssh console -a <app-name> -C "python scripts/seed_basic.py"

Exit codes:
  0 success (even if nothing new inserted)
  1 unrecoverable error (missing files / parse issues)
"""
from __future__ import annotations

import json
import csv
import math
import sys
from pathlib import Path
from typing import Dict, List

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.models import models as m  # ensure models imported

BASE_DIR = Path(__file__).resolve().parent.parent  # backend/ (scripts/ is sibling of app/)
DATA_DIR = BASE_DIR / "data"
TERMS_JSON_PATH = DATA_DIR / "derm_dictionary.json"
CASES_CSV_PATH = DATA_DIR / "ai_prediction_by_id.csv"


# ---------------- Helpers -----------------

def log(msg: str):  # small consistent logger
    print(f"[seed_basic] {msg}")


def ensure_roles(db: Session, role_names: List[str]) -> int:
    """Ensure desired roles exist. If legacy 'Dermatology Specialist' present, remove it.

    Returns number of newly inserted roles (excludes deletions).
    """
    roles = db.execute(select(m.Role)).scalars().all()
    existing_lc = {r.name.lower(): r for r in roles}
    # Remove deprecated role if exists
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


def load_terms_from_json(db: Session, path: Path) -> tuple[int, int]:
    """Return (terms_inserted, synonyms_inserted).

    JSON structure expected: list[ {"id": int, "canonical": str, "synonyms": [..], "abbreviations": [..]} ]
    Abbreviations treated as synonyms.
    """
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"Failed to parse JSON {path}: {e}")

    existing_terms = {t.id for t in db.execute(select(m.DiagnosisTerm)).scalars().all()}
    existing_synonyms_lc = {s.synonym.lower() for s in db.execute(select(m.DiagnosisSynonym)).scalars().all()}

    terms_new = 0
    syns_new = 0
    for obj in entries:
        try:
            term_id = int(obj["id"])
            canonical = (obj.get("canonical") or "").strip()
        except Exception:
            continue
        if canonical and term_id not in existing_terms:
            db.add(m.DiagnosisTerm(id=term_id, name=canonical))
            existing_terms.add(term_id)
            terms_new += 1
        # synonyms + abbreviations
        for key in ("synonyms", "abbreviations"):
            for syn in obj.get(key) or []:
                syn = syn.strip()
                if not syn:
                    continue
                lower = syn.lower()
                if lower in existing_synonyms_lc:
                    continue
                db.add(m.DiagnosisSynonym(diagnosis_term_id=term_id, synonym=syn))
                existing_synonyms_lc.add(lower)
                syns_new += 1
    return terms_new, syns_new


def parse_case_row(header_ids: List[int], row: dict) -> tuple[int, str, int | None, Dict[int, float]]:
    case_id = int(row["case_id"])  # may raise
    image_path = row["image_path"].strip()
    gt = int(row["gt"]) if row["gt"] != "" else None
    probs: Dict[int, float] = {}
    for term_id in header_ids:
        raw = row.get(str(term_id), "")
        if raw == "":
            continue
        try:
            val = float(raw)
            if math.isfinite(val):
                probs[term_id] = val
        except ValueError:
            continue
    return case_id, image_path, gt, probs


def ensure_case_and_image(db: Session, case_id: int, gt: int | None, probs: Dict[int, float], image_path: str) -> bool:
    if db.get(m.Case, case_id):
        return False
    db.add(m.Case(id=case_id, ground_truth_diagnosis_id=gt, ai_predictions_json=probs))
    db.add(m.Image(case_id=case_id, image_url=image_path))
    return True


def ensure_ai_outputs(db: Session, case_id: int, probs: Dict[int, float]) -> int:
    # top 3 by probability
    top3 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
    inserted = 0
    existing_ranks = {o.rank for o in db.execute(select(m.AIOutput).where(m.AIOutput.case_id == case_id)).scalars()}
    for rank, (term_id, score) in enumerate(top3, start=1):
        if rank in existing_ranks:
            continue
        db.add(m.AIOutput(case_id=case_id, rank=rank, prediction_id=term_id, confidence_score=score))
        inserted += 1
    return inserted


def load_cases(db: Session, csv_path: Path) -> tuple[int, int, int]:
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    reader = csv.DictReader(lines)
    if not reader.fieldnames or reader.fieldnames[:3] != ["case_id", "image_path", "gt"]:
        raise SystemExit("Unexpected ai_prediction_by_id.csv header")
    header_ids: List[int] = []
    for col in reader.fieldnames[3:]:
        try:
            header_ids.append(int(col))
        except Exception:
            raise SystemExit(f"Non-numeric diagnosis term header found: {col}")

    cases_new = images_new = ai_rows = 0
    for idx, row in enumerate(reader, start=1):
        try:
            case_id, image_path, gt, probs = parse_case_row(header_ids, row)
        except Exception as e:  # noqa: BLE001
            log(f"Skipping malformed row {idx}: {e}")
            continue
        created = ensure_case_and_image(db, case_id, gt, probs, image_path)
        if created:
            cases_new += 1
            images_new += 1
            ai_rows += ensure_ai_outputs(db, case_id, probs)
    return cases_new, images_new, ai_rows


def main():
    # Validate files exist
    if not TERMS_JSON_PATH.exists():
        log(f"Missing terms JSON file: {TERMS_JSON_PATH}")
        sys.exit(1)
    if not CASES_CSV_PATH.exists():
        log(f"Missing cases CSV file: {CASES_CSV_PATH}")
        sys.exit(1)

    db: Session = SessionLocal()
    try:
        # 1. Roles
        roles_created = ensure_roles(db, ["GP", "Nurse", "Other"])
        # 2 & 3. Terms + synonyms
        terms_new, syns_new = load_terms_from_json(db, TERMS_JSON_PATH)
        # 4-6. Cases, images, AI outputs
        cases_new, images_new, ai_rows = load_cases(db, CASES_CSV_PATH)
        db.commit()
        log("Seeding complete:")
        log(f"  Roles inserted: {roles_created}")
        log(f"  Terms inserted: {terms_new}")
        log(f"  Synonyms inserted: {syns_new}")
        log(f"  Cases inserted: {cases_new}")
        log(f"  Images inserted: {images_new}")
        log(f"  AIOutput rows inserted: {ai_rows}")
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        db.rollback()
        log(f"Error during seeding: {e}")
        sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":  # pragma: no cover
    main()
