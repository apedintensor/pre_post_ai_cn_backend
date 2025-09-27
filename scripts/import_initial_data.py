"""Unified data import script.

Loads:
    * Roles (GP, Nurse, Other) – removes legacy 'Dermatology Specialist' if present
  * Diagnosis terms + synonyms from derm_dictionary.csv
  * Cases, Images, full probability vector, and Top-3 AI outputs from ai_prediction_by_id.csv

Idempotent: safe to re-run; existing rows are skipped.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

from sqlalchemy.orm import Session
from sqlalchemy import select

from app.db.session import SessionLocal
from app.models import models as m


# ---------------- Utility helpers -----------------

def ensure_roles(db: Session, role_names: List[str]) -> None:
    roles = db.execute(select(m.Role)).scalars().all()
    existing_lc = {r.name.lower(): r for r in roles}
    legacy = existing_lc.get("dermatology specialist")
    if legacy:
        db.delete(legacy)
        print("Removed legacy role 'Dermatology Specialist'")
    created = 0
    for name in role_names:
        if name.lower() not in existing_lc:
            db.add(m.Role(name=name))
            created += 1
    if created:
        print(f"Inserted {created} new roles")
    else:
        print("Roles already present (after legacy cleanup)")


def load_terms_and_synonyms(db: Session, csv_path: Path, dry_run: bool = False) -> Tuple[int, int]:
    """Return (#terms_inserted, #synonyms_inserted)."""
    reader = csv.DictReader(csv_path.read_text(encoding="utf-8").splitlines())
    required_cols = {"id", "canonical", "type", "alias"}
    if set(reader.fieldnames or []) != required_cols:
        raise SystemExit(f"Unexpected derm_dictionary header: {reader.fieldnames}")

    # Cache existing
    existing_terms = {t.id: t for t in db.execute(select(m.DiagnosisTerm)).scalars().all()}
    existing_synonyms_lc = {s.synonym.lower() for s in db.execute(select(m.DiagnosisSynonym)).scalars().all()}

    new_terms = 0
    new_syns = 0

    for row in reader:
        try:
            term_id = int(row["id"]) if row["id"] != "" else None
        except ValueError:
            print(f"Skipping row with invalid id: {row}")
            continue
        canonical = (row["canonical"] or "").strip()
        rtype = (row["type"] or "").strip().lower()
        alias = (row["alias"] or "").strip()

        if canonical and term_id is not None and term_id not in existing_terms:
            if not dry_run:
                term = m.DiagnosisTerm(id=term_id, name=canonical)
                db.add(term)
            existing_terms[term_id] = True  # sentinel
            new_terms += 1

        if rtype in {"synonym", "abbreviation"} and alias:
            if alias.lower() not in existing_synonyms_lc:
                if not dry_run:
                    db.add(m.DiagnosisSynonym(diagnosis_term_id=term_id, synonym=alias))
                existing_synonyms_lc.add(alias.lower())
                new_syns += 1

    print(f"Terms inserted: {new_terms}; Synonyms inserted: {new_syns}")
    return new_terms, new_syns


def parse_case_row(header_ids: List[int], row: Dict[str, str]) -> Tuple[int, str, int, Dict[int, float]]:
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
            # Skip malformed token
            continue
    return case_id, image_path, gt, probs


def ensure_case(db: Session, case_id: int, gt: int | None, probs: Dict[int, float], image_path: str, dry_run: bool) -> bool:
    exists = db.get(m.Case, case_id)
    if exists:
        return False
    if not dry_run:
        case = m.Case(id=case_id, ground_truth_diagnosis_id=gt, ai_predictions_json=probs)
        db.add(case)
        # image (one per case)
        db.add(m.Image(case_id=case_id, image_url=image_path))
    return True


def ensure_ai_outputs(db: Session, case_id: int, probs: Dict[int, float], dry_run: bool) -> int:
    # Determine top-3
    top3 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
    inserted = 0
    for rank, (term_id, score) in enumerate(top3, start=1):
        # Check uniqueness by (case_id, rank)
        exists = db.execute(
            select(m.AIOutput).where(m.AIOutput.case_id == case_id, m.AIOutput.rank == rank)
        ).scalar_one_or_none()
        if exists:
            continue
        if not dry_run:
            db.add(m.AIOutput(case_id=case_id, rank=rank, prediction_id=term_id, confidence_score=score))
        inserted += 1
    return inserted


def load_cases(db: Session, csv_path: Path, batch_size: int, max_cases: int | None, dry_run: bool = False) -> Tuple[int, int, int]:
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    reader = csv.DictReader(lines)
    # Extract numeric term id headers (exclude case_id,image_path,gt)
    header_ids: List[int] = []
    for col in reader.fieldnames[3:]:  # first three fixed
        try:
            header_ids.append(int(col))
        except (TypeError, ValueError):
            raise SystemExit(f"Non-numeric diagnosis term header found: {col}")

    cases_new = images_new = ai_rows = 0
    for idx, row in enumerate(reader, start=1):
        if max_cases and idx > max_cases:
            break
        try:
            case_id, image_path, gt, probs = parse_case_row(header_ids, row)
        except Exception as e:  # noqa
            print(f"Skipping malformed case row {idx}: {e}")
            continue
        created = ensure_case(db, case_id, gt, probs, image_path, dry_run)
        if created:
            cases_new += 1
            images_new += 1  # exactly one per new case
            ai_rows += ensure_ai_outputs(db, case_id, probs, dry_run)
        if not dry_run and cases_new and cases_new % batch_size == 0:
            db.commit()
            print(f"Committed {cases_new} cases so far ...")

    if not dry_run:
        db.commit()
    print(f"Cases inserted: {cases_new}; Images inserted: {images_new}; AI output rows inserted: {ai_rows}")
    return cases_new, images_new, ai_rows


# ---------------- CLI -----------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Import initial reader study data")
    p.add_argument("--terms", type=Path, required=True, help="Path to derm_dictionary.csv")
    p.add_argument("--cases", type=Path, required=True, help="Path to ai_prediction_by_id.csv")
    p.add_argument("--commit-batch-size", type=int, default=200)
    p.add_argument("--max-cases", type=int, default=None, help="Limit number of cases (debug)")
    p.add_argument("--dry-run", action="store_true", help="Parse & report without writing")
    return p


def main():
    args = build_arg_parser().parse_args()

    if not args.terms.exists():
        raise SystemExit(f"Terms file not found: {args.terms}")
    if not args.cases.exists():
        raise SystemExit(f"Cases file not found: {args.cases}")

    db: Session = SessionLocal()
    try:
        ensure_roles(db, ["GP", "Nurse", "Other"])
        load_terms_and_synonyms(db, args.terms, dry_run=args.dry_run)
        load_cases(
            db,
            args.cases,
            batch_size=args.commit_batch_size,
            max_cases=args.max_cases,
            dry_run=args.dry_run,
        )
        if args.dry_run:
            print("Dry run complete – no changes committed.")
        else:
            db.commit()
            print("Import finished successfully.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
