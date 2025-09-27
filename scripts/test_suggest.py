"""Seed derm dictionary terms & synonyms then exercise /diagnosis_terms/suggest endpoint.

Usage:
  uv run python scripts/test_suggest.py
  # or if virtualenv active:
  python scripts/test_suggest.py

Outputs sample queries and their top suggestions.
"""
from __future__ import annotations
import json
import asyncio
from pathlib import Path
from typing import Sequence

from httpx import AsyncClient, ASGITransport
from sqlalchemy.exc import IntegrityError

from app.main import app  # imports settings & routes
from app.db.session import SessionLocal
from app.db.base import Base
from app.models import models
from app.crud import crud_diagnosis_term

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "derm_dictionary.json"

# Sample query fragments to probe name, synonym, abbreviation coverage
QUERY_FRAGMENTS: list[str] = [
    "epi",            # epidermal cyst
    "cyst",           # substring
    "alo",            # alopecia areata
    "aa",             # abbreviation (AA)
    "bcc",            # abbreviation (BCC)
    "ak",             # abbreviation (AK)
    "squa",           # squamous cell carcinoma
    "herp",           # herpes simplex / herpes zoster
    "shing",          # synonym shingles
    "jock",           # tinea cruris synonym
    "tinea",          # many tinea terms
    "pity",           # pityriasis variants
    "ecz",            # eczema forms
    "psor",           # psoriasis
    "melan",          # melanoma / melanocytic nevus / melasma
    "kerat",          # actinic/seborrheic keratosis
    "wart",           # direct name
]


def seed_terms() -> None:
    db = SessionLocal()
    try:
        # Ensure tables exist (idempotent)
        Base.metadata.create_all(bind=db.get_bind())
        with DATA_PATH.open("r", encoding="utf-8") as f:
            records = json.load(f)
        added_terms = 0
        added_syns = 0
        for rec in records:
            name: str = rec["canonical"].strip()
            term = crud_diagnosis_term.get_diagnosis_term_by_name(db, name)
            if not term:
                term = models.DiagnosisTerm(name=name)
                db.add(term)
                db.commit()
                db.refresh(term)
                added_terms += 1
            # Combine synonyms + abbreviations as synonyms set
            syn_list: Sequence[str] = list({*(rec.get("synonyms") or []), *(rec.get("abbreviations") or [])})
            for syn in syn_list:
                syn = syn.strip()
                if not syn or syn.lower() == name.lower():
                    continue
                existing = (
                    db.query(models.DiagnosisSynonym)
                    .filter(models.DiagnosisSynonym.diagnosis_term_id == term.id,
                            models.DiagnosisSynonym.synonym == syn)
                    .first()
                )
                if existing:
                    continue
                db.add(models.DiagnosisSynonym(diagnosis_term_id=term.id, synonym=syn))
                try:
                    db.commit()
                    added_syns += 1
                except IntegrityError:
                    db.rollback()
        print(f"Seed complete: +{added_terms} terms, +{added_syns} synonyms/abbrev")
    finally:
        db.close()


async def exercise_suggest():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # No auth required currently for suggest; if added later, adjust here.
        for q in QUERY_FRAGMENTS:
            resp = await client.get("/diagnosis_terms/suggest", params={"q": q})
            if resp.status_code != 200:
                print(f"q='{q}': ERROR {resp.status_code} {resp.text}")
                continue
            data = resp.json()
            print(f"\nq='{q}' -> {len(data)} suggestions")
            for item in data[:5]:
                syns = ", ".join(item.get("synonyms") or [])
                print(f"  - id={item['id']} label='{item['label']}' match='{item['match']}' src={item['source']} score={item['score']:.1f} syns=[{syns}]")


def main():
    if not DATA_PATH.exists():
        raise SystemExit(f"Dictionary file not found: {DATA_PATH}")
    seed_terms()
    asyncio.run(exercise_suggest())


if __name__ == "__main__":
    main()
