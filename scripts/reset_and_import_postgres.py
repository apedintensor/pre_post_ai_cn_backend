"""Reset a PostgreSQL database to the current SQLAlchemy schema and import initial data.

Usage (local / Fly remote via ssh console):
  python scripts/reset_and_import_postgres.py --terms data/derm_dictionary.csv --cases data/ai_prediction_by_id.csv --yes

On Fly (assuming app name reader-study-api):
  fly ssh console -a reader-study-api -C "python scripts/reset_and_import_postgres.py --terms data/derm_dictionary.csv --cases data/ai_prediction_by_id.csv --yes"

Safety:
  - Refuses to run if database URL looks like SQLite.
  - Requires --yes to actually DROP all tables.
  - Uses metadata.drop_all/create_all (no Alembic migrations yet).

Environment:
  Relies on DATABASE_URL / SQLALCHEMY_SYNC_DATABASE_URI in settings.
"""
from __future__ import annotations
import argparse
import os
import sys
from sqlalchemy import text
from app.core.config import settings
from app.db.base import Base
from app.db.session import sync_engine, SessionLocal
from app.models import models  # noqa: F401 import all models so metadata is populated
from import_initial_data import ensure_roles, load_terms_and_synonyms, load_cases  # reuse functions
from pathlib import Path


def is_sqlite(url: str) -> bool:
    return url.startswith("sqlite://")

def assert_postgres():
    url = settings.SQLALCHEMY_SYNC_DATABASE_URI
    if is_sqlite(url):
        print(f"Refusing to run: database URL appears to be SQLite: {url}")
        sys.exit(1)
    if not (url.startswith("postgres://") or url.startswith("postgresql://")):
        print(f"Refusing to run: not a PostgreSQL URL: {url}")
        sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser(description="Hard reset DB then import initial data")
    p.add_argument("--terms", type=Path, required=True, help="derm_dictionary.csv path")
    p.add_argument("--cases", type=Path, required=True, help="ai_prediction_by_id.csv path")
    p.add_argument("--max-cases", type=int, default=None)
    p.add_argument("--dry-run", action="store_true", help="Parse but do not write data")
    p.add_argument("--yes", action="store_true", help="Confirm destructive drop/create")
    return p.parse_args()


def drop_all(engine):
    # Use metadata reflection-safe approach
    print("Dropping all tables ...")
    Base.metadata.drop_all(bind=engine)


def create_all(engine):
    print("Creating tables per current models ...")
    Base.metadata.create_all(bind=engine)


def main():
    args = parse_args()
    assert_postgres()
    if not args.yes and not args.dry_run:
        print("Add --yes to actually drop & recreate tables (or --dry-run). Aborting.")
        sys.exit(1)
    if not args.terms.exists():
        sys.exit(f"Terms file not found: {args.terms}")
    if not args.cases.exists():
        sys.exit(f"Cases file not found: {args.cases}")

    if args.dry_run:
        print("Running in DRY RUN mode – schema not dropped/created.")
    else:
        drop_all(sync_engine)
        create_all(sync_engine)

    db = SessionLocal()
    try:
        from import_initial_data import ensure_roles, load_terms_and_synonyms, load_cases
        ensure_roles(db, ["GP", "Nurse", "Other"])
        load_terms_and_synonyms(db, args.terms, dry_run=args.dry_run)
        load_cases(db, args.cases, batch_size=200, max_cases=args.max_cases, dry_run=args.dry_run)
        if args.dry_run:
            print("Dry run complete – no changes committed.")
        else:
            db.commit()
            print("Reset + import complete.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
