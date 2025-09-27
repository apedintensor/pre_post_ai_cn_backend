"""Utility script to (re)build the database schema for the refactored models.

Usage (from backend/ directory):
  python -m scripts.rebuild_schema --drop

Options:
  --drop   Drop existing tables before creating (DANGEROUS: data loss)
"""
from sqlalchemy import inspect
from sqlalchemy.orm import Session
from app.db.session import sync_engine, SessionLocal
from app.db.base import Base

# Import models so metadata is populated
from app.models import models  # noqa: F401
import argparse


def main(drop: bool = False):
    if drop:
        Base.metadata.drop_all(bind=sync_engine)
        print("Dropped all tables.")
    Base.metadata.create_all(bind=sync_engine)
    print("Created tables.")
    insp = inspect(sync_engine)
    print("Current tables:", insp.get_table_names())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--drop", action="store_true", help="Drop existing tables before create")
    args = parser.parse_args()
    main(drop=args.drop)
