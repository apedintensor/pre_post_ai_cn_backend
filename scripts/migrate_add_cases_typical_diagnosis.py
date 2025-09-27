"""
Add cases.typical_diagnosis to Postgres (compat with ORM/tests).

Usage:
  python -m scripts.migrate_add_cases_typical_diagnosis

Idempotent: uses IF NOT EXISTS guard.
"""
from __future__ import annotations

from sqlalchemy import text
from app.db.session import sync_engine


def run() -> None:
    if sync_engine.dialect.name != "postgresql":
        print(f"Warning: dialect is {sync_engine.dialect.name}; this script targets Postgres.")

    sql = "ALTER TABLE cases ADD COLUMN IF NOT EXISTS typical_diagnosis boolean"
    with sync_engine.begin() as conn:
        print(f"Executing: {sql}")
        conn.execute(text(sql))
    print("cases.typical_diagnosis ensured.")


if __name__ == "__main__":
    run()
