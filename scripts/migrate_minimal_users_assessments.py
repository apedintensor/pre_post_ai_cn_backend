"""
Minimal one-shot migration for Fly.io Postgres:
 - Add users.gender and backfill to 'Male' when missing
 - Add assessments.investigation_action and next_step_action (left NULL)
 - Drop legacy assessments.biopsy_recommended and referral_recommended

Usage (inside app container or local with DATABASE_URL pointing to Postgres):
  python -m scripts.migrate_minimal_users_assessments

Notes:
 - Idempotent: uses IF NOT EXISTS / IF EXISTS where applicable.
 - Designed for PostgreSQL; on SQLite, some DDL may not work.
 - Connection comes from app.core.config settings (DATABASE_URL / SYNC_DATABASE_URI).
"""
from __future__ import annotations

from sqlalchemy import text

# Reuse project engine/settings
from app.db.session import sync_engine


def run_migration() -> None:
    dialect = sync_engine.dialect.name
    if dialect != "postgresql":
        print(f"Warning: Dialect is '{dialect}'. This script targets PostgreSQL DDL.")

    statements = [
        # Users: ensure gender column and default existing records to 'Male' (when empty/null)
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS gender varchar",
        "UPDATE users SET gender = 'Male' WHERE gender IS NULL OR gender = ''",

        # Assessments: add new action columns (nullable)
        "ALTER TABLE assessments ADD COLUMN IF NOT EXISTS investigation_action varchar",
        "ALTER TABLE assessments ADD COLUMN IF NOT EXISTS next_step_action varchar",

        # Keep NULLs for backfill (no-op safe guard / documents intent)
        "UPDATE assessments SET investigation_action = NULL WHERE investigation_action IS NULL",
        "UPDATE assessments SET next_step_action = NULL WHERE next_step_action IS NULL",

        # Drop legacy columns (now unused by the app)
        "ALTER TABLE assessments DROP COLUMN IF EXISTS biopsy_recommended",
        "ALTER TABLE assessments DROP COLUMN IF EXISTS referral_recommended",
    ]

    with sync_engine.begin() as conn:
        for sql in statements:
            print(f"Executing: {sql}")
            conn.execute(text(sql))

    print("Migration completed successfully.")


if __name__ == "__main__":
    run_migration()
