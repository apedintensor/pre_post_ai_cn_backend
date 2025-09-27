-- Minimal migration: add users.gender, add assessment actions, drop legacy flags.
-- Safe to run multiple times (IF NOT EXISTS / IF EXISTS).

BEGIN;

-- Users: add gender and backfill as 'Male' when missing
ALTER TABLE users ADD COLUMN IF NOT EXISTS gender varchar;
UPDATE users SET gender = 'Male' WHERE gender IS NULL OR gender = '';

-- Assessments: add new action fields (default NULL)
ALTER TABLE assessments ADD COLUMN IF NOT EXISTS investigation_action varchar;
ALTER TABLE assessments ADD COLUMN IF NOT EXISTS next_step_action varchar;

-- Explicitly ensure missing values remain NULL (no-op but documents intent)
UPDATE assessments SET investigation_action = NULL WHERE investigation_action IS NULL;
UPDATE assessments SET next_step_action = NULL WHERE next_step_action IS NULL;

-- Drop legacy columns no longer used
ALTER TABLE assessments DROP COLUMN IF EXISTS biopsy_recommended;
ALTER TABLE assessments DROP COLUMN IF EXISTS referral_recommended;

COMMIT;

-- Usage on Fly Postgres:
--   fly pg connect -a <your-pg-app>
--   -- then paste this script into the psql session and run.
