-- Keep users & roles; purge the rest. Safe for Fly.io Postgres.
-- Usage: fly pg connect -a <pg-app>  (opens psql), then paste this whole script.
-- Or run via psql -f after connecting a proxy.

BEGIN;

-- Ensure schema matches app expectations
ALTER TABLE cases ADD COLUMN IF NOT EXISTS typical_diagnosis boolean;

ALTER TABLE users ADD COLUMN IF NOT EXISTS gender varchar;
UPDATE users SET gender = 'Male' WHERE gender IS NULL OR gender = '';

-- Drop legacy columns no longer used by the app (optional, safe)
ALTER TABLE assessments DROP COLUMN IF EXISTS biopsy_recommended;
ALTER TABLE assessments DROP COLUMN IF EXISTS referral_recommended;

-- Ensure optional tables exist so TRUNCATE doesn't fail if referenced elsewhere
CREATE TABLE IF NOT EXISTS management_strategies (
  id SERIAL PRIMARY KEY,
  name varchar NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS case_metadata (
  id SERIAL PRIMARY KEY,
  case_id integer NOT NULL UNIQUE REFERENCES cases(id),
  age integer,
  gender varchar,
  fever_history boolean,
  psoriasis_history boolean,
  other_notes text
);

-- Purge domain data (keep users and roles)
TRUNCATE TABLE
  diagnosis_entries,
  assessments,
  reader_case_assignments,
  block_feedback,
  ai_outputs,
  images,
  case_metadata,
  management_strategies,
  diagnosis_synonyms,
  diagnosis_terms,
  cases
RESTART IDENTITY CASCADE;

COMMIT;

-- Note:
-- 1) After this, re-seed terms/cases/images/AI outputs, e.g. run
--    python scripts/seed_basic.py in the app container (Fly SSH).
-- 2) The /management_strategies endpoint seeds defaults on first GET if empty.
