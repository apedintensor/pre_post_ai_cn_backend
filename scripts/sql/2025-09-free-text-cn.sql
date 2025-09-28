-- Free-text only diagnostics (CN version)
-- Make mapping to canonical diagnosis term optional

BEGIN;

ALTER TABLE diagnosis_entries
  ALTER COLUMN diagnosis_term_id DROP NOT NULL;

-- Optional: ensure at most one entry per assessment (comment out if keeping ranked list)
-- DROP CONSTRAINT IF EXISTS diagnosis_entries_assessment_id_rank_key;
-- DO $$
-- BEGIN
--   IF NOT EXISTS (
--     SELECT 1 FROM pg_indexes
--     WHERE schemaname = 'public'
--       AND tablename = 'diagnosis_entries'
--       AND indexname = 'diagnosis_entries_assessment_id_unique'
--   ) THEN
--     CREATE UNIQUE INDEX diagnosis_entries_assessment_id_unique
--       ON diagnosis_entries (assessment_id);
--   END IF;
-- END $$;

COMMIT;
