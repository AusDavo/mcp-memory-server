-- 001: record which model produced each vector.
--
-- Run BEFORE deploying the server version that writes embedding_model: that
-- version refuses to start while rows have no model recorded. Safe to rerun.
--
--   docker exec -i mcp-memory-db psql -U memory memory -v ON_ERROR_STOP=1 \
--     < migrations/001-embedding-model.sql

BEGIN;
ALTER TABLE memories ADD COLUMN IF NOT EXISTS embedding_model TEXT;
-- Every existing vector came from the original default model.
UPDATE memories SET embedding_model = 'text-embedding-3-small'
WHERE embedding_model IS NULL AND embedding IS NOT NULL;
COMMIT;
