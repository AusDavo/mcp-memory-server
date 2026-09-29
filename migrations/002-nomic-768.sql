-- 002: switch to nomic-embed-text (768 dims). Two phases.
--
-- Phase A (server running, no downtime): add the staging columns, then fill
-- them with backfill_embeddings.py.
--
--   docker exec -i mcp-memory-db psql -U memory memory -v ON_ERROR_STOP=1 \
--     -v phase=a < migrations/002-nomic-768.sql
--
-- Phase B (server stopped): rerun the backfill to catch rows written or edited
-- since phase A, then swap the columns. Aborts, changing nothing, if any row
-- lacks a current vector. The OpenAI vectors are kept as embedding_openai for
-- rollback (see README); drop that column after the soak period.
--
--   docker exec -i mcp-memory-db psql -U memory memory -v ON_ERROR_STOP=1 \
--     -v phase=b < migrations/002-nomic-768.sql

\if :{?phase}
\else
  \echo 'set -v phase=a or -v phase=b'
  \quit
\endif

SELECT :'phase' = 'a' AS is_a \gset

\if :is_a
BEGIN;
ALTER TABLE memories ADD COLUMN IF NOT EXISTS embedding_768 vector(768);
-- md5 of the content each staged vector was made from, so the phase B
-- backfill can tell which rows changed after they were staged.
ALTER TABLE memories ADD COLUMN IF NOT EXISTS embedding_768_src TEXT;
ALTER TABLE memories ADD COLUMN IF NOT EXISTS embedding_768_truncated BOOLEAN;
COMMIT;
\else
BEGIN;
DO $$
DECLARE stale int;
BEGIN
  SELECT COUNT(*) INTO stale FROM memories
  WHERE embedding_768 IS NULL OR embedding_768_src IS DISTINCT FROM md5(content);
  IF stale > 0 THEN
    RAISE EXCEPTION '% row(s) lack a current 768-dim vector; rerun backfill_embeddings.py', stale;
  END IF;
END $$;

DROP INDEX IF EXISTS idx_memories_embedding;
ALTER TABLE memories RENAME COLUMN embedding TO embedding_openai;
ALTER TABLE memories RENAME COLUMN embedding_768 TO embedding;
UPDATE memories SET
  embedding_model = 'nomic-embed-text',
  metadata = CASE WHEN embedding_768_truncated
    THEN jsonb_set(COALESCE(metadata, '{}'), '{embedding}', '{"truncated": true, "max_tokens": 2048}')
    ELSE metadata - 'embedding' END;
ALTER TABLE memories DROP COLUMN embedding_768_src, DROP COLUMN embedding_768_truncated;
CREATE INDEX idx_memories_embedding ON memories USING hnsw (embedding vector_cosine_ops);
COMMIT;
\endif
