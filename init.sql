CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content TEXT NOT NULL,
    embedding vector(1536),
    source TEXT DEFAULT 'manual',
    tags TEXT[] DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- HNSW index for semantic search (works on empty tables, unlike IVFFlat)
CREATE INDEX IF NOT EXISTS idx_memories_embedding
    ON memories USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_memories_tags ON memories USING GIN (tags);

CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_memories_content_fts
    ON memories USING GIN (to_tsvector('english', content));
