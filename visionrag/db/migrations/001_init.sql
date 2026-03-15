CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    document_id UUID PRIMARY KEY,
    s3_bucket TEXT NOT NULL,
    s3_key TEXT NOT NULL,
    source_etag TEXT,
    status TEXT NOT NULL CHECK (status IN ('pending', 'indexed', 'failed')),
    page_count INT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (s3_bucket, s3_key)
);

CREATE TABLE IF NOT EXISTS ingestion_jobs (
    job_id UUID PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    status TEXT NOT NULL CHECK (status IN ('queued', 'leased', 'succeeded', 'failed', 'dead_letter')),
    attempt_count INT NOT NULL DEFAULT 0,
    lease_owner TEXT,
    lease_expires_at TIMESTAMPTZ,
    next_run_at TIMESTAMPTZ,
    error_code TEXT,
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_queue
    ON ingestion_jobs (status, next_run_at, created_at);

CREATE TABLE IF NOT EXISTS patch_embeddings (
    id BIGSERIAL PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    page_number INT NOT NULL,
    patch_index INT NOT NULL,
    patch_bbox JSONB NOT NULL,
    embedding VECTOR(128) NOT NULL,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (document_id, page_number, patch_index, model_version)
);

CREATE INDEX IF NOT EXISTS idx_patch_embeddings_doc_page
    ON patch_embeddings (document_id, page_number);

CREATE INDEX IF NOT EXISTS idx_patch_embeddings_hnsw
    ON patch_embeddings USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS schema_migrations (
    migration_id TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
