from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Iterable
from uuid import UUID, uuid4

import psycopg
from psycopg.rows import dict_row

from visionrag.types import ClaimedJob, DocumentRecord, JobRecord, PatchEmbedding, PatchSearchHit


def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{float(item):.8f}" for item in values) + "]"


class StorageRepository:
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    def _connect(self) -> psycopg.Connection:
        return psycopg.connect(self._dsn, row_factory=dict_row)

    def healthcheck(self) -> bool:
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
            return True
        except Exception:
            return False

    def upsert_document(
        self,
        s3_bucket: str,
        s3_key: str,
        document_id: UUID | None = None,
        status: str = "pending",
    ) -> DocumentRecord:
        doc_id = document_id or uuid4()
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents (document_id, s3_bucket, s3_key, status)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (s3_bucket, s3_key)
                DO UPDATE SET updated_at = NOW(), status = EXCLUDED.status
                RETURNING document_id, s3_bucket, s3_key, source_etag, status, page_count
                """,
                (doc_id, s3_bucket, s3_key, status),
            )
            row = cur.fetchone()
            conn.commit()
        return DocumentRecord(
            document_id=row["document_id"],
            s3_bucket=row["s3_bucket"],
            s3_key=row["s3_key"],
            source_etag=row["source_etag"],
            status=row["status"],
            page_count=row["page_count"],
        )

    def create_ingestion_job(self, document_id: UUID) -> JobRecord:
        job_id = uuid4()
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ingestion_jobs (job_id, document_id, status, next_run_at)
                VALUES (%s, %s, 'queued', NOW())
                RETURNING job_id, document_id, status, attempt_count, lease_owner,
                          lease_expires_at, error_code, error_message, next_run_at
                """,
                (job_id, document_id),
            )
            row = cur.fetchone()
            conn.commit()
        return JobRecord(**row)

    def get_job(self, job_id: UUID) -> JobRecord | None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT job_id, document_id, status, attempt_count, lease_owner,
                       lease_expires_at, error_code, error_message, next_run_at
                FROM ingestion_jobs
                WHERE job_id = %s
                """,
                (job_id,),
            )
            row = cur.fetchone()
        if not row:
            return None
        return JobRecord(**row)

    def claim_next_job(self, worker_id: str, lease_seconds: int, max_attempts: int) -> ClaimedJob | None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                WITH candidate AS (
                    SELECT j.job_id
                    FROM ingestion_jobs j
                    WHERE (
                            j.status = 'queued'
                            OR (j.status = 'leased' AND j.lease_expires_at < NOW())
                            OR (j.status = 'failed' AND j.attempt_count < %s)
                        )
                      AND COALESCE(j.next_run_at, NOW()) <= NOW()
                    ORDER BY j.created_at
                    FOR UPDATE SKIP LOCKED
                    LIMIT 1
                )
                UPDATE ingestion_jobs j
                SET status = 'leased',
                    lease_owner = %s,
                    lease_expires_at = NOW() + (%s * INTERVAL '1 second'),
                    started_at = COALESCE(j.started_at, NOW()),
                    updated_at = NOW(),
                    attempt_count = j.attempt_count + 1
                FROM candidate c
                WHERE j.job_id = c.job_id
                RETURNING j.job_id, j.document_id, j.attempt_count
                """,
                (max_attempts, worker_id, lease_seconds),
            )
            claimed = cur.fetchone()
            if not claimed:
                conn.commit()
                return None

            cur.execute(
                """
                SELECT d.s3_bucket, d.s3_key, d.source_etag, d.page_count
                FROM documents d
                WHERE d.document_id = %s
                """,
                (claimed["document_id"],),
            )
            doc = cur.fetchone()
            conn.commit()

        return ClaimedJob(
            job_id=claimed["job_id"],
            document_id=claimed["document_id"],
            s3_bucket=doc["s3_bucket"],
            s3_key=doc["s3_key"],
            source_etag=doc["source_etag"],
            page_count=doc["page_count"],
            attempt_count=claimed["attempt_count"],
        )

    def renew_lease(self, job_id: UUID, worker_id: str, lease_seconds: int) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE ingestion_jobs
                SET lease_expires_at = NOW() + (%s * INTERVAL '1 second'),
                    updated_at = NOW()
                WHERE job_id = %s
                  AND lease_owner = %s
                  AND status = 'leased'
                """,
                (lease_seconds, job_id, worker_id),
            )
            conn.commit()

    def mark_job_success(
        self,
        job_id: UUID,
        document_id: UUID,
        page_count: int,
        source_etag: str | None,
    ) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE ingestion_jobs
                SET status = 'succeeded',
                    finished_at = NOW(),
                    lease_owner = NULL,
                    lease_expires_at = NULL,
                    next_run_at = NULL,
                    updated_at = NOW()
                WHERE job_id = %s
                """,
                (job_id,),
            )
            cur.execute(
                """
                UPDATE documents
                SET status = 'indexed',
                    source_etag = %s,
                    page_count = %s,
                    updated_at = NOW()
                WHERE document_id = %s
                """,
                (source_etag, page_count, document_id),
            )
            conn.commit()

    def mark_job_failure(
        self,
        job_id: UUID,
        document_id: UUID,
        attempt_count: int,
        max_attempts: int,
        retry_delay_seconds: int,
        error_code: str,
        error_message: str,
    ) -> None:
        is_dead = attempt_count >= max_attempts
        next_status = "dead_letter" if is_dead else "failed"
        next_run = None if is_dead else datetime.now(timezone.utc) + timedelta(seconds=retry_delay_seconds)

        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE ingestion_jobs
                SET status = %s,
                    error_code = %s,
                    error_message = %s,
                    next_run_at = %s,
                    lease_owner = NULL,
                    lease_expires_at = NULL,
                    finished_at = CASE WHEN %s = 'dead_letter' THEN NOW() ELSE finished_at END,
                    updated_at = NOW()
                WHERE job_id = %s
                """,
                (next_status, error_code, error_message[:2000], next_run, next_status, job_id),
            )
            if is_dead:
                cur.execute(
                    """
                    UPDATE documents
                    SET status = 'failed', updated_at = NOW()
                    WHERE document_id = %s
                    """,
                    (document_id,),
                )
            conn.commit()

    def replace_patch_embeddings(
        self,
        document_id: UUID,
        model_name: str,
        model_version: str,
        patches: Iterable[PatchEmbedding],
    ) -> int:
        rows = list(patches)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM patch_embeddings
                WHERE document_id = %s AND model_name = %s AND model_version = %s
                """,
                (document_id, model_name, model_version),
            )
            cur.executemany(
                """
                INSERT INTO patch_embeddings (
                    document_id, page_number, patch_index, patch_bbox, embedding, model_name, model_version
                )
                VALUES (%s, %s, %s, %s, %s::vector, %s, %s)
                """,
                [
                    (
                        document_id,
                        row.page_number,
                        row.patch_index,
                        json.dumps(row.patch_bbox.as_json()),
                        _vector_literal(row.embedding),
                        model_name,
                        model_version,
                    )
                    for row in rows
                ],
            )
            conn.commit()
        return len(rows)

    def has_embeddings_for_version(self, document_id: UUID, model_name: str, model_version: str) -> bool:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT EXISTS(
                    SELECT 1
                    FROM patch_embeddings
                    WHERE document_id = %s AND model_name = %s AND model_version = %s
                )
                """,
                (document_id, model_name, model_version),
            )
            exists = bool(cur.fetchone()["exists"])
        return exists

    def search_patches(
        self,
        query_vector: list[float],
        k: int,
        model_name: str,
        model_version: str,
    ) -> list[PatchSearchHit]:
        query_literal = _vector_literal(query_vector)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    p.document_id,
                    d.s3_bucket,
                    d.s3_key,
                    p.page_number,
                    p.patch_index,
                    p.patch_bbox,
                    1 - (p.embedding <=> %s::vector) AS score
                FROM patch_embeddings p
                JOIN documents d ON d.document_id = p.document_id
                WHERE p.model_name = %s
                  AND p.model_version = %s
                ORDER BY p.embedding <=> %s::vector
                LIMIT %s
                """,
                (query_literal, model_name, model_version, query_literal, k),
            )
            rows = cur.fetchall()
        return [PatchSearchHit(**row) for row in rows]

    def get_queue_depth(self) -> int:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*) AS queue_depth
                FROM ingestion_jobs
                WHERE status IN ('queued', 'failed')
                  AND COALESCE(next_run_at, NOW()) <= NOW()
                """
            )
            return int(cur.fetchone()["queue_depth"])
