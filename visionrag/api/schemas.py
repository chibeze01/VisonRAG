from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class DocumentIngestRequest(BaseModel):
    s3_bucket: str | None = None
    s3_key: str
    document_id: UUID | None = None


class DocumentIngestResponse(BaseModel):
    job_id: UUID
    document_id: UUID
    status: str


class JobStatusResponse(BaseModel):
    job_id: UUID
    document_id: UUID
    status: str
    attempt_count: int
    lease_owner: str | None = None
    lease_expires_at: datetime | None = None
    error_code: str | None = None
    error_message: str | None = None
    next_run_at: datetime | None = None


class QueryRequestSchema(BaseModel):
    query: str = Field(min_length=1)
    top_k_patches: int | None = None
    top_k_pages: int | None = None
    generate_answer: bool = False


class PageResultSchema(BaseModel):
    document_id: UUID
    s3_bucket: str
    s3_key: str
    page_number: int
    score: float


class QueryResponseSchema(BaseModel):
    pages: list[PageResultSchema]
    answer: str | None = None

