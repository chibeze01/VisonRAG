from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID


@dataclass(frozen=True)
class PatchBBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def as_json(self) -> dict[str, float]:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}


@dataclass(frozen=True)
class PatchEmbedding:
    page_number: int
    patch_index: int
    patch_bbox: PatchBBox
    embedding: list[float]


@dataclass(frozen=True)
class DocumentRecord:
    document_id: UUID
    s3_bucket: str
    s3_key: str
    source_etag: str | None
    status: str
    page_count: int | None


@dataclass(frozen=True)
class JobRecord:
    job_id: UUID
    document_id: UUID
    status: str
    attempt_count: int
    lease_owner: str | None
    lease_expires_at: datetime | None
    error_code: str | None
    error_message: str | None
    next_run_at: datetime | None


@dataclass(frozen=True)
class ClaimedJob:
    job_id: UUID
    document_id: UUID
    s3_bucket: str
    s3_key: str
    source_etag: str | None
    page_count: int | None
    attempt_count: int


@dataclass(frozen=True)
class PatchSearchHit:
    document_id: UUID
    s3_bucket: str
    s3_key: str
    page_number: int
    patch_index: int
    patch_bbox: dict[str, Any]
    score: float


@dataclass(frozen=True)
class PageResult:
    document_id: UUID
    s3_bucket: str
    s3_key: str
    page_number: int
    score: float
