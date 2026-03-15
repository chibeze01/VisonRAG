from __future__ import annotations

import logging
import socket
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from visionrag.config import Settings
from visionrag.metrics import InMemoryMetrics
from visionrag.providers.embedding import EmbeddingProvider
from visionrag.providers.page_resolver import PageResolver
from visionrag.providers.s3_client import S3Client
from visionrag.types import ClaimedJob, PatchEmbedding

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from visionrag.db.repository import StorageRepository
else:
    StorageRepository = Any


def compute_backoff_delay(attempt_count: int, base_seconds: int, max_seconds: int) -> int:
    delay = base_seconds * (2 ** max(0, attempt_count - 1))
    return min(delay, max_seconds)


@dataclass(frozen=True)
class WorkerOutcome:
    processed: bool
    job_id: str | None = None


class IngestionWorker:
    def __init__(
        self,
        settings: Settings,
        repository: StorageRepository,
        embedding_provider: EmbeddingProvider,
        page_resolver: PageResolver,
        s3_client: S3Client,
        metrics: InMemoryMetrics,
        worker_id: str | None = None,
    ) -> None:
        self._settings = settings
        self._repository = repository
        self._embedding_provider = embedding_provider
        self._page_resolver = page_resolver
        self._s3_client = s3_client
        self._metrics = metrics
        self._worker_id = worker_id or f"{socket.gethostname()}-{int(time.time())}"

    def run_forever(self) -> None:
        logger.info("Worker started with id=%s", self._worker_id)
        self._embedding_provider.warm()
        while True:
            outcome = self.run_once()
            if not outcome.processed:
                time.sleep(self._settings.worker_poll_interval_seconds)

    def run_once(self) -> WorkerOutcome:
        claimed = self._repository.claim_next_job(
            worker_id=self._worker_id,
            lease_seconds=self._settings.worker_lease_seconds,
            max_attempts=self._settings.worker_max_attempts,
        )
        if not claimed:
            return WorkerOutcome(processed=False)

        started = time.perf_counter()
        try:
            self._process_job(claimed)
            self._metrics.inc("ingestion.jobs_succeeded")
            self._metrics.observe("ingestion.job_seconds", time.perf_counter() - started)
            return WorkerOutcome(processed=True, job_id=str(claimed.job_id))
        except Exception as exc:
            logger.exception("Job failed job_id=%s document_id=%s", claimed.job_id, claimed.document_id)
            delay = compute_backoff_delay(
                attempt_count=claimed.attempt_count,
                base_seconds=self._settings.worker_retry_base_seconds,
                max_seconds=self._settings.worker_retry_max_seconds,
            )
            self._repository.mark_job_failure(
                job_id=claimed.job_id,
                document_id=claimed.document_id,
                attempt_count=claimed.attempt_count,
                max_attempts=self._settings.worker_max_attempts,
                retry_delay_seconds=delay,
                error_code=type(exc).__name__,
                error_message=str(exc),
            )
            self._metrics.inc("ingestion.jobs_failed")
            return WorkerOutcome(processed=True, job_id=str(claimed.job_id))

    def _process_job(self, claimed: ClaimedJob) -> None:
        logger.info(
            "Processing job_id=%s document_id=%s s3=%s/%s attempt=%s",
            claimed.job_id,
            claimed.document_id,
            claimed.s3_bucket,
            claimed.s3_key,
            claimed.attempt_count,
        )

        object_info = self._s3_client.head_object(claimed.s3_bucket, claimed.s3_key)
        etag = object_info.etag

        if (
            claimed.source_etag
            and etag
            and claimed.source_etag == etag
            and self._repository.has_embeddings_for_version(
                document_id=claimed.document_id,
                model_name=self._settings.model_name,
                model_version=self._settings.model_version,
            )
        ):
            logger.info(
                "Skipping reindex (etag unchanged) document_id=%s etag=%s", claimed.document_id, etag
            )
            self._repository.mark_job_success(
                job_id=claimed.job_id,
                document_id=claimed.document_id,
                page_count=claimed.page_count or 0,
                source_etag=etag,
            )
            return

        pdf_bytes = self._page_resolver.fetch_pdf(claimed.s3_bucket, claimed.s3_key)
        page_count = self._count_pages(pdf_bytes)
        all_rows: list[PatchEmbedding] = []

        for page_number in range(1, page_count + 1):
            page_images = self._page_resolver.render_pages(pdf_bytes=pdf_bytes, page_numbers=[page_number])
            image = page_images.get(page_number)
            if image is None:
                continue
            all_rows.extend(self._embedding_provider.embed_page(image=image, page_number=page_number))

            if page_number % self._settings.worker_lease_refresh_every_pages == 0:
                self._repository.renew_lease(
                    job_id=claimed.job_id,
                    worker_id=self._worker_id,
                    lease_seconds=self._settings.worker_lease_seconds,
                )

        self._repository.replace_patch_embeddings(
            document_id=claimed.document_id,
            model_name=self._settings.model_name,
            model_version=self._settings.model_version,
            patches=all_rows,
        )
        self._repository.mark_job_success(
            job_id=claimed.job_id,
            document_id=claimed.document_id,
            page_count=page_count,
            source_etag=etag,
        )

    @staticmethod
    def _count_pages(pdf_bytes: bytes) -> int:
        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError("PyMuPDF is required for page counting.") from exc

        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            return len(doc)
