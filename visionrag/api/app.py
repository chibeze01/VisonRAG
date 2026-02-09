from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any
from uuid import UUID

from fastapi import FastAPI, HTTPException

from visionrag.api.schemas import (
    DocumentIngestRequest,
    DocumentIngestResponse,
    JobStatusResponse,
    QueryRequestSchema,
    QueryResponseSchema,
)
from visionrag.config import Settings
from visionrag.db.repository import StorageRepository
from visionrag.logging_utils import configure_logging
from visionrag.metrics import InMemoryMetrics
from visionrag.providers.answer_generator import GeminiAnswerGenerator
from visionrag.providers.embedding import ColPaliEmbeddingProvider
from visionrag.providers.page_resolver import PageResolver
from visionrag.providers.s3_client import S3Client
from visionrag.services.query_service import QueryRequest, QueryService

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings or Settings.from_env()
    configure_logging(cfg.log_level)

    repository = StorageRepository(cfg.postgres_dsn)
    s3_client = S3Client(region_name=cfg.aws_region)
    page_resolver = PageResolver(s3_client=s3_client, dpi=cfg.render_dpi)
    embedding_provider = ColPaliEmbeddingProvider(model_name=cfg.model_name, device=cfg.model_device)
    metrics = InMemoryMetrics()
    answer_generator = None
    if cfg.gemini_api_key:
        answer_generator = GeminiAnswerGenerator(api_key=cfg.gemini_api_key, model=cfg.gemini_model)

    query_service = QueryService(
        settings=cfg,
        repository=repository,
        embedding_provider=embedding_provider,
        page_resolver=page_resolver,
        answer_generator=answer_generator,
        metrics=metrics,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        logger.info("Starting query-api")
        yield
        logger.info("Stopping query-api")

    app = FastAPI(title="VisionRAG Query API", version="1.0.0", lifespan=lifespan)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz() -> dict[str, Any]:
        db_ok = repository.healthcheck()
        s3_ok = s3_client.healthcheck(cfg.default_s3_bucket)
        model_ready = True
        try:
            embedding_provider.warm()
        except Exception:
            model_ready = False
        status = "ready" if db_ok and s3_ok and model_ready else "not_ready"
        return {"status": status, "db": db_ok, "s3": s3_ok, "model": model_ready}

    @app.post("/v1/documents", response_model=DocumentIngestResponse)
    def ingest_document(payload: DocumentIngestRequest) -> DocumentIngestResponse:
        bucket = payload.s3_bucket or cfg.default_s3_bucket
        if not bucket:
            raise HTTPException(status_code=400, detail="s3_bucket is required when DEFAULT_S3_BUCKET is not set.")

        doc = repository.upsert_document(
            s3_bucket=bucket,
            s3_key=payload.s3_key,
            document_id=payload.document_id,
            status="pending",
        )
        job = repository.create_ingestion_job(doc.document_id)
        return DocumentIngestResponse(job_id=job.job_id, document_id=doc.document_id, status=job.status)

    @app.get("/v1/jobs/{job_id}", response_model=JobStatusResponse)
    def get_job_status(job_id: UUID) -> JobStatusResponse:
        job = repository.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found.")
        return JobStatusResponse(**job.__dict__)

    @app.post("/v1/query", response_model=QueryResponseSchema)
    def query(payload: QueryRequestSchema) -> QueryResponseSchema:
        top_k_patches = payload.top_k_patches or cfg.query_default_top_k_patches
        top_k_pages = payload.top_k_pages or cfg.query_default_top_k_pages

        result = query_service.query(
            QueryRequest(
                query=payload.query,
                top_k_patches=top_k_patches,
                top_k_pages=top_k_pages,
                generate_answer=payload.generate_answer,
            )
        )
        return QueryResponseSchema(
            pages=[page.__dict__ for page in result.pages],
            answer=result.answer,
        )

    @app.get("/metrics")
    def get_metrics() -> dict[str, Any]:
        return {name: point.__dict__ for name, point in query_service.metrics_snapshot().items()}

    return app


app = create_app()
