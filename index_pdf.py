from __future__ import annotations

import argparse

from visionrag.config import Settings
from visionrag.db.repository import StorageRepository
from visionrag.logging_utils import configure_logging
from visionrag.metrics import InMemoryMetrics
from visionrag.Light_merge import LightMerger
from visionrag.providers.embedding import ColPaliEmbeddingProvider
from visionrag.providers.page_resolver import PageResolver
from visionrag.providers.s3_client import S3Client
from visionrag.services.worker_service import IngestionWorker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Queue and optionally process a PDF ingestion job from S3.")
    parser.add_argument("--s3-key", required=True, help="S3 object key for the PDF.")
    parser.add_argument("--s3-bucket", default=None, help="S3 bucket (uses DEFAULT_S3_BUCKET if omitted).")
    parser.add_argument(
        "--process-now",
        action="store_true",
        help="Process exactly one claimed job immediately (useful for local testing).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings.from_env()
    configure_logging(settings.log_level)

    bucket = args.s3_bucket or settings.default_s3_bucket
    if not bucket:
        raise ValueError("s3_bucket must be provided via --s3-bucket or DEFAULT_S3_BUCKET.")

    repository = StorageRepository(settings.postgres_dsn)
    document = repository.upsert_document(s3_bucket=bucket, s3_key=args.s3_key, status="pending")
    job = repository.create_ingestion_job(document.document_id)

    print(f"Queued job_id={job.job_id} document_id={document.document_id} s3={bucket}/{args.s3_key}")

    if not args.process_now:
        return

    s3_client = S3Client(region_name=settings.aws_region)
    page_resolver = PageResolver(s3_client=s3_client, dpi=settings.render_dpi)
    merger = (
        LightMerger(
            merge_factor=settings.light_merge_factor,
            min_clusters=settings.light_merge_min_clusters,
            bbox_density_percentile=settings.light_merge_bbox_density_percentile,
        )
        if settings.light_merge_enabled
        else None
    )
    embedding_provider = ColPaliEmbeddingProvider(
        model_name=settings.model_name,
        device=settings.model_device,
        merger=merger,
    )
    worker = IngestionWorker(
        settings=settings,
        repository=repository,
        embedding_provider=embedding_provider,
        page_resolver=page_resolver,
        s3_client=s3_client,
        metrics=InMemoryMetrics(),
    )
    outcome = worker.run_once()
    print(f"Processed: {outcome.processed} job_id={outcome.job_id}")


if __name__ == "__main__":
    main()
