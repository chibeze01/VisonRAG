from __future__ import annotations

from visionrag.config import Settings
from visionrag.db.repository import StorageRepository
from visionrag.logging_utils import configure_logging
from visionrag.metrics import InMemoryMetrics
from visionrag.Light_merge import LightMerger
from visionrag.providers.embedding import ColPaliEmbeddingProvider
from visionrag.providers.page_resolver import PageResolver
from visionrag.providers.s3_client import S3Client
from visionrag.services.worker_service import IngestionWorker


def main() -> None:
    settings = Settings.from_env()
    configure_logging(settings.log_level)

    repository = StorageRepository(settings.postgres_dsn)
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
    metrics = InMemoryMetrics()

    worker = IngestionWorker(
        settings=settings,
        repository=repository,
        embedding_provider=embedding_provider,
        page_resolver=page_resolver,
        s3_client=s3_client,
        metrics=metrics,
    )
    worker.run_forever()


if __name__ == "__main__":
    main()

