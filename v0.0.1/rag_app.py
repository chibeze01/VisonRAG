from __future__ import annotations

from visionrag.config import Settings
from visionrag.db.repository import StorageRepository
from visionrag.logging_utils import configure_logging
from visionrag.metrics import InMemoryMetrics
from visionrag.providers.answer_generator import create_answer_generator
from visionrag.providers.embedding import ColPaliEmbeddingProvider
from visionrag.providers.page_resolver import PageResolver
from visionrag.providers.s3_client import S3Client
from visionrag.services.query_service import QueryRequest, QueryService


def main() -> None:
    settings = Settings.from_env()
    configure_logging(settings.log_level)

    repository = StorageRepository(settings.postgres_dsn)
    s3_client = S3Client(region_name=settings.aws_region)
    page_resolver = PageResolver(s3_client=s3_client, dpi=settings.render_dpi)
    embedding_provider = ColPaliEmbeddingProvider(model_name=settings.model_name, device=settings.model_device)
    metrics = InMemoryMetrics()

    answer_generator = create_answer_generator(settings)

    query_service = QueryService(
        settings=settings,
        repository=repository,
        embedding_provider=embedding_provider,
        page_resolver=page_resolver,
        answer_generator=answer_generator,
        metrics=metrics,
    )

    while True:
        user_query = input("\nEnter your question (or 'quit' to exit): ").strip()
        if user_query.lower() in {"quit", "exit", "q"}:
            break

        response = query_service.query(
            QueryRequest(
                query=user_query,
                top_k_patches=settings.query_default_top_k_patches,
                top_k_pages=settings.query_default_top_k_pages,
                generate_answer=bool(answer_generator),
            )
        )

        if not response.pages:
            print("No relevant pages found.")
            continue

        print("\nTop evidence pages:")
        for page in response.pages:
            print(
                f"- document_id={page.document_id} s3={page.s3_bucket}/{page.s3_key} "
                f"page={page.page_number} score={page.score:.4f}"
            )

        if response.answer:
            print("\n--- ANSWER ---")
            print(response.answer)


if __name__ == "__main__":
    main()
