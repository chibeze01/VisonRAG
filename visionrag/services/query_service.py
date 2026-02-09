from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from PIL import Image

from visionrag.config import Settings
from visionrag.metrics import InMemoryMetrics
from visionrag.providers.answer_generator import AnswerGenerator
from visionrag.providers.embedding import EmbeddingProvider
from visionrag.providers.page_resolver import PageResolver
from visionrag.rerank import aggregate_patch_scores
from visionrag.types import PageResult

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from visionrag.db.repository import StorageRepository
else:
    StorageRepository = Any


@dataclass(frozen=True)
class QueryRequest:
    query: str
    top_k_patches: int
    top_k_pages: int
    generate_answer: bool


@dataclass(frozen=True)
class QueryResponse:
    pages: list[PageResult]
    answer: str | None


class QueryService:
    def __init__(
        self,
        settings: Settings,
        repository: StorageRepository,
        embedding_provider: EmbeddingProvider,
        page_resolver: PageResolver,
        metrics: InMemoryMetrics,
        answer_generator: AnswerGenerator | None = None,
    ) -> None:
        self._settings = settings
        self._repository = repository
        self._embedding_provider = embedding_provider
        self._page_resolver = page_resolver
        self._answer_generator = answer_generator
        self._metrics = metrics

    def query(self, request: QueryRequest) -> QueryResponse:
        start_total = time.perf_counter()

        start_embed = time.perf_counter()
        query_vector = self._embedding_provider.embed_query(request.query)
        self._metrics.observe("query.embed_seconds", time.perf_counter() - start_embed)

        start_search = time.perf_counter()
        patch_hits = self._repository.search_patches(
            query_vector=query_vector,
            k=request.top_k_patches,
            model_name=self._settings.model_name,
            model_version=self._settings.model_version,
        )
        self._metrics.observe("query.vector_search_seconds", time.perf_counter() - start_search)

        ranked_pages = aggregate_patch_scores(
            hits=patch_hits,
            top_k_pages=request.top_k_pages,
            top_m_patches=self._settings.query_page_top_m_patches,
            max_pages_per_document=self._settings.query_max_pages_per_document,
        )

        answer: str | None = None
        if request.generate_answer and ranked_pages:
            if not self._answer_generator:
                raise RuntimeError("No answer generator configured.")

            start_fetch = time.perf_counter()
            images = self._load_images_for_pages(ranked_pages)
            self._metrics.observe("query.render_seconds", time.perf_counter() - start_fetch)

            start_llm = time.perf_counter()
            answer = self._answer_generator.answer(request.query, images, ranked_pages)
            self._metrics.observe("query.llm_seconds", time.perf_counter() - start_llm)

        self._metrics.observe("query.total_seconds", time.perf_counter() - start_total)
        self._metrics.inc("query.requests")
        return QueryResponse(pages=ranked_pages, answer=answer)

    def _load_images_for_pages(self, pages: list[PageResult]) -> list[Image.Image]:
        per_doc: dict[tuple[str, str], list[int]] = {}
        for page in pages:
            key = (page.s3_bucket, page.s3_key)
            per_doc.setdefault(key, []).append(page.page_number)

        rendered: dict[tuple[str, str, int], Image.Image] = {}
        for (bucket, s3_key), page_numbers in per_doc.items():
            pdf_bytes = self._page_resolver.fetch_pdf(bucket=bucket, key=s3_key)
            images = self._page_resolver.render_pages(pdf_bytes=pdf_bytes, page_numbers=page_numbers)
            for page_number, image in images.items():
                rendered[(bucket, s3_key, page_number)] = image

        ordered_images: list[Image.Image] = []
        for row in pages:
            image = rendered.get((row.s3_bucket, row.s3_key, row.page_number))
            if image is None:
                logger.warning(
                    "Missing rendered page image for %s/%s page %s",
                    row.s3_bucket,
                    row.s3_key,
                    row.page_number,
                )
                continue
            ordered_images.append(image)
        return ordered_images

    def metrics_snapshot(self) -> dict[str, Any]:
        return self._metrics.snapshot()
