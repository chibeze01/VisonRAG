from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from visionrag.types import PageResult, PatchSearchHit


def aggregate_patch_scores(
    hits: Iterable[PatchSearchHit],
    top_k_pages: int,
    top_m_patches: int = 5,
    max_pages_per_document: int = 2,
) -> list[PageResult]:
    grouped: dict[tuple[str, int], list[PatchSearchHit]] = defaultdict(list)
    document_meta: dict[str, tuple[str, str]] = {}

    for hit in hits:
        doc_key = str(hit.document_id)
        grouped[(doc_key, hit.page_number)].append(hit)
        document_meta[doc_key] = (hit.s3_bucket, hit.s3_key)

    candidates: list[PageResult] = []
    for (doc_key, page_number), page_hits in grouped.items():
        top_hits = sorted(page_hits, key=lambda item: item.score, reverse=True)[:top_m_patches]
        # Weighted by rank so one noisy high patch does not dominate by itself.
        weighted = 0.0
        for idx, hit in enumerate(top_hits):
            weighted += hit.score / float(idx + 1)
        bucket, s3_key = document_meta[doc_key]
        candidates.append(
            PageResult(
                document_id=top_hits[0].document_id,
                s3_bucket=bucket,
                s3_key=s3_key,
                page_number=page_number,
                score=weighted,
            )
        )

    ranked = sorted(candidates, key=lambda item: item.score, reverse=True)
    if max_pages_per_document <= 0:
        return ranked[:top_k_pages]

    selected: list[PageResult] = []
    per_doc_counts: dict[str, int] = defaultdict(int)
    for item in ranked:
        doc_key = str(item.document_id)
        if per_doc_counts[doc_key] >= max_pages_per_document:
            continue
        selected.append(item)
        per_doc_counts[doc_key] += 1
        if len(selected) == top_k_pages:
            break

    return selected

