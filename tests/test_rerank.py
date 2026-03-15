from __future__ import annotations

from uuid import uuid4

from visionrag.rerank import aggregate_patch_scores
from visionrag.types import PatchSearchHit


def test_aggregate_patch_scores_respects_top_m() -> None:
    doc_id = uuid4()
    hits = [
        PatchSearchHit(doc_id, "b", "k", 1, 0, {"x1": 0, "y1": 0, "x2": 0.1, "y2": 0.1}, 0.9),
        PatchSearchHit(doc_id, "b", "k", 1, 1, {"x1": 0, "y1": 0, "x2": 0.2, "y2": 0.2}, 0.8),
        PatchSearchHit(doc_id, "b", "k", 1, 2, {"x1": 0, "y1": 0, "x2": 0.3, "y2": 0.3}, 0.1),
        PatchSearchHit(doc_id, "b", "k", 2, 0, {"x1": 0, "y1": 0, "x2": 0.1, "y2": 0.1}, 0.7),
    ]
    pages = aggregate_patch_scores(hits=hits, top_k_pages=2, top_m_patches=2, max_pages_per_document=2)
    assert len(pages) == 2
    assert pages[0].page_number == 1
    assert pages[0].score > pages[1].score


def test_aggregate_patch_scores_applies_diversity_guard() -> None:
    doc_a = uuid4()
    doc_b = uuid4()
    hits = [
        PatchSearchHit(doc_a, "b", "a", 1, 0, {}, 0.95),
        PatchSearchHit(doc_a, "b", "a", 2, 0, {}, 0.94),
        PatchSearchHit(doc_b, "b", "b", 1, 0, {}, 0.93),
    ]
    pages = aggregate_patch_scores(hits=hits, top_k_pages=3, top_m_patches=1, max_pages_per_document=1)
    assert len(pages) == 2
    assert {str(item.document_id) for item in pages} == {str(doc_a), str(doc_b)}

