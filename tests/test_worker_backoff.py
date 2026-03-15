from __future__ import annotations

from visionrag.services.worker_service import compute_backoff_delay


def test_compute_backoff_delay_increases_exponentially() -> None:
    assert compute_backoff_delay(1, 5, 300) == 5
    assert compute_backoff_delay(2, 5, 300) == 10
    assert compute_backoff_delay(3, 5, 300) == 20


def test_compute_backoff_delay_is_capped() -> None:
    assert compute_backoff_delay(10, 5, 60) == 60

