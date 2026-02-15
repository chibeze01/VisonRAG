from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name, default)
    if value is None:
        return None
    return value.strip()


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = _env(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    postgres_dsn: str
    default_s3_bucket: str | None
    aws_region: str
    model_name: str
    model_version: str
    model_device: str
    azure_openai_endpoint: str | None
    azure_openai_api_key: str | None
    azure_openai_api_version: str
    azure_openai_deployment: str | None
    gemini_api_key: str | None
    gemini_model: str
    worker_poll_interval_seconds: int
    worker_lease_seconds: int
    worker_max_attempts: int
    worker_retry_base_seconds: int
    worker_retry_max_seconds: int
    worker_lease_refresh_every_pages: int
    query_default_top_k_patches: int
    query_default_top_k_pages: int
    query_page_top_m_patches: int
    query_max_pages_per_document: int
    render_dpi: int
    log_level: str
    metrics_enabled: bool

    @classmethod
    def from_env(cls, load_env: bool = True) -> "Settings":
        if load_env:
            load_dotenv()

        postgres_dsn = _env("POSTGRES_DSN")
        if not postgres_dsn:
            raise ValueError("POSTGRES_DSN is required.")

        return cls(
            postgres_dsn=postgres_dsn,
            default_s3_bucket=_env("DEFAULT_S3_BUCKET"),
            aws_region=_env("AWS_REGION", "us-east-1") or "us-east-1",
            model_name=_env("COLPALI_MODEL_NAME", "vidore/colqwen2-v1.0") or "vidore/colqwen2-v1.0",
            model_version=_env("COLPALI_MODEL_VERSION", "v1") or "v1",
            model_device=_env("COLPALI_DEVICE", "cpu") or "cpu",
            azure_openai_endpoint=_env("AZURE_OPENAI_ENDPOINT"),
            azure_openai_api_key=_env("AZURE_OPENAI_API_KEY"),
            azure_openai_api_version=_env("AZURE_OPENAI_API_VERSION", "2024-10-21") or "2024-10-21",
            azure_openai_deployment=_env("AZURE_OPENAI_DEPLOYMENT"),
            gemini_api_key=_env("GEMINI_API_KEY"),
            gemini_model=_env("GEMINI_MODEL", "gemini-2.0-flash") or "gemini-2.0-flash",
            worker_poll_interval_seconds=_env_int("WORKER_POLL_INTERVAL_SECONDS", 3),
            worker_lease_seconds=_env_int("WORKER_LEASE_SECONDS", 120),
            worker_max_attempts=_env_int("WORKER_MAX_ATTEMPTS", 5),
            worker_retry_base_seconds=_env_int("WORKER_RETRY_BASE_SECONDS", 5),
            worker_retry_max_seconds=_env_int("WORKER_RETRY_MAX_SECONDS", 300),
            worker_lease_refresh_every_pages=_env_int("WORKER_LEASE_REFRESH_EVERY_PAGES", 5),
            query_default_top_k_patches=_env_int("QUERY_DEFAULT_TOP_K_PATCHES", 40),
            query_default_top_k_pages=_env_int("QUERY_DEFAULT_TOP_K_PAGES", 5),
            query_page_top_m_patches=_env_int("QUERY_PAGE_TOP_M_PATCHES", 5),
            query_max_pages_per_document=_env_int("QUERY_MAX_PAGES_PER_DOCUMENT", 2),
            render_dpi=_env_int("RENDER_DPI", 200),
            log_level=_env("LOG_LEVEL", "INFO") or "INFO",
            metrics_enabled=_env_bool("METRICS_ENABLED", True),
        )
