from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class S3ObjectInfo:
    etag: str | None
    content_length: int | None


class S3Client:
    def __init__(self, region_name: str) -> None:
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError("boto3 is required for S3 access.") from exc
        self._client = boto3.client("s3", region_name=region_name)

    def fetch_pdf_bytes(self, bucket: str, key: str) -> bytes:
        response = self._client.get_object(Bucket=bucket, Key=key)
        return response["Body"].read()

    def head_object(self, bucket: str, key: str) -> S3ObjectInfo:
        response = self._client.head_object(Bucket=bucket, Key=key)
        etag = response.get("ETag")
        if isinstance(etag, str):
            etag = etag.strip('"')
        return S3ObjectInfo(etag=etag, content_length=response.get("ContentLength"))

    def healthcheck(self, bucket: str | None = None) -> bool:
        try:
            if bucket:
                self._client.head_bucket(Bucket=bucket)
            return True
        except Exception:
            return False

