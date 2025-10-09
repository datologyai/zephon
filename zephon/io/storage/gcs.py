"""Google Cloud Storage backend."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Mapping

from ._utils import OpenViaDownloadMixin, split_url

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - import for static analysis only
    pass  # type: ignore[attr-defined]


class GCSBackend(OpenViaDownloadMixin):
    """GCS backend with ADC or S3-compatible fallback.

    - If `GCS_KEY` and `GCS_SECRET` are set, use S3-compatible client to
      `https://storage.googleapis.com` (mirrors Mosaic).
    - Otherwise use `google-cloud-storage` with Application Default Credentials.
    """

    def __init__(self) -> None:
        self._client: Any | None = None  # Either google-cloud client or boto3 client
        self._mode: str | None = None  # "gcs" or "s3compat"
        self._logged_mode = False

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        if "GCS_KEY" in os.environ and "GCS_SECRET" in os.environ:
            from boto3.session import Session

            self._client = Session().client(
                "s3",
                region_name="auto",
                endpoint_url="https://storage.googleapis.com",
                aws_access_key_id=os.environ["GCS_KEY"],
                aws_secret_access_key=os.environ["GCS_SECRET"],
            )
            self._mode = "s3compat"
        else:
            try:
                from google.auth import default as default_auth
                from google.cloud import (
                    storage as gcs_storage,  # type: ignore[attr-defined]
                )
            except Exception as exc:
                raise ImportError(
                    "google-cloud-storage is required for GCS; install zephon[cloud-gcs]"
                ) from exc
            credentials, _ = default_auth()
            self._client = gcs_storage.Client(credentials=credentials)
            self._mode = "gcs"
        if not self._logged_mode:
            if self._mode == "s3compat":
                logger.info(
                    "Using GCS access via S3-compatible credentials from GCS_KEY/GCS_SECRET."
                )
            else:
                logger.info("Using GCS access via Application Default Credentials.")
            self._logged_mode = True

    # ``open`` is provided by OpenViaDownloadMixin using ``download``.

    def exists(self, path: str) -> bool:
        scheme, bucket, key = split_url(path)
        if scheme not in {"gs", "gcs"} or not bucket or not key:
            return False
        self._ensure_client()
        assert self._client is not None and self._mode is not None

        if self._mode == "gcs":
            bucket_obj = self._client.bucket(bucket)
            blob = bucket_obj.blob(key)
            return bool(blob.exists())
        else:
            from botocore.exceptions import ClientError

            try:
                self._client.head_object(Bucket=bucket, Key=key)
                return True
            except ClientError as e:
                code = str(e.response.get("Error", {}).get("Code"))
                if code in {"403", "404", "NoSuchKey"}:
                    return False
                raise

    def download(self, src: str, dst: str, timeout: float | None = None) -> None:
        scheme, bucket, key = split_url(src)
        if scheme not in {"gs", "gcs"} or not bucket or not key:
            raise ValueError(f"Invalid GCS URL: {src}")
        self._ensure_client()
        assert self._client is not None and self._mode is not None

        if self._mode == "gcs":
            bucket_obj = self._client.bucket(bucket)
            blob = bucket_obj.blob(key)
            blob.download_to_filename(dst)
        else:
            from boto3.s3.transfer import TransferConfig
            from botocore.exceptions import ClientError

            try:
                self._client.download_file(
                    bucket, key, dst, Config=TransferConfig(use_threads=False)
                )
            except ClientError as e:
                code = str(e.response.get("Error", {}).get("Code"))
                if code in {"403", "404", "NoSuchKey"}:
                    raise FileNotFoundError(f"Object not found: {src}") from e
                raise

    def listdir(self, path: str) -> list[str]:
        scheme, bucket, prefix = split_url(path)
        if scheme not in {"gs", "gcs"} or not bucket:
            raise NotADirectoryError(f"Not a GCS directory: {path}")
        self._ensure_client()
        assert self._client is not None and self._mode is not None

        base = (
            (prefix + "/") if (prefix and not prefix.endswith("/")) else (prefix or "")
        )

        if self._mode == "gcs":
            iterator = self._client.list_blobs(bucket, prefix=base, delimiter="/")
            names: list[str] = []
            for blob in iterator:
                name = getattr(blob, "name", "")
                if not name.startswith(base):
                    continue
                child = name[len(base) :]
                if child and "/" not in child:
                    names.append(child)
            return sorted(names)
        else:
            paginator = self._client.get_paginator("list_objects_v2")
            entries: list[str] = []
            for page in paginator.paginate(Bucket=bucket, Prefix=base, Delimiter="/"):
                for obj in page.get("Contents", []) or []:
                    key = obj.get("Key", "")
                    if not key.startswith(base):
                        continue
                    name = key[len(base) :]
                    if name and "/" not in name:
                        entries.append(name)
            return sorted(entries)

    def stat(self, path: str) -> Mapping[str, int]:
        scheme, bucket, key = split_url(path)
        if scheme not in {"gs", "gcs"} or not bucket or not key:
            raise FileNotFoundError(f"Invalid GCS path for stat: {path}")
        self._ensure_client()
        assert self._client is not None and self._mode is not None

        if self._mode == "gcs":
            bucket_obj = self._client.bucket(bucket)
            blob = bucket_obj.get_blob(key)
            if blob is None:
                raise FileNotFoundError(f"Object not found: {path}")
            return {"size": int(blob.size or 0)}
        else:
            from botocore.exceptions import ClientError

            try:
                head = self._client.head_object(Bucket=bucket, Key=key)
            except ClientError as e:
                code = str(e.response.get("Error", {}).get("Code"))
                if code in {"403", "404", "NoSuchKey"}:
                    raise FileNotFoundError(f"Object not found: {path}") from e
                raise
            return {"size": int(head.get("ContentLength", 0))}


__all__ = ["GCSBackend"]
