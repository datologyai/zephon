"""AWS S3-backed storage implementation."""

from __future__ import annotations

import logging
import os
from typing import Any, Mapping

from ._utils import OpenViaDownloadMixin, split_url

logger = logging.getLogger(__name__)


class S3Backend(OpenViaDownloadMixin):
    """AWS S3-backed storage implementation.

    Behavior mirrors Mosaic Streaming's S3 downloader where sensible:
    - Credentials via boto3 default chain; fallback to unsigned for public buckets.
    - Requester pays via `ZEPHON_AWS_REQUESTER_PAYS` or
      `MOSAICML_STREAMING_AWS_REQUESTER_PAYS`.
    - Optional custom endpoint via `S3_ENDPOINT_URL`.
    """

    def __init__(self) -> None:
        self._client: Any | None = None
        pays_env = (
            os.environ.get("ZEPHON_AWS_REQUESTER_PAYS")
            or os.environ.get("MOSAICML_STREAMING_AWS_REQUESTER_PAYS")
            or ""
        )
        self._requester_pays = [b.strip() for b in pays_env.split(",") if b.strip()]
        if not os.environ.get("ZEPHON_AWS_REQUESTER_PAYS") and self._requester_pays:
            logger.info(
                "Using requester-pays buckets from MOSAICML_STREAMING_AWS_REQUESTER_PAYS; set "
                + "ZEPHON_AWS_REQUESTER_PAYS to override."
            )
        self._logged_unsigned = False
        self._logged_unsigned_retry = False

    # ----------------
    # Client handling
    # ----------------
    def _ensure_client(
        self, *, timeout: float | None = None, unsigned_ok: bool = True
    ) -> None:
        if self._client is not None:
            return
        try:
            self._client = self._create_client(unsigned=False, timeout=timeout)
        except Exception as exc:
            if not unsigned_ok:
                raise
            if not self._logged_unsigned:
                logger.info(
                    "Falling back to unsigned S3 access due to %s; assuming public bucket.",
                    exc.__class__.__name__,
                )
                self._logged_unsigned = True
            self._client = self._create_client(unsigned=True, timeout=timeout)

    def _create_client(self, *, unsigned: bool, timeout: float | None) -> Any:
        from boto3.session import Session
        from botocore import UNSIGNED
        from botocore.config import Config

        cfg: dict[str, Any] = {"retries": {"mode": "adaptive"}}
        if timeout and timeout > 0:
            cfg["read_timeout"] = float(timeout)
        if unsigned:
            cfg["signature_version"] = UNSIGNED
        config = Config(**cfg)
        sess = Session()
        endpoint = os.environ.get("S3_ENDPOINT_URL")
        return sess.client("s3", config=config, endpoint_url=endpoint)

    # -------------
    # API methods
    # -------------

    def exists(self, path: str) -> bool:
        scheme, bucket, key = split_url(path)
        if scheme != "s3" or not bucket or not key:
            return False
        self._ensure_client(unsigned_ok=True)
        assert self._client is not None
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
        if scheme != "s3" or not bucket or not key:
            raise ValueError(f"Invalid S3 URL: {src}")
        self._ensure_client(timeout=timeout, unsigned_ok=True)
        assert self._client is not None
        from boto3.s3.transfer import TransferConfig
        from botocore.exceptions import ClientError

        extra_args: dict[str, Any] = {}
        if bucket in self._requester_pays:
            extra_args["RequestPayer"] = "requester"
        try:
            self._client.download_file(
                bucket,
                key,
                dst,
                ExtraArgs=extra_args or None,
                Config=TransferConfig(use_threads=False),
            )
        except ClientError as e:
            code = str(e.response.get("Error", {}).get("Code"))
            if code in {"403", "404", "NoSuchKey"}:
                raise FileNotFoundError(f"Object not found: {src}") from e
            if code == "400":
                if not self._logged_unsigned_retry:
                    logger.info(
                        "Retrying S3 download without credentials after 400 error for %s.",
                        src,
                    )
                    self._logged_unsigned_retry = True
                self._client = self._create_client(unsigned=True, timeout=timeout)
                self.download(src, dst, timeout)
                return
            raise

    def listdir(self, path: str) -> list[str]:
        scheme, bucket, prefix = split_url(path)
        if scheme != "s3" or not bucket:
            raise NotADirectoryError(f"Not an S3 directory: {path}")
        self._ensure_client(unsigned_ok=True)
        assert self._client is not None

        base = (
            (prefix + "/") if (prefix and not prefix.endswith("/")) else (prefix or "")
        )
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
        if scheme != "s3" or not bucket or not key:
            raise FileNotFoundError(f"Invalid S3 path for stat: {path}")
        self._ensure_client(unsigned_ok=True)
        assert self._client is not None
        from botocore.exceptions import ClientError

        try:
            head = self._client.head_object(Bucket=bucket, Key=key)
        except ClientError as e:
            code = str(e.response.get("Error", {}).get("Code"))
            if code in {"403", "404", "NoSuchKey"}:
                raise FileNotFoundError(f"Object not found: {path}") from e
            raise
        size = int(head.get("ContentLength", 0))
        return {"size": size}


__all__ = ["S3Backend"]
