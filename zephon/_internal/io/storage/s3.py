"""AWS S3-backed storage implementation using obstore."""

from __future__ import annotations

import configparser
import logging
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ._utils import split_url
from .obstore import ObstoreBackend

logger = logging.getLogger(__name__)


def _get_aws_region_from_env_and_files(profile: str = "default") -> str | None:
    """Get AWS region from environment variables and config files.

    Checks in order:
    1. AWS_REGION / AWS_DEFAULT_REGION environment variables
    2. ~/.aws/config file
    """
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if region:
        return region

    config_path = Path.home() / ".aws" / "config"
    if config_path.exists():
        parser = configparser.ConfigParser()
        try:
            parser.read(config_path)
            section = profile if profile == "default" else f"profile {profile}"
            if section in parser and "region" in parser[section]:
                return parser[section]["region"]
        except Exception as e:
            logger.debug("Failed to read AWS config file %s: %s", config_path, e)

    return None


def _client_options(endpoint: str | None) -> dict[str, Any]:
    """Build obstore client options for an S3 store.

    obstore rejects ``http://`` endpoints unless ``allow_http`` is set in the
    client options, and it does not read ``AWS_ALLOW_HTTP`` from the
    environment. Enable it for plain-HTTP endpoints (MinIO, moto, LocalStack,
    in-cluster S3) and when ``AWS_ALLOW_HTTP`` is truthy.
    """
    options: dict[str, Any] = {"timeout": "120s"}
    allow_http_env = os.environ.get("AWS_ALLOW_HTTP", "").strip().lower()
    if (endpoint and endpoint.strip().lower().startswith("http://")) or (
        allow_http_env in ("1", "true", "yes")
    ):
        options["allow_http"] = True
    return options


def _get_aws_credentials_from_env_and_files() -> dict[str, str] | None:
    """Read AWS credentials from environment variables and standard AWS files.

    Why we don't just use obstore's default credential chain:
    obstore (via the Rust object_store crate) tries multiple credential sources
    including the EC2 Instance Metadata Service (IMDS) at 169.254.169.254. On
    non-EC2 machines, this results in a ~20+ second timeout before falling back
    to other sources, making the first request extremely slow.

    By checking env vars and ~/.aws/credentials ourselves first, we can skip
    the IMDS probe entirely when credentials are available locally. If no
    credentials are found, we return None and let obstore try its full chain
    (which will work on EC2 where IMDS is actually available).

    Checks sources in this order (matching boto3 behavior):
    1. Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    2. ~/.aws/credentials file (respects AWS_PROFILE)
    3. Returns None to let obstore try its default chain (IMDS on EC2, etc.)
    """
    profile = os.environ.get("AWS_PROFILE", "default")

    # 1. Environment variables
    access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    if access_key and secret_key:
        config: dict[str, str] = {
            "aws_access_key_id": access_key,
            "aws_secret_access_key": secret_key,
        }
        session_token = os.environ.get("AWS_SESSION_TOKEN")
        if session_token:
            config["aws_session_token"] = session_token
        region = _get_aws_region_from_env_and_files(profile)
        if region:
            config["aws_region"] = region
        return config

    # 2. ~/.aws/credentials file
    credentials_path = Path.home() / ".aws" / "credentials"
    if credentials_path.exists():
        parser = configparser.ConfigParser()
        try:
            parser.read(credentials_path)
            if profile in parser:
                section = parser[profile]
                if (
                    "aws_access_key_id" in section
                    and "aws_secret_access_key" in section
                ):
                    creds: dict[str, str] = {
                        "aws_access_key_id": section["aws_access_key_id"],
                        "aws_secret_access_key": section["aws_secret_access_key"],
                    }
                    if "aws_session_token" in section:
                        creds["aws_session_token"] = section["aws_session_token"]
                    region = _get_aws_region_from_env_and_files(profile)
                    if region:
                        creds["aws_region"] = region
                    return creds
        except Exception as e:
            logger.debug(
                "Failed to read AWS credentials file %s: %s", credentials_path, e
            )

    # 3. Let obstore try its default chain
    return None


class S3Backend(ObstoreBackend):
    """AWS S3-backed storage implementation using obstore.

    Credentials are resolved from (in order):
    - Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    - ~/.aws/credentials file (respects AWS_PROFILE)
    - obstore default chain (IMDS on EC2, etc.)

    Features:
    - Requester pays via ZEPHON_AWS_REQUESTER_PAYS or MOSAICML_STREAMING_AWS_REQUESTER_PAYS
    - Custom endpoint via S3_ENDPOINT_URL
    """

    valid_schemes = frozenset({"s3"})

    def __init__(self) -> None:
        super().__init__()
        self._stores: dict[str, Any] = {}  # bucket -> S3Store
        self._base_config: dict[str, Any] | None = None
        self._unsigned_stores: dict[str, Any] = {}  # bucket -> unsigned S3Store
        self._logged_unsigned_fallback = False

        pays_env = (
            os.environ.get("ZEPHON_AWS_REQUESTER_PAYS")
            or os.environ.get("MOSAICML_STREAMING_AWS_REQUESTER_PAYS")
            or ""
        )
        self._requester_pays = [b.strip() for b in pays_env.split(",") if b.strip()]
        if not os.environ.get("ZEPHON_AWS_REQUESTER_PAYS") and self._requester_pays:
            logger.info(
                "Using requester-pays buckets from MOSAICML_STREAMING_AWS_REQUESTER_PAYS; "
                "set ZEPHON_AWS_REQUESTER_PAYS to override."
            )

    def _get_store(self, bucket: str) -> Any:
        """Get or create an S3Store for the given bucket.

        Credentials are resolved once on first call and cached for the lifetime
        of this backend instance. Stores are also cached per bucket.
        """
        if bucket in self._stores:
            return self._stores[bucket]

        from obstore.store import S3Store

        # Build config on first use (cached for all subsequent calls)
        if self._base_config is None:
            creds = _get_aws_credentials_from_env_and_files()
            if creds:
                self._base_config = creds
            else:
                # No local credentials found - let obstore try its default chain
                # (includes IMDS for EC2). If that fails, we'll fall back to
                # unsigned access on auth errors (see _try_with_unsigned_fallback).
                region = _get_aws_region_from_env_and_files()
                self._base_config = {"aws_region": region} if region else {}

        # Copy base config and add bucket-specific settings
        config = dict(self._base_config)

        # Custom endpoint
        endpoint = os.environ.get("S3_ENDPOINT_URL")
        if endpoint:
            config["aws_endpoint"] = endpoint

        # Requester pays
        if bucket in self._requester_pays:
            config["request_payer"] = True

        store = S3Store.from_url(
            f"s3://{bucket}", config=config, client_options=_client_options(endpoint)
        )
        self._stores[bucket] = store
        return store

    def _get_unsigned_store(self, bucket: str) -> Any:
        """Get or create an unsigned S3Store for public bucket access."""
        if bucket in self._unsigned_stores:
            return self._unsigned_stores[bucket]

        from obstore.store import S3Store

        config: dict[str, Any] = {"skip_signature": True}
        endpoint = os.environ.get("S3_ENDPOINT_URL")
        if endpoint:
            config["aws_endpoint"] = endpoint
        region = _get_aws_region_from_env_and_files()
        if region:
            config["aws_region"] = region

        store = S3Store.from_url(
            f"s3://{bucket}", config=config, client_options=_client_options(endpoint)
        )
        self._unsigned_stores[bucket] = store
        return store

    def _should_retry_unsigned(self, error: str) -> bool:
        """Check if error suggests we should retry with unsigned access."""
        # 400: Some public buckets reject signed requests
        # 403 + "InvalidAccessKeyId": Credentials are invalid/not found
        # "NoCredentialProviders": obstore couldn't find any credentials
        return (
            "400" in error
            or ("403" in error and "InvalidAccessKeyId" in error)
            or "NoCredentialProviders" in error
            or "credential" in error.lower()
        )

    def download(self, src: str, dst: str, timeout: float | None = None) -> None:
        """Download a file from S3 to local disk."""
        # timeout param kept for StorageBackend protocol compatibility but unused.
        # obstore only supports store-level timeout (set in _get_store via client_options).
        # See: https://developmentseed.org/obstore/latest/api/store/config/
        del timeout
        scheme, bucket, key = split_url(src)
        if scheme != "s3" or not bucket or not key:
            raise ValueError(f"Invalid S3 URL: {src}")

        store = self._get_store(bucket)
        try:
            self._download_with_store(store, key, src, dst)
        except Exception as e:
            err = str(e)
            # Retry with unsigned access for credential-related errors
            if self._should_retry_unsigned(err):
                if not self._logged_unsigned_fallback:
                    logger.info(
                        "Retrying S3 download with unsigned access for %s "
                        "(assuming public bucket).",
                        src,
                    )
                    self._logged_unsigned_fallback = True
                unsigned_store = self._get_unsigned_store(bucket)
                self._download_with_store(unsigned_store, key, src, dst)
                return
            raise

    def walk(self, path: str) -> Iterator[tuple[str, int]]:
        """Recursively walk an S3 prefix with the same unsigned-bucket fallback as download."""
        scheme, bucket, prefix = split_url(path)
        if scheme != "s3" or not bucket:
            raise ValueError(f"Invalid S3 URL: {path}")

        store = self._get_store(bucket)
        base = prefix if (not prefix or prefix.endswith("/")) else prefix + "/"
        try:
            yield from self._walk_with_store(store, base)
        except Exception as e:
            err = str(e)
            if self._should_retry_unsigned(err):
                if not self._logged_unsigned_fallback:
                    logger.info(
                        "Retrying S3 listing with unsigned access for %s "
                        "(assuming public bucket).",
                        path,
                    )
                    self._logged_unsigned_fallback = True
                unsigned_store = self._get_unsigned_store(bucket)
                yield from self._walk_with_store(unsigned_store, base)
                return
            raise


__all__ = ["S3Backend"]
