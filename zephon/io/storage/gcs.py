"""Google Cloud Storage backend using obstore."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from .obstore import ObstoreBackend

logger = logging.getLogger(__name__)


def _get_gcs_credentials_from_env_and_files() -> dict[str, str] | None:
    """Read GCS credentials from environment variables and standard locations.

    Why we don't just use obstore's default credential chain:
    Similar to AWS, obstore tries the GCE metadata service (169.254.169.254) for
    credentials. On non-GCE machines, this causes a long timeout before falling
    back to other sources, making the first request very slow.

    By checking GOOGLE_APPLICATION_CREDENTIALS and the gcloud default credentials
    file ourselves first, we skip the metadata service probe when credentials are
    available locally. If no credentials are found, we return None and let obstore
    try its full chain (which works on GCE where the metadata service exists).

    Checks sources in order:
    1. GOOGLE_APPLICATION_CREDENTIALS environment variable (path to service account JSON)
    2. ~/.config/gcloud/application_default_credentials.json (gcloud auth default)
    3. Returns None to let obstore try its default chain (GCE metadata, etc.)
    """
    # 1. GOOGLE_APPLICATION_CREDENTIALS
    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if creds_path and Path(creds_path).exists():
        return {"google_service_account": creds_path}

    # 2. gcloud default application credentials
    gcloud_path = (
        Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
    )
    if gcloud_path.exists():
        return {"google_service_account": str(gcloud_path)}

    # 3. Let obstore try its default chain
    return None


class GCSBackend(ObstoreBackend):
    """GCS backend using obstore.

    Credentials are resolved from (in order):
    - GOOGLE_APPLICATION_CREDENTIALS environment variable
    - ~/.config/gcloud/application_default_credentials.json
    - obstore default chain (GCE metadata, etc.)

    Legacy S3-compatible mode (deprecated):
    - If GCS_KEY and GCS_SECRET are set, uses S3 protocol to storage.googleapis.com
    """

    valid_schemes = frozenset({"gs", "gcs"})

    def __init__(self) -> None:
        self._stores: dict[str, Any] = {}  # bucket -> Store
        self._base_config: dict[str, Any] | None = None
        self._use_s3_compat: bool | None = None
        self._logged_s3_compat_warning = False

    def _get_store(self, bucket: str) -> Any:
        """Get or create a store for the given bucket.

        Credentials are resolved once on first call and cached for the lifetime
        of this backend instance. Stores are also cached per bucket.
        """
        if bucket in self._stores:
            return self._stores[bucket]

        # Determine mode on first use (cached for all subsequent calls)
        if self._use_s3_compat is None:
            self._use_s3_compat = "GCS_KEY" in os.environ and "GCS_SECRET" in os.environ

        if self._use_s3_compat:
            # Deprecated S3-compatible mode
            if not self._logged_s3_compat_warning:
                logger.warning(
                    "GCS_KEY/GCS_SECRET S3-compatible mode is deprecated. "
                    "Please migrate to GOOGLE_APPLICATION_CREDENTIALS or gcloud auth."
                )
                self._logged_s3_compat_warning = True

            from obstore.store import S3Store

            store = S3Store.from_url(
                f"s3://{bucket}",
                config={
                    "aws_access_key_id": os.environ["GCS_KEY"],
                    "aws_secret_access_key": os.environ["GCS_SECRET"],
                    "aws_endpoint": "https://storage.googleapis.com",
                    "aws_region": "auto",
                },
                client_options={"timeout": "120s"},
            )
        else:
            # Native GCS mode
            from obstore.store import GCSStore

            if self._base_config is None:
                self._base_config = _get_gcs_credentials_from_env_and_files() or {}

            store = GCSStore.from_url(
                f"gs://{bucket}",
                config=self._base_config,
                client_options={"timeout": "120s"},
            )

        self._stores[bucket] = store
        return store


__all__ = ["GCSBackend"]
