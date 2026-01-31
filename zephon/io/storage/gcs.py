"""Google Cloud Storage backend using obstore."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Mapping

from ._utils import OpenViaDownloadMixin, split_url

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


class GCSBackend(OpenViaDownloadMixin):
    """GCS backend using obstore.

    Credentials are resolved from (in order):
    - GOOGLE_APPLICATION_CREDENTIALS environment variable
    - ~/.config/gcloud/application_default_credentials.json
    - obstore default chain (GCE metadata, etc.)

    Legacy S3-compatible mode (deprecated):
    - If GCS_KEY and GCS_SECRET are set, uses S3 protocol to storage.googleapis.com
    """

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

    def exists(self, path: str) -> bool:
        """Check if a file exists at the given GCS path."""
        scheme, bucket, key = split_url(path)
        if scheme not in {"gs", "gcs"} or not bucket or not key:
            return False

        import obstore as obs

        store = self._get_store(bucket)
        try:
            obs.head(store, key)
            return True
        except Exception as e:
            err = str(e)
            if "404" in err or "NoSuchKey" in err or "NotFound" in err:
                return False
            if "403" in err or "AccessDenied" in err:
                return False
            raise

    def download(self, src: str, dst: str, timeout: float | None = None) -> None:
        """Download a file from GCS to local disk."""
        # timeout param kept for StorageBackend protocol compatibility but unused.
        # obstore only supports store-level timeout (set in _get_store via client_options).
        # See: https://developmentseed.org/obstore/latest/api/store/config/
        del timeout
        scheme, bucket, key = split_url(src)
        if scheme not in {"gs", "gcs"} or not bucket or not key:
            raise ValueError(f"Invalid GCS URL: {src}")

        import obstore as obs

        store = self._get_store(bucket)
        try:
            data = obs.get(store, key).bytes()
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "wb") as f:
                f.write(data)
        except Exception as e:
            err = str(e)
            if "403" in err or "AccessDenied" in err:
                raise FileNotFoundError(f"Access denied: {src}") from e
            if "404" in err or "NoSuchKey" in err or "NotFound" in err:
                raise FileNotFoundError(f"Object not found: {src}") from e
            raise

    def listdir(self, path: str) -> list[str]:
        """List files in a GCS directory (prefix)."""
        scheme, bucket, prefix = split_url(path)
        if scheme not in {"gs", "gcs"} or not bucket:
            raise NotADirectoryError(f"Not a GCS directory: {path}")

        import obstore as obs

        store = self._get_store(bucket)
        base = (
            (prefix + "/") if (prefix and not prefix.endswith("/")) else (prefix or "")
        )

        entries: list[str] = []
        for chunk in obs.list(store, prefix=base):
            for obj in chunk:
                obj_path = obj["path"]
                if not obj_path.startswith(base):
                    continue
                name = obj_path[len(base) :]
                # Only direct children (no nested paths)
                if name and "/" not in name:
                    entries.append(name)

        return sorted(entries)

    def stat(self, path: str) -> Mapping[str, int]:
        """Get file metadata (size) for a GCS object."""
        scheme, bucket, key = split_url(path)
        if scheme not in {"gs", "gcs"} or not bucket or not key:
            raise FileNotFoundError(f"Invalid GCS path: {path}")

        import obstore as obs

        store = self._get_store(bucket)
        try:
            meta = obs.head(store, key)
            return {"size": meta["size"]}
        except Exception as e:
            err = str(e)
            if "404" in err or "NoSuchKey" in err or "NotFound" in err:
                raise FileNotFoundError(f"Object not found: {path}") from e
            if "403" in err or "AccessDenied" in err:
                raise FileNotFoundError(f"Access denied: {path}") from e
            raise


__all__ = ["GCSBackend"]
