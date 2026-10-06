"""Azure Blob Storage / ADLS Gen2 backend using obstore."""

from __future__ import annotations

import os
from typing import Any

from ._utils import split_url
from .obstore import ObstoreBackend


class AzureBackend(ObstoreBackend):
    """Azure backend using obstore.

    Accepts ``az://<container>/<path>`` (also ``azure://``, ``abfs://`` and
    ``abfss://``), with the storage account taken from
    ``AZURE_STORAGE_ACCOUNT_NAME``, and the fully qualified
    ``abfs[s]://<container>@<account>.dfs.core.windows.net/<path>`` form.

    Credentials come from obstore's Azure chain, configured through the
    standard ``AZURE_*`` environment variables (account key, SAS token,
    service principal, workload identity, ...), falling back to managed
    identity.
    """

    valid_schemes = frozenset({"az", "azure", "abfs", "abfss"})

    def __init__(self) -> None:
        super().__init__()
        self._stores: dict[str, Any] = {}  # container (or container@host) -> Store

    def canonical_root(self, path: str, fmt: str | None = None) -> str:
        """Pin a bare-container URL to the account it resolves to in this process.

        ``az://<container>/...`` names different data under different
        ``AZURE_STORAGE_ACCOUNT_NAME`` values; the fully qualified form does not.
        """
        del fmt
        scheme, bucket, key = split_url(path)
        account = os.environ.get("AZURE_STORAGE_ACCOUNT_NAME")
        if scheme not in self.valid_schemes or not bucket or "@" in bucket:
            return path  # Not ours, or already names its account.
        if not account:
            return path  # The store will fail with a clearer error.
        return f"abfss://{bucket}@{account}.dfs.core.windows.net/{key}".rstrip("/")

    def _get_store(self, bucket: str) -> Any:
        """Get or create an AzureStore for ``bucket``, cached per container."""
        if bucket in self._stores:
            return self._stores[bucket]

        from obstore.store import AzureStore

        # ``container@account.<host>`` only parses under the abfs schemes;
        # a bare container parses under any of them.
        scheme = "abfss" if "@" in bucket else "az"
        store = AzureStore.from_url(
            f"{scheme}://{bucket}", client_options={"timeout": "120s"}
        )
        self._stores[bucket] = store
        return store


__all__ = ["AzureBackend"]
