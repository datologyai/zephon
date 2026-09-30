"""Tests for AzureBackend using obstore."""

from pathlib import Path

import pytest

from tests.helpers.storage import _install_obstore_stubs
from zephon._internal.io.storage.azure import AzureBackend


def test_azure_download_stat_exists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state = _install_obstore_stubs(monkeypatch)
    state["objects"][("cont", "folder/file.bin")] = b"payload"

    backend = AzureBackend()

    out = tmp_path / "file.bin"
    backend.download("az://cont/folder/file.bin", str(out))
    assert out.read_bytes() == b"payload"
    assert backend.exists("az://cont/folder/file.bin") is True
    assert backend.exists("az://cont/folder/missing.bin") is False
    assert backend.stat("az://cont/folder/file.bin")["size"] == 7
    assert state["store_type"] == ["azure"]


def test_azure_store_cached_per_container(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _install_obstore_stubs(monkeypatch)
    state["objects"][("cont", "a")] = b"x"

    backend = AzureBackend()
    # Every short scheme names the same container, so they share one store.
    for scheme in ("az", "azure", "abfs", "abfss"):
        assert backend.exists(f"{scheme}://cont/a") is True
    backend.exists("az://other/a")

    assert state["store_type"] == ["azure", "azure"]


def test_azure_fully_qualified_url(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _install_obstore_stubs(monkeypatch)
    netloc = "cont@acct.dfs.core.windows.net"
    state["objects"][(netloc, "prefix/a.jsonl")] = b"{}"
    state["objects"][(netloc, "prefix/sub/b.jsonl")] = b"{}"

    backend = AzureBackend()
    store = backend._get_store(netloc)

    assert store._url == f"abfss://{netloc}"
    assert backend.listdir(f"abfss://{netloc}/prefix") == ["a.jsonl"]
    assert backend.glob(f"abfss://{netloc}/prefix/a*") == [
        f"abfss://{netloc}/prefix/a.jsonl"
    ]


def test_azure_put_read_range_delete(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _install_obstore_stubs(monkeypatch)

    backend = AzureBackend()
    backend.put("az://cont/ckpt/state.bin", b"0123456789")

    assert state["objects"][("cont", "ckpt/state.bin")] == b"0123456789"
    assert bytes(backend.read_range("az://cont/ckpt/state.bin", 2, length=3)) == b"234"

    backend.delete("az://cont/ckpt/state.bin")
    assert ("cont", "ckpt/state.bin") not in state["objects"]


def test_azure_rejects_foreign_scheme() -> None:
    with pytest.raises(ValueError, match="Invalid URL"):
        AzureBackend().download("s3://bucket/key", "/unused")


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("az://cont/data", "abfss://cont@acct.dfs.core.windows.net/data"),
        ("abfs://cont/data/", "abfss://cont@acct.dfs.core.windows.net/data"),
        ("azure://cont", "abfss://cont@acct.dfs.core.windows.net"),
        (
            "abfss://cont@other.dfs.core.windows.net/data",
            "abfss://cont@other.dfs.core.windows.net/data",
        ),
    ],
)
def test_azure_canonical_root_names_account(
    monkeypatch: pytest.MonkeyPatch, path: str, expected: str
) -> None:
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_NAME", "acct")
    assert AzureBackend().canonical_root(path) == expected


def test_azure_canonical_root_without_account_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AZURE_STORAGE_ACCOUNT_NAME", raising=False)
    assert AzureBackend().canonical_root("az://cont/data") == "az://cont/data"
