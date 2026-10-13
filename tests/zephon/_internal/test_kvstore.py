import time
from typing import Any

import pytest

from tests.helpers.kvstore import InMemoryKVStore
from zephon._internal import kvstore
from zephon._internal.kvstore import TorchKVStore, open_kv_store


@pytest.fixture
def torch_store() -> Any:
    dist = pytest.importorskip("torch.distributed")
    server = dist.TCPStore("127.0.0.1", 0, is_master=True, wait_for_workers=False)
    yield TorchKVStore("127.0.0.1", server.port, timeout_s=5.0)
    del server


def test_torch_store_reads_and_writes(torch_store: TorchKVStore) -> None:
    assert torch_store.get("missing") is None
    assert not torch_store.exists("missing")
    torch_store.set("a", b"\x00\xffbinary")
    assert torch_store.get("a") == b"\x00\xffbinary"
    assert torch_store.exists("a")
    torch_store.delete("a")
    assert torch_store.get("a") is None


def test_torch_store_set_if_absent_returns_stored_value(
    torch_store: TorchKVStore,
) -> None:
    assert torch_store.set_if_absent("k", b"first") == b"first"
    assert torch_store.set_if_absent("k", b"second") == b"first"
    assert torch_store.get("k") == b"first"


def test_torch_store_append_and_get_many(torch_store: TorchKVStore) -> None:
    torch_store.append("p", b"a\n")
    torch_store.append("p", b"b\n")
    torch_store.set("x", b"1")
    assert torch_store.get_many(["p", "x"]) == [b"a\nb\n", b"1"]
    # A missing key falls back to per-key reads instead of blocking.
    assert torch_store.get_many(["x", "missing"]) == [b"1", None]
    assert torch_store.get_many([]) == []


def test_torch_store_wait(torch_store: TorchKVStore) -> None:
    torch_store.set("ready", b"1")
    assert torch_store.wait(["ready"], 0)
    assert torch_store.wait(["ready"], 1e-6)
    assert not torch_store.wait(["ready", "missing"], 0)
    start = time.monotonic()
    assert not torch_store.wait(["missing"], 0.05)
    assert time.monotonic() - start < 2


def test_polling_wait_checks_readiness_without_budget() -> None:
    store = InMemoryKVStore()
    store.set("ready", b"1")
    assert store.wait(["ready"], 0)
    assert not store.wait(["ready", "missing"], 0)


def test_torch_store_uri_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MASTER_ADDR", raising=False)
    monkeypatch.delenv("MASTER_PORT", raising=False)
    with pytest.raises(ValueError, match="MASTER_ADDR"):
        open_kv_store("torch://", timeout_s=1.0)
    store = open_kv_store("torch://node-0:1234", timeout_s=7.0)
    assert isinstance(store, TorchKVStore)
    assert (store.host, store.port, store._timeout_s) == ("node-0", 1234, 7.0)

    monkeypatch.setenv("MASTER_ADDR", "leader")
    monkeypatch.setenv("MASTER_PORT", "29500")
    store = open_kv_store("torch://", timeout_s=1.0)
    assert isinstance(store, TorchKVStore)
    assert (store.host, store.port) == ("leader", 29500)


def test_open_kv_store_rejects_unknown_scheme() -> None:
    with pytest.raises(ValueError, match="registered schemes"):
        open_kv_store("nope://somewhere", timeout_s=1.0)


def test_registered_scheme_is_opened(monkeypatch: pytest.MonkeyPatch) -> None:
    store = InMemoryKVStore()
    monkeypatch.setitem(kvstore._REGISTRY, "memory", lambda _uri, _t: store)
    assert open_kv_store("MEMORY://x", timeout_s=1.0) is store


@pytest.mark.parametrize(
    "uri,error",
    [
        ("torch://host:1234/run-a", "only a host and port"),
        ("torch://user@host:1234", "only a host and port"),
        ("torch://host:1234?x=1", "only a host and port"),
        ("torch://host:0", "not a valid port"),
        ("torch://host:99999", "invalid port"),
        ("torch://host", "both a host and a port"),
        ("torch://:1234", "both a host and a port"),
    ],
)
def test_torch_store_rejects_ambiguous_uris(uri: str, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        open_kv_store(uri, timeout_s=1.0)


def test_torch_store_names_invalid_master_port(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MASTER_ADDR", "leader")
    monkeypatch.setenv("MASTER_PORT", "not-a-port")
    with pytest.raises(ValueError, match="MASTER_PORT is not a valid port"):
        open_kv_store("torch://", timeout_s=1.0)


def test_duplicate_scheme_registration_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(kvstore._REGISTRY, "memory", lambda _uri, _t: InMemoryKVStore())
    with pytest.raises(ValueError, match="already registered"):
        kvstore.register_kv_store("MEMORY", lambda _uri, _t: InMemoryKVStore())


def test_torch_store_reconnects_after_failure() -> None:
    dist = pytest.importorskip("torch.distributed")
    server = dist.TCPStore("127.0.0.1", 0, is_master=True, wait_for_workers=False)
    port = server.port
    store = TorchKVStore("127.0.0.1", port, timeout_s=1.0)
    store.set("k", b"v")
    del server
    with pytest.raises(RuntimeError, match=rf"TCPStore \w+ of 'k' at 127.0.0.1:{port}"):
        store.set("k", b"v")
        store.get("k")
    replacement = dist.TCPStore(
        "127.0.0.1", port, is_master=True, wait_for_workers=False
    )
    store.set("k", b"again")
    assert store.get("k") == b"again"
    del replacement
