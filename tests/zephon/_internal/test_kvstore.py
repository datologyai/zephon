import pytest

from tests.helpers.kvstore import InMemoryKVStore
from zephon._internal import kvstore
from zephon._internal.kvstore import (
    TorchKVStore,
    open_kv_store,
    read_staged_value,
    stage_value,
)


def test_polling_wait_checks_readiness_without_budget() -> None:
    store = InMemoryKVStore()
    store.set("ready", b"1")
    assert store.wait(["ready"], 0)
    assert not store.wait(["ready", "missing"], 0)


def test_torch_store_uri_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MASTER_ADDR", raising=False)
    monkeypatch.delenv("MASTER_PORT", raising=False)
    # The environment is read on first use, not when the store is built.
    unresolved = open_kv_store("torch://", timeout_s=1.0)
    assert isinstance(unresolved, TorchKVStore)
    with pytest.raises(ValueError, match="MASTER_ADDR"):
        _ = unresolved.host
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
    store = open_kv_store("torch://", timeout_s=1.0)
    with pytest.raises(ValueError, match="MASTER_PORT is not a valid port"):
        store.check_ready()


def test_duplicate_scheme_registration_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(kvstore._REGISTRY, "memory", lambda _uri, _t: InMemoryKVStore())
    with pytest.raises(ValueError, match="already registered"):
        kvstore.register_kv_store("MEMORY", lambda _uri, _t: InMemoryKVStore())


def _small_store(max_value_bytes: int) -> InMemoryKVStore:
    store = InMemoryKVStore()
    store.max_value_bytes = max_value_bytes
    return store


def test_small_value_travels_inside_its_manifest() -> None:
    store = _small_store(1024)
    staged = stage_value(store, "k", b"small")
    assert staged.chunk_keys == []
    assert store.data == {}
    assert read_staged_value(store, "k", staged.manifest) == b"small"


def test_large_value_is_written_in_chunks_before_its_manifest() -> None:
    store = _small_store(10)
    data = bytes(range(256)) * 3
    staged = stage_value(store, "k", data)
    assert len(staged.chunk_keys) == 77
    assert all(len(store.data[key]) <= 10 for key in staged.chunk_keys)
    assert len(staged.manifest) <= 64
    assert read_staged_value(store, "k", staged.manifest) == data


def test_identical_values_get_distinct_manifests() -> None:
    # Distinct write IDs let set_if_absent tell which publisher won.
    store = _small_store(10)
    first = stage_value(store, "k", b"same bytes, twice")
    second = stage_value(store, "k", b"same bytes, twice")
    assert first.manifest != second.manifest
    assert set(first.chunk_keys).isdisjoint(second.chunk_keys)


def test_incomplete_or_altered_value_is_rejected() -> None:
    store = _small_store(10)
    staged = stage_value(store, "k", b"x" * 100)
    store.data[staged.chunk_keys[3]] = b"y" * 10
    with pytest.raises(RuntimeError, match="doesn't match its manifest"):
        read_staged_value(store, "k", staged.manifest)
    del store.data[staged.chunk_keys[0]]
    with pytest.raises(RuntimeError, match="missing 1 of 10 chunks"):
        read_staged_value(store, "k", staged.manifest)
    with pytest.raises(RuntimeError, match="no valid manifest"):
        read_staged_value(store, "k", b"not a manifest")


@pytest.mark.parametrize("op", ["set", "set_if_absent", "append"])
def test_empty_values_are_rejected(op: str) -> None:
    store = InMemoryKVStore()
    with pytest.raises(ValueError, match="values must be nonempty"):
        getattr(store, op)("k", b"")
    assert store.data == {}
