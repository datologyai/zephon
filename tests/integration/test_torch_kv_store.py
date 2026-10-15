"""TorchKVStore against real in-process TCPStore servers."""

import socket
import time
from typing import Any

import pytest

from zephon._internal.kvstore import TorchKVStore, read_staged_value, stage_value

pytestmark = pytest.mark.integration


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


def test_torch_store_moves_values_above_its_write_limit(
    torch_store: TorchKVStore,
) -> None:
    data = bytes(range(256)) * (20 * 1024 * 1024 // 256)
    staged = stage_value(torch_store, "big", data)
    assert len(staged.chunk_keys) == 5
    torch_store.set("big", staged.manifest)
    manifest = torch_store.get("big")
    assert manifest is not None
    assert read_staged_value(torch_store, "big", manifest) == data


def test_torch_store_readiness_names_the_endpoint() -> None:
    pytest.importorskip("torch.distributed")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    store = TorchKVStore("127.0.0.1", port, timeout_s=1.0)
    with pytest.raises(RuntimeError, match=f"127.0.0.1:{port}.*init_process_group"):
        store.check_ready()


def test_torch_store_readiness_passes_for_a_running_store(
    torch_store: TorchKVStore,
) -> None:
    torch_store.check_ready()
