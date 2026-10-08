"""Direct shared allocation and its backpressure behavior."""

import errno
from typing import Any

import pytest

from zephon._internal.utils import shm_coalesce

torch = pytest.importorskip("torch")


@pytest.mark.parametrize(
    "strategy", sorted(torch.multiprocessing.get_all_sharing_strategies())
)
@pytest.mark.parametrize(
    "dtype", [torch.uint8, torch.int64, torch.float64, torch.bfloat16]
)
def test_direct_shared_allocation(
    monkeypatch: pytest.MonkeyPatch, strategy: str, dtype: Any
) -> None:
    def reject_private_storage_copy(*args: object, **kwargs: object) -> None:
        pytest.fail("Shared allocation must not copy a private tensor's storage")

    monkeypatch.setattr(torch.Tensor, "share_memory_", reject_private_storage_copy)
    previous = torch.multiprocessing.get_sharing_strategy()
    try:
        torch.multiprocessing.set_sharing_strategy(strategy)
        buf = shm_coalesce._alloc_shm_buffer(37, dtype, "test")
        assert buf.device.type == "cpu"
        assert buf.dtype == dtype
        assert buf.shape == (37,)
        assert buf.is_shared()
        assert buf.untyped_storage().nbytes() == 37 * buf.element_size()
        buf.fill_(3)
        assert buf.tolist() == [3] * 37
    finally:
        torch.multiprocessing.set_sharing_strategy(previous)


def test_direct_shared_allocation_retries_enospc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allocate = torch.UntypedStorage._new_shared
    attempts = 0
    waits: list[str] = []

    def transient_failure(size: int, *, device: str) -> object:
        nonlocal attempts
        attempts += 1
        if attempts <= 2:
            raise OSError(errno.ENOSPC, "No space left on device")
        return allocate(size, device=device)

    monkeypatch.setattr(torch.UntypedStorage, "_new_shared", transient_failure)
    monkeypatch.setattr(shm_coalesce, "wait_for_shm_space", waits.append)
    buf = shm_coalesce._alloc_shm_buffer(8, torch.int64, "coalesce[test]")
    assert attempts == 3
    assert waits == ["coalesce[test]"] * 2
    assert buf.is_shared()
    buf.copy_(torch.arange(8))
    assert buf.tolist() == list(range(8))


def test_direct_shared_allocation_propagates_other_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("unexpected allocation error")

    def reject_wait(*args: object, **kwargs: object) -> None:
        pytest.fail("Only shared-memory exhaustion should be retried")

    monkeypatch.setattr(torch.UntypedStorage, "_new_shared", fail)
    monkeypatch.setattr(shm_coalesce, "wait_for_shm_space", reject_wait)
    with pytest.raises(RuntimeError, match="unexpected allocation error"):
        shm_coalesce._alloc_shm_buffer(8, torch.int64, "test")
