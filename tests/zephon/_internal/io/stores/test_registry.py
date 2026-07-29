import pytest

from zephon._internal.io.stores.registry import DatasetStoreRegistry


class _DummyView:
    def open(self, shard_id: int):  # pragma: no cover - trivial protocol impl
        return shard_id


def test_dataset_store_registry_register_and_lookup() -> None:
    reg = DatasetStoreRegistry()
    v = _DummyView()
    reg.register(1, v)
    assert reg.for_dataset(1) is v
    with pytest.raises(KeyError):
        reg.for_dataset(2)
