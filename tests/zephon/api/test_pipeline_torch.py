import sys
from importlib.machinery import ModuleSpec
from types import ModuleType
from typing import Any

import pytest

from tests.helpers.work import FakeIndexableWorkSource, make_inmem_dataset
from zephon.api import Pipeline as PublicPipeline
from zephon.core.constants import SampleRecord


def _install_torch_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    torch_mod = ModuleType("torch")
    utils_mod = ModuleType("torch.utils")
    data_mod = ModuleType("torch.utils.data")

    class IterableDataset: ...

    class Dataset: ...

    data_mod.IterableDataset = IterableDataset
    data_mod.Dataset = Dataset
    torch_mod.utils = utils_mod
    utils_mod.data = data_mod

    # minimal specs so importlib.util.find_spec doesn't raise
    torch_mod.__spec__ = ModuleSpec("torch", loader=None)
    utils_mod.__spec__ = ModuleSpec("torch.utils", loader=None)
    data_mod.__spec__ = ModuleSpec("torch.utils.data", loader=None)

    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "torch.utils", utils_mod)
    monkeypatch.setitem(sys.modules, "torch.utils.data", data_mod)


def test_to_torch_dataset_iterates_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_torch_stubs(monkeypatch)
    rows = [{"text": f"s{i}"} for i in range(3)]
    ds = make_inmem_dataset("tiny", rows)
    ws = FakeIndexableWorkSource(ds, chunk_size=2)
    pipe = (
        PublicPipeline(ws)
        .decode_text()
        .options(deterministic=True, max_workers=1, default_stage_prefetch=0)
    )

    torch_ds = pipe.to_torch_dataset()
    out = list(iter(torch_ds))
    # Expect three SampleRecord items
    assert len(out) == 3
    assert all(isinstance(x, SampleRecord) for x in out)


def test_to_indexable_torch_dataset_with_indexable_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_torch_stubs(monkeypatch)
    # Build an indexable pipeline: fetch -> decode_text -> tokenize (no batch/materialize)
    rows = [{"text": f"sample {i}"} for i in range(4)]
    ds = make_inmem_dataset("tiny", rows)
    ws = FakeIndexableWorkSource(ds, chunk_size=2)
    pipe = (
        PublicPipeline(ws)
        .decode_text()
        .tokenize(tokenizer_id="__fallback__")
        .options(deterministic=True, max_workers=1, default_stage_prefetch=0)
    )

    assert pipe.is_indexable
    torch_ds = pipe.to_indexable_torch_dataset()
    assert len(torch_ds) == len(ws)
    item0 = torch_ds[0]
    assert isinstance(item0, SampleRecord)
    assert isinstance(_payload_dict(item0).get("input_ids"), list)


def test_to_indexable_raises_when_not_indexable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_torch_stubs(monkeypatch)
    rows = [{"text": f"s{i}"} for i in range(3)]
    ds = make_inmem_dataset("tiny", rows)
    ws = FakeIndexableWorkSource(ds, chunk_size=2)
    pipe = PublicPipeline(ws).decode_text().batch(2)
    assert not pipe.is_indexable
    with pytest.raises(RuntimeError):
        _ = pipe.to_indexable_torch_dataset()


def _payload_dict(record: SampleRecord) -> dict[str, Any]:
    payload = record.payload
    assert isinstance(payload, dict)
    return payload
