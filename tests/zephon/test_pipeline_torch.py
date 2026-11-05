import sys
from importlib.machinery import ModuleSpec
from types import ModuleType
from typing import Any

import pytest

from tests.helpers.work import FakeIndexableWorkSource, make_inmem_dataset
from zephon import Pipeline as PublicPipeline
from zephon.types import SampleRecord


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


def _make_six_record_pipe(*, mtp: bool) -> PublicPipeline:
    rows = [{"text": f"s{i}"} for i in range(6)]
    ws = FakeIndexableWorkSource(make_inmem_dataset("tiny", rows), chunk_size=3)
    return (
        PublicPipeline(ws)
        .decode_text()
        .options(
            deterministic=True,
            max_workers=1,
            default_stage_prefetch=0,
            mtp_mode=mtp,
        )
    )


@pytest.mark.parametrize("mtp", [False, True])
def test_stateful_torch_dataset_state_dict_after_exhaustion(
    monkeypatch: pytest.MonkeyPatch, mtp: bool
) -> None:
    _install_torch_stubs(monkeypatch)

    pipe = _make_six_record_pipe(mtp=mtp)
    torch_ds = pipe.to_torch_dataset(stateful=True)
    assert len(list(iter(torch_ds))) == 6

    # An end-of-epoch checkpoint must carry the final position, not None.
    state = torch_ds.state_dict()
    assert state["engine"] is not None
    assert state["engine"]["progress"] == pipe.checkpoint()["progress"]

    # Resuming from it yields nothing more instead of replaying the epoch.
    resumed = _make_six_record_pipe(mtp=mtp).to_torch_dataset(stateful=True)
    resumed.load_state_dict(state)
    assert list(iter(resumed)) == []


@pytest.mark.parametrize("mtp", [False, True])
def test_stateful_torch_dataset_resave_loaded_state_after_exhaustion(
    monkeypatch: pytest.MonkeyPatch, mtp: bool
) -> None:
    _install_torch_stubs(monkeypatch)

    torch_ds = _make_six_record_pipe(mtp=mtp).to_torch_dataset(stateful=True)
    it = iter(torch_ds)
    next(it)
    next(it)
    saved_after_two = torch_ds.state_dict()
    assert len(list(it)) == 4

    # Loading a checkpoint into an exhausted dataset and saving again before
    # iterating must return the loaded position, not the previous iteration's
    # end-of-stream state.
    torch_ds.load_state_dict(saved_after_two)
    resaved = torch_ds.state_dict()
    assert resaved == saved_after_two

    resumed = _make_six_record_pipe(mtp=mtp).to_torch_dataset(stateful=True)
    resumed.load_state_dict(resaved)
    assert len(list(iter(resumed))) == 4


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
        .tokenize(tokenizer_id="__fallback__", field="text")
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
