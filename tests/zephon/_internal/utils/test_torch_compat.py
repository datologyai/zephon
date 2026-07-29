# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader, IterableDataset


class _ProbeDataset(IterableDataset):
    """Yields the detected loader kind from inside the worker."""

    def __init__(self, tag: str):
        self._tag = tag

    def __iter__(self):
        # Import locally so the class remains easily picklable under spawn.
        from zephon._internal.utils.torch_compat import detect_loader_kind

        # Return a simple tuple to avoid collate/batching surprises.
        yield (detect_loader_kind(), self._tag)


@pytest.mark.parametrize("num_workers", [0, 1])
def test_detect_loader_kind_in_vanilla_dataloader_worker(num_workers: int):
    """Should report 'vanilla' when running under torch.utils.data.DataLoader workers."""
    ds = _ProbeDataset(tag="vanilla")
    # batch_size=None -> no batching; persistent_workers=False for clean teardown
    dl = DataLoader(
        ds,
        batch_size=None,
        num_workers=num_workers,
        persistent_workers=False,
        **({} if num_workers == 0 else {"prefetch_factor": 1}),
    )
    kind, tag = next(iter(dl))
    assert tag == "vanilla"
    assert kind == "vanilla"


@pytest.mark.parametrize("num_workers", [0, 1])
def test_detect_loader_kind_in_torchdata_stateful_worker(num_workers: int):
    """
    Should report 'torchdata' when running under torchdata StatefulDataLoader workers.
    Skips cleanly if torchdata isn't installed.
    """
    torchdata = pytest.importorskip("torchdata.stateful_dataloader")
    StatefulDataLoader = torchdata.StatefulDataLoader  # type: ignore[attr-defined]

    ds = _ProbeDataset(tag="torchdata")
    sdl = StatefulDataLoader(
        ds,
        batch_size=None,
        num_workers=num_workers,
        persistent_workers=False,
        in_order=True,  # keeps behavior deterministic; not strictly required
    )
    kind, tag = next(iter(sdl))
    assert tag == "torchdata"
    assert kind == "torchdata"


def test_detect_loader_kind_unknown_without_any_loader():
    # Direct call outside of any DataLoader context should be 'unknown'
    from zephon._internal.utils.torch_compat import detect_loader_kind

    assert detect_loader_kind() == "unknown"
