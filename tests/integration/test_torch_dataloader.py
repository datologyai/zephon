# tests/test_torch_iterable_dataloader_integration.py
#
# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

from collections import Counter

import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader

# torchdata is optional; stateful tests will skip if the class isn't importable.
try:
    import torchdata as _td  # noqa: F401

    try:
        from torchdata.stateful_dataloader import (
            StatefulDataLoader,  # type: ignore[attr-defined]
        )
    except Exception:  # pragma: no cover
        StatefulDataLoader = None  # type: ignore[assignment]
except Exception:  # pragma: no cover
    StatefulDataLoader = None  # type: ignore[assignment]

from zephon.api import Pipeline as PublicPipeline
from zephon.io import Dataset, InMemoryShard
from zephon.work.static_mixture import StaticMixtureWorkSource

# ------------------------
# Shared helpers
# ------------------------


def make_dataset(name: str, sample_count: int) -> Dataset:
    rows = [{"text": f"{name}-{i}"} for i in range(sample_count)]
    return Dataset.from_dict(name, {0: InMemoryShard(rows)})


def _extract_texts(item) -> list[str]:
    """Return flat list of 'text' payloads from a SampleRecord or SampleBatch."""
    from zephon.core.constants import SampleBatch, SampleRecord

    if isinstance(item, SampleRecord):
        payload = item.payload
        assert isinstance(payload, dict)
        return [str(payload.get("text", ""))]
    assert isinstance(item, SampleBatch)
    texts = []
    for record in item.records:
        payload = record.payload
        assert isinstance(payload, dict)
        texts.append(str(payload.get("text", "")))
    return texts


def _build_pipe(
    ds: Dataset,
    *,
    with_batch: bool,
    batch_size: int = 8,
    chunk_size: int = 16,
    stage_prefetch: int = 0,
    final_prefetch: int = 0,
    seed: int = 7,
) -> PublicPipeline:
    work = StaticMixtureWorkSource(
        [ds],
        {ds.name: 1.0},
        chunk_size=chunk_size,
        seed=seed,
        shuffle_shards=False,
        shuffle_within_shard=False,
    )
    pipe = (
        PublicPipeline(work)
        .decode_text()
        .tokenize(tokenizer_id="__fallback__", parallelism=2)
    )
    if with_batch:
        pipe = pipe.batch(batch_size, drop_last=False)

    return pipe.options(
        deterministic=True,
        canonical_replicas=1,  # single-lane; DL workers fan-in to the same pipeline
        num_ranks=1,
        physical_rank=0,
        mapping_strategy="contiguous",
        default_stage_prefetch=stage_prefetch,
        prefetch_batches=final_prefetch,
        max_workers=8,
    )


def _drain_pipe_to_texts(pipe: PublicPipeline) -> list[str]:
    out: list[str] = []
    for item in pipe:
        out.extend(_extract_texts(item))
    return out


def _dataloader_unbatched(dataset, *, num_workers: int):
    """
    Make an *unbatched* DataLoader so Zephon controls batching.
    Prefer batch_size=None when available; otherwise emulate with batch_size=1.
    """
    try:
        return DataLoader(
            dataset,
            batch_size=None,  # PyTorch (recent): yields the dataset item directly
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
        )
    except TypeError:
        # Older PyTorch: emulate "no batching"
        def _collate_one(batch):
            assert len(batch) == 1
            return batch[0]

        return DataLoader(
            dataset,
            batch_size=1,
            collate_fn=_collate_one,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
        )


def _collect_texts_from_dl(dl, *, flat_limit: int | None = None) -> list[str]:
    """
    Iterate a DataLoader whose dataset yields Zephon SampleRecord/SampleBatch.
    Flatten to texts. If flat_limit is set, stop *after* the item that crosses it.
    """
    out: list[str] = []
    for item in dl:
        texts = _extract_texts(item)
        out.extend(texts)
        if flat_limit is not None and len(out) >= flat_limit:
            break
    return out


# ------------------------
# 1) Basic equivalence across num_workers (IterableDataset)
# ------------------------


@pytest.mark.integration
@pytest.mark.parametrize("with_batch", [False, True])
def test_iterable_dataloader_num_workers_equivalence_multiset(with_batch: bool) -> None:
    """
    Using Pipeline.to_torch_dataset() as an IterableDataset, verify that
    DataLoader over num_workers ∈ {0..4} produces the *same multiset* of texts
    (order may differ). Batching stays inside Zephon.
    """
    N = 256
    ds = make_dataset("alpha", N)

    # Canonical baseline: iterate the pipeline directly (single-thread).
    baseline = _drain_pipe_to_texts(
        _build_pipe(ds, with_batch=with_batch, chunk_size=16)
    )

    # Sanity
    assert len(baseline) == N
    assert len(set(baseline)) == N  # uniqueness makes multiset checks meaningful

    # Compare each worker count to the baseline via multiset (Counter) equality.
    for nw in range(0, 5):
        pipe = _build_pipe(ds, with_batch=with_batch, chunk_size=16)
        it_ds = pipe.to_torch_dataset()
        dl = _dataloader_unbatched(it_ds, num_workers=nw)
        got = _collect_texts_from_dl(dl)

        assert len(got) == N
        # order may differ; assert no drops/dups:
        assert Counter(got) == Counter(baseline), f"mismatch at num_workers={nw}"
        if nw < 2:
            assert got == baseline, (
                "For 0 and 1 worker, ordering between baseline and dataloader should be identical."
            )


# ------------------------
# 2) Determinism: repeated runs with the same worker count match
# ------------------------


@pytest.mark.integration
@pytest.mark.parametrize("with_batch", [False, True])
@pytest.mark.parametrize("num_workers", [0, 2, 4])
def test_iterable_dataloader_repeatability_multiset(
    with_batch: bool, num_workers: int
) -> None:
    N = 200
    ds = make_dataset("beta", N)
    pipe1 = _build_pipe(ds, with_batch=with_batch, chunk_size=10)
    pipe2 = _build_pipe(ds, with_batch=with_batch, chunk_size=10)

    dl1 = _dataloader_unbatched(pipe1.to_torch_dataset(), num_workers=num_workers)
    dl2 = _dataloader_unbatched(pipe2.to_torch_dataset(), num_workers=num_workers)

    a = _collect_texts_from_dl(dl1)
    b = _collect_texts_from_dl(dl2)

    assert len(a) == len(b) == N
    assert Counter(a) == Counter(b)
    assert a == b


# ------------------------
# 3) Stateful resume (elastic continuation) with constant num_workers
# ------------------------


@pytest.mark.integration
@pytest.mark.parametrize("with_batch", [False, True])
@pytest.mark.parametrize("num_workers", [0, 2, 4])
@pytest.mark.parametrize("cut", [17, 63, 128])
def test_stateful_dataloader_resume_multiset_equivalence(
    with_batch: bool, num_workers: int, cut: int
) -> None:
    """
    End-to-end resume with torchdata.StatefulDataLoader on an IterableDataset.
    - Keep num_workers constant across the resume boundary.
    - Save both Zephon engine checkpoint and DataLoader state.
    - After resume, the combined multiset of outputs must equal a fresh run.
    """
    if StatefulDataLoader is None:
        pytest.skip("torchdata.StatefulDataLoader not available")

    sample_count = 400
    ds = make_dataset("gamma", sample_count)

    def make_pipe():
        # modest prefetch to exercise inflight records without complicating ordering
        return _build_pipe(
            ds, with_batch=with_batch, chunk_size=8, stage_prefetch=2, final_prefetch=2
        )

    # Baseline (fresh run) with the same worker count, to compare against.
    baseline_pipe = make_pipe()
    baseline_dl = StatefulDataLoader(
        baseline_pipe.to_torch_dataset(),
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        # keep DataLoader "unbatched" — each item is a Zephon record or batch
        batch_size=None
        if "batch_size" in StatefulDataLoader.__init__.__code__.co_varnames
        else 1,  # type: ignore[attr-defined]
    )
    baseline_flat = _collect_texts_from_dl(baseline_dl)
    assert len(baseline_flat) == sample_count

    # --- Phase 1: run until 'cut', capture states ---
    p1 = make_pipe()
    dl1 = StatefulDataLoader(
        p1.to_torch_dataset(),
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        batch_size=None
        if "batch_size" in StatefulDataLoader.__init__.__code__.co_varnames
        else 1,  # type: ignore[attr-defined]
    )

    prefix = _collect_texts_from_dl(dl1, flat_limit=cut)
    assert len(prefix) >= min(cut, sample_count)
    dl_state = dl1.state_dict()

    # --- Phase 2: restore both and drain the rest ---
    p2 = make_pipe()
    dl2 = StatefulDataLoader(
        p2.to_torch_dataset(),
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        batch_size=None
        if "batch_size" in StatefulDataLoader.__init__.__code__.co_varnames
        else 1,  # type: ignore[attr-defined]
    )
    dl2.load_state_dict(dl_state)

    suffix = _collect_texts_from_dl(dl2)
    resumed = prefix + suffix

    assert len(resumed) == sample_count
    assert Counter(resumed) == Counter(baseline_flat)
    assert resumed == baseline_flat


@pytest.mark.integration
@pytest.mark.parametrize("num_workers", [0, 1, 4])
def test_microbatch_shape_invariance_across_workers(num_workers: int) -> None:
    """
    Zephon performs batching; the PyTorch DataLoader stays unbatched.
    Assert that:
      - every yielded item is a SampleBatch,
      - the multiset of microbatch sizes equals a single-process baseline,
      - the final ragged microbatch (if any) is preserved.
    """
    from collections import Counter

    from zephon.core.constants import SampleBatch

    total = 224
    batch_size = 8
    chunk_size = 7
    ds = make_dataset("epsilon", total)

    def build():
        return _build_pipe(
            ds,
            with_batch=True,
            batch_size=batch_size,
            chunk_size=chunk_size,
            stage_prefetch=2,
            final_prefetch=2,
            seed=123,
        )

    # Baseline: direct pipeline iteration (single-process)
    base_sizes: list[int] = []
    for item in build():
        assert isinstance(item, SampleBatch)
        base_sizes.append(len(item.records))

    # Sanity: counts add up
    assert sum(base_sizes) == total
    # With drop_last=False, expect the last size to be (total % batch_size)
    assert base_sizes[-1] == (total % batch_size or batch_size)

    # Compare against DataLoader with various worker counts
    pipe = build()
    it_ds = pipe.to_torch_dataset()
    dl = _dataloader_unbatched(it_ds, num_workers=num_workers)

    got_sizes: list[int] = []
    for item in dl:
        assert isinstance(item, SampleBatch), (
            "DL must yield SampleBatch when Zephon batches"
        )
        got_sizes.append(len(item.records))

    # Same total and same order of batch sizes
    assert sum(got_sizes) == total
    assert Counter(got_sizes) == Counter(base_sizes), (
        f"batch-size histogram mismatch for num_workers={num_workers}"
    )
    assert got_sizes == base_sizes
