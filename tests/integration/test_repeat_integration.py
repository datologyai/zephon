# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for exhausted_policy='repeat' through a real Pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from zephon.api import Pipeline as PublicPipeline
from zephon.core.constants import SampleRecord
from zephon.io.dataset import Dataset
from zephon.work.base import MixtureReadConfig, MixtureReadMode
from zephon.work.static_mixture import StaticMixtureWorkSource

pytestmark = pytest.mark.integration


def _write_jsonl(root: Path, name: str, values: list[int]) -> Dataset:
    """Write a single-shard JSONL dataset and return the Dataset descriptor."""
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"text": v}) for v in values]
    (d / "data_0.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return Dataset.from_path(name, str(d), fmt="jsonl")


def _collect_values(pipe: PublicPipeline) -> list[tuple[str, int]]:
    """Drain a pipeline, returning (dataset_name, text_value) pairs."""
    id_to_name: dict[int, str] = {
        i: ds.name for i, ds in pipe.ws.datasets_by_id.items()
    }
    out: list[tuple[str, int]] = []
    for rec in pipe:
        assert isinstance(rec, SampleRecord)
        ds_id = int(rec.meta.sample_id[0])
        name = id_to_name.get(ds_id, str(ds_id))
        payload = rec.payload
        assert isinstance(payload, dict)
        val = int(payload.get("text", 0))
        out.append((name, val))
    return out


def test_repeat_produces_more_samples_than_dataset(tmp_path: Path) -> None:
    """A repeat-policy pipeline emits more samples than the underlying datasets contain."""
    # Small datasets: 10 JS samples, 10 HTML samples
    ds_js = _write_jsonl(tmp_path, "js", list(range(0, 20, 2)))
    ds_html = _write_jsonl(tmp_path, "html", list(range(1, 20, 2)))

    work = StaticMixtureWorkSource(
        [ds_js, ds_html],
        {ds_js.name: 0.6, ds_html.name: 0.4},
        chunk_size=10,
        seed=42,
        shuffle_shards=False,
        exhausted_policy="repeat",
        reshuffle_on_repeat=False,
        max_repeats=3,
    )

    pipe = PublicPipeline(work)
    pipe = pipe.decode_text()
    pipe = pipe.options(
        deterministic=True,
        runner="threads",
        max_workers=2,
        mixture_config=MixtureReadConfig(
            mode=MixtureReadMode.WEIGHTED_ROUND_ROBIN, seed=0
        ),
    )

    items = _collect_values(pipe)

    # Total unique samples = 20.  With max_repeats=3 we get multiple epochs.
    assert len(items) > 20, (
        f"Expected more samples than the dataset size (20), got {len(items)}"
    )


def test_repeat_checkpoint_restore_through_pipeline(tmp_path: Path) -> None:
    """Checkpoint/restore a repeat-policy pipeline and verify continuation matches baseline."""
    ds = _write_jsonl(tmp_path, "alpha", list(range(20)))

    def make_work() -> StaticMixtureWorkSource:
        return StaticMixtureWorkSource(
            [ds],
            {ds.name: 1.0},
            chunk_size=5,
            seed=7,
            shuffle_shards=False,
            exhausted_policy="repeat",
            reshuffle_on_repeat=False,
            max_repeats=2,
        )

    def make_pipe(work: StaticMixtureWorkSource) -> PublicPipeline:
        pipe = PublicPipeline(work)
        pipe = pipe.decode_text()
        pipe = pipe.options(
            deterministic=True,
            runner="threads",
            max_workers=2,
            mixture_config=MixtureReadConfig(
                mode=MixtureReadMode.WEIGHTED_ROUND_ROBIN, seed=0
            ),
        )
        return pipe

    # Baseline: drain fully
    baseline = _collect_values(make_pipe(make_work()))

    # Partial run, checkpoint, restore, continue
    work_save = make_work()
    pipe_save = make_pipe(work_save)
    prefix: list[tuple[str, int]] = []
    for i, rec in enumerate(pipe_save):
        assert isinstance(rec, SampleRecord)
        ds_id = int(rec.meta.sample_id[0])
        val = int(rec.payload.get("text", 0))  # type: ignore[union-attr]
        prefix.append((ds.name, val))
        if i + 1 >= len(baseline) // 2:
            break

    state = pipe_save.checkpoint()

    work_load = make_work()
    pipe_load = make_pipe(work_load)
    pipe_load.restore(state)
    suffix = _collect_values(pipe_load)

    combined = prefix + suffix
    assert combined == baseline, (
        f"prefix({len(prefix)}) + suffix({len(suffix)}) != baseline({len(baseline)})"
    )


def test_repeat_reshuffle_produces_different_order_per_epoch(tmp_path: Path) -> None:
    """With reshuffle_on_repeat, each epoch sees a different sample order through the pipeline."""
    ds = _write_jsonl(tmp_path, "alpha", list(range(30)))

    work = StaticMixtureWorkSource(
        [ds],
        {ds.name: 1.0},
        chunk_size=10,
        seed=42,
        shuffle_shards=True,
        shuffle_within_shard=True,
        exhausted_policy="repeat",
        reshuffle_on_repeat=True,
        max_repeats=2,
    )

    pipe = PublicPipeline(work)
    pipe = pipe.decode_text()
    pipe = pipe.options(
        deterministic=True,
        runner="threads",
        max_workers=2,
        mixture_config=MixtureReadConfig(
            mode=MixtureReadMode.WEIGHTED_ROUND_ROBIN, seed=0
        ),
    )

    items = _collect_values(pipe)
    n = 30  # dataset size

    # Should have multiple epochs worth of data
    assert len(items) > n

    # Extract values for each epoch
    epoch1_vals = [v for _, v in items[:n]]
    epoch2_vals = [v for _, v in items[n : 2 * n]]

    # Same values, different order
    assert sorted(epoch1_vals) == sorted(epoch2_vals)
    assert epoch1_vals != epoch2_vals
