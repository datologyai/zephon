from typing import Mapping

from zephon.core.constants import SampleRecord
from zephon.core.engine import Engine, RuntimeOptions
from zephon.core.graph import Graph
from zephon.core.planner import Planner
from zephon.io import InMemoryShard
from zephon.io.dataset import Dataset
from zephon.ops.delay import DelayById
from zephon.ops.fetch import FetchOp
from zephon.work.base import MixtureReadConfig, MixtureReadMode
from zephon.work.static_mixture import StaticMixtureWorkSource


def _mk_dataset(name: str, shards: Mapping[int, int]) -> Dataset:
    data: dict[int, InMemoryShard] = {}
    for sid, count in shards.items():
        rows = [{"text": f"{name}:{sid}:{i}"} for i in range(count)]
        data[int(sid)] = InMemoryShard(rows)
    return Dataset.from_dict(name, data)


def _run_engine(
    deterministic: bool, workers: int, mode: MixtureReadMode
) -> list[tuple[int, int, int]]:
    # Two small in-memory datasets with two shards each
    ds_a = _mk_dataset("A", {0: 20, 1: 20})
    ds_b = _mk_dataset("B", {0: 15, 1: 15})

    mixture = {"A": 0.6, "B": 0.4}
    work = StaticMixtureWorkSource(
        [ds_a, ds_b],
        mixture,
        chunk_size=32,
        seed=123,
        shuffle_shards=True,
        shuffle_within_shard=True,
    )

    # Build a simple graph: FetchOp -> DelayById (delay after fetch to act on SampleRecord)
    g = Graph()
    g.add("fetch", FetchOp())
    g.add("delay", DelayById(max_delay_ms=2.0), *g.nodes)

    plan = Planner().make_plan(g)

    opts = RuntimeOptions(
        deterministic=deterministic,
        max_workers=workers,
        mixture_config=MixtureReadConfig(mode=mode, seed=999),
        default_stage_prefetch=0,
    )
    eng = Engine(plan, opts, work)

    # Collect a fixed number of outputs to make the test bounded
    out_ids: list[tuple[int, int, int]] = []
    for rec in eng.build_iter():
        assert isinstance(rec, SampleRecord)
        out_ids.append(rec.meta.sample_id)
        if len(out_ids) >= 120:
            break
    eng.close()
    return out_ids


def test_engine_reproducible_wrr() -> None:
    first = _run_engine(
        deterministic=True, workers=4, mode=MixtureReadMode.WEIGHTED_ROUND_ROBIN
    )
    second = _run_engine(
        deterministic=True, workers=16, mode=MixtureReadMode.WEIGHTED_ROUND_ROBIN
    )
    assert first == second


def test_engine_reproducible_weighted_random() -> None:
    first = _run_engine(
        deterministic=True, workers=2, mode=MixtureReadMode.WEIGHTED_RANDOM
    )
    second = _run_engine(
        deterministic=True, workers=8, mode=MixtureReadMode.WEIGHTED_RANDOM
    )
    assert first == second
