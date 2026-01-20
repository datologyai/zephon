# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0


import pytest

from zephon.core.constants import SampleMeta, SampleRecord
from zephon.ops.delay import DelayById


def _rec(local_id: int) -> SampleRecord:
    meta = SampleMeta(sample_id=(0, 0, local_id), lane_id=0, chunk_id=0)
    return SampleRecord(meta=meta, payload={"v": local_id})


def _expected_delay_ms(local_id: int, max_delay_ms: float, slots: int = 5) -> float:
    bucket = abs(int(local_id)) % slots
    return max_delay_ms * (bucket / max(1, slots - 1))


def test_delaybyid_process_one_equivalence(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[float] = []

    def fake_sleep(secs: float) -> None:
        called.append(secs)

    monkeypatch.setattr("time.sleep", fake_sleep)
    op = DelayById(max_delay_ms=2.0)
    single = op.process_one(_rec(0))
    many = op.process_many([_rec(0)])
    assert single[0].payload == many[0].payload
    # delay for id=0 bucket should be 0 and may not call sleep
    assert all(secs >= 0.0 for secs in called)


def test_delaybyid_records_sleep_called_with_expected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[float] = []

    def fake_sleep(secs: float) -> None:
        captured.append(secs)

    monkeypatch.setattr("time.sleep", fake_sleep)

    max_ms = 10.0
    op = DelayById(max_delay_ms=max_ms)
    # Choose ids that map to different buckets modulo 5
    local_ids = [1, 2, 3, 4]
    _ = op.process_many([_rec(i) for i in local_ids])
    # Non-zero buckets must produce sleeps > 0
    assert len(captured) == len(local_ids)
    for idx, lid in enumerate(local_ids):
        exp = _expected_delay_ms(lid, max_ms) / 1000.0
        assert pytest.approx(captured[idx], rel=1e-6, abs=1e-6) == exp


def test_delaybyid_non_record_items_use_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[float] = []

    def fake_sleep(secs: float) -> None:
        captured.append(secs)

    monkeypatch.setattr("time.sleep", fake_sleep)
    max_ms = 7.0
    op = DelayById(max_delay_ms=max_ms)
    data = [0, 1, 2, 3, 4]
    _ = op.process_many(data)
    # For integers, hash is stable = value; zero bucket yields no sleep call
    expected = []
    for v in data:
        bucket = abs(int(hash(v))) % 5
        exp = (max_ms * (bucket / 4)) / 1000.0
        expected.append(exp)
    # Only non-zero delays trigger time.sleep
    non_zero = [x for x in expected if x > 0]
    assert len(captured) == len(non_zero)
    j = 0
    for exp in expected:
        if exp == 0:
            continue
        assert pytest.approx(captured[j], rel=1e-6, abs=1e-6) == exp
        j += 1
    assert j == len(captured)


def test_delaybyid_traits_and_accumulator() -> None:
    op = DelayById(max_delay_ms=1.0, max_batch=16, max_latency_ms=50)
    t = op.traits()

    assert t.indexable is True and t.parallelism == 8

    # Test deterministic mode disables time-based flushing
    acc_det = op.accumulator(deterministic=True)
    assert acc_det._max_batch == 16
    assert acc_det._max_latency_ms is None

    # Test non-deterministic mode preserves latency config
    acc_nondet = op.accumulator(deterministic=False)
    assert acc_nondet._max_batch == 16
    assert acc_nondet._max_latency_ms == 50
