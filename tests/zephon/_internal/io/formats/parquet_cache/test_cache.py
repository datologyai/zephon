# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import multiprocessing
import threading
from pathlib import Path

import pyarrow as pa
import pytest

from tests._catalog_helpers import catalog_set_from_locators
from zephon._internal.io.formats import ensure_builtin_formats
from zephon._internal.io.formats.parquet_cache import cache as parquet_rg_cache_module
from zephon._internal.io.formats.parquet_cache.admission import RGPublicationResult
from zephon._internal.io.formats.parquet_cache.cache import ParquetRGCache
from zephon._internal.io.formats.parquet_cache.codec import ArrowRGFileCodec
from zephon._internal.io.formats.parquet_cache.index import ParquetRGIndex
from zephon._internal.io.types import ShardFile, ShardLocator


def _locator(row_group_rows: list[int]) -> ShardLocator:
    return ShardLocator(
        dataset="dataset",
        shard_id=0,
        format="parquet",
        root="s3://bucket/dataset",
        raw=ShardFile(
            basename="00000.parquet",
            bytes=10_000,
            hashes={"sha256": "source-digest"},
        ),
        extra={
            "num_rows": sum(row_group_rows),
            "num_row_groups": len(row_group_rows),
            "row_groups": [
                {"num_rows": rows, "total_byte_size": rows * 100}
                for rows in row_group_rows
            ],
        },
    )


def _index(
    catalog_dir: Path,
    *,
    row_group_rows: list[int],
) -> ParquetRGIndex:
    ensure_builtin_formats(required={"parquet"})
    catalog_set = catalog_set_from_locators(
        [_locator(row_group_rows)],
        catalog_dir,
        counts={0: sum(row_group_rows)},
    )
    return ParquetRGIndex(catalog_set)


def _read_shared_payload_in_child(
    root: str,
    catalog_dir: str,
    output: multiprocessing.Queue,
) -> None:
    index = _index(Path(catalog_dir), row_group_rows=[3])
    with ParquetRGCache(
        root=root,
        index=index,
        limit_bytes=1_000_000,
        min_free_bytes=0,
    ) as cache:

        def forbidden_decode():
            raise AssertionError(
                "cross-process decoded hit unexpectedly decoded Parquet"
            )

        rows = cache.get_or_decode(
            slot=0,
            local_indices=[2, 0],
            decode=forbidden_decode,
        )
        output.put(([int(row["value"]) for row in rows], cache.stats()["hits"]))


def test_build_then_call_scoped_mmap_hit_avoids_decode(tmp_path: Path) -> None:
    index = _index(tmp_path / "catalog", row_group_rows=[3])
    table = pa.table({"value": [10, 20, 30], "tokens": [[1], [2, 3], [4]]})
    decodes = 0

    def decode():
        nonlocal decodes
        decodes += 1
        return table

    with ParquetRGCache(
        root=tmp_path / "decoded",
        index=index,
        limit_bytes=1_000_000,
        min_free_bytes=0,
    ) as cache:
        first = cache.get_or_decode(slot=0, local_indices=[2, 0, 2], decode=decode)
        second = cache.get_or_decode(slot=0, local_indices=[1], decode=decode)
        stats = cache.stats()

    assert decodes == 1
    assert [int(row["value"]) for row in first] == [30, 10, 30]
    assert [int(row["value"]) for row in second] == [20]
    assert stats["publications"] == 1
    assert stats["hits"] == 1
    assert stats["ready_count"] == 1


def test_unreadable_ready_payload_warns_and_rebuilds(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    index = _index(tmp_path / "catalog", row_group_rows=[3])
    table = pa.table({"value": [10, 20, 30]})
    decodes = 0

    def decode():
        nonlocal decodes
        decodes += 1
        return table

    with ParquetRGCache(
        root=tmp_path / "decoded",
        index=index,
        limit_bytes=1_000_000,
        min_free_bytes=0,
    ) as cache:
        cache.get_or_decode(slot=0, local_indices=[0], decode=decode)
        cache._final_path(0).write_bytes(b"corrupt")
        caplog.set_level(logging.WARNING, logger=parquet_rg_cache_module.__name__)

        rows = cache.get_or_decode(slot=0, local_indices=[1], decode=decode)
        stats = cache.stats()

    assert int(rows[0]["value"]) == 20
    assert decodes == 2
    assert stats["corruptions"] == 1
    assert any(
        "Discarding unreadable decoded RG cache payload" in record.message
        for record in caplog.records
    )


def test_same_rg_waiter_reuses_one_publisher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This test isolates same-RG coordination, not the production timeout path.
    # Cold Arrow startup can exceed 100 ms on a loaded test worker.
    monkeypatch.setattr(parquet_rg_cache_module, "_SAME_RG_WAIT_SECONDS", 2.0)
    index = _index(tmp_path / "catalog", row_group_rows=[3])
    table = pa.table({"value": [10, 20, 30]})
    decode_started = threading.Event()
    release_decode = threading.Event()
    decode_lock = threading.Lock()
    decodes = 0
    results: list[int] = []

    def decode():
        nonlocal decodes
        with decode_lock:
            decodes += 1
        decode_started.set()
        assert release_decode.wait(timeout=2)
        return table

    with ParquetRGCache(
        root=tmp_path / "decoded",
        index=index,
        limit_bytes=1_000_000,
        min_free_bytes=0,
    ) as cache:
        first = threading.Thread(
            target=lambda: results.append(
                int(
                    cache.get_or_decode(
                        slot=0,
                        local_indices=[0],
                        decode=decode,
                    )[0]["value"]
                )
            )
        )
        first.start()
        assert decode_started.wait(timeout=2)
        wait_started = threading.Event()
        original_wait = cache._admission.wait_ready_shared

        def observed_wait(slot: int, *, timeout: float):
            wait_started.set()
            return original_wait(slot, timeout=timeout)

        monkeypatch.setattr(cache._admission, "wait_ready_shared", observed_wait)
        waiter = threading.Thread(
            target=lambda: results.append(
                int(
                    cache.get_or_decode(
                        slot=0,
                        local_indices=[1],
                        decode=decode,
                    )[0]["value"]
                )
            )
        )
        waiter.start()
        assert wait_started.wait(timeout=2)
        release_decode.set()
        first.join(timeout=2)
        waiter.join(timeout=2)
        assert not first.is_alive()
        assert not waiter.is_alive()

    assert decodes == 1
    assert sorted(results) == [10, 20]


def test_independently_spawned_process_reuses_decoded_payload(tmp_path: Path) -> None:
    root = tmp_path / "decoded"
    index = _index(tmp_path / "parent-catalog", row_group_rows=[3])
    table = pa.table({"value": [10, 20, 30]})
    parent = ParquetRGCache(
        root=root,
        index=index,
        limit_bytes=1_000_000,
        min_free_bytes=0,
    )
    parent.get_or_decode(slot=0, local_indices=[0], decode=lambda: table)

    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    process = context.Process(
        target=_read_shared_payload_in_child,
        args=(str(root), str(tmp_path / "child-catalog"), output),
    )
    process.start()
    process.join(timeout=15)
    try:
        assert process.exitcode == 0
        values, hits = output.get(timeout=1)
        assert values == [30, 10]
        assert hits == 1
    finally:
        parent.close()


def test_global_clock_evicts_across_row_groups_to_admit_next_payload(
    tmp_path: Path,
) -> None:
    index = _index(tmp_path / "catalog", row_group_rows=[3, 3])
    first_table = pa.table({"value": [1, 2, 3], "text": ["a", "b", "c"]})
    second_table = pa.table({"value": [4, 5, 6], "text": ["d", "e", "f"]})
    codec = ArrowRGFileCodec()
    first_bytes = codec.measure(first_table, index.identity_for(0))
    second_bytes = codec.measure(second_table, index.identity_for(1))
    limit = max(first_bytes, second_bytes) + 8
    assert first_bytes + second_bytes > limit

    with ParquetRGCache(
        root=tmp_path / "decoded",
        index=index,
        limit_bytes=limit,
        min_free_bytes=0,
    ) as cache:
        cache.get_or_decode(slot=0, local_indices=[0], decode=lambda: first_table)
        cache.get_or_decode(slot=1, local_indices=[0], decode=lambda: second_table)
        stats = cache.stats()

        first_decodes = 0

        def decode_first_again():
            nonlocal first_decodes
            first_decodes += 1
            return first_table

        cache.get_or_decode(
            slot=0,
            local_indices=[1],
            decode=decode_first_again,
        )

    assert stats["ready_count"] == 1
    assert stats["accounted_bytes"] <= limit
    assert stats["evictions"] >= 1
    assert first_decodes == 1


def test_clock_stops_evicting_as_soon_as_next_payload_fits(tmp_path: Path) -> None:
    index = _index(tmp_path / "catalog", row_group_rows=[3, 3, 3, 3])
    tables = [pa.table({"value": [base, base + 1, base + 2]}) for base in range(4)]
    codec = ArrowRGFileCodec()
    payload_bytes = [
        codec.measure(table, index.identity_for(slot))
        for slot, table in enumerate(tables)
    ]
    limit = sum(payload_bytes[:3])

    with ParquetRGCache(
        root=tmp_path / "decoded",
        index=index,
        limit_bytes=limit,
        min_free_bytes=0,
    ) as cache:
        for slot in range(3):
            cache.get_or_decode(
                slot=slot,
                local_indices=[0],
                decode=lambda slot=slot: tables[slot],
            )

        cache.get_or_decode(slot=3, local_indices=[0], decode=lambda: tables[3])
        stats = cache.stats()

    assert stats["publications"] == 4
    assert stats["evictions"] == 1
    assert stats["ready_count"] == 3
    assert stats["accounted_bytes"] <= limit


def test_thrashing_warns_once_per_participant_after_node_wide_eviction(
    tmp_path: Path,
    monkeypatch,
    caplog,
) -> None:
    monkeypatch.setattr(
        parquet_rg_cache_module,
        "_THRASH_MIN_PUBLICATIONS_AFTER_FILL",
        3,
    )
    index = _index(tmp_path / "catalog", row_group_rows=[3, 3])
    tables = [
        pa.table({"value": [1, 2, 3]}),
        pa.table({"value": [4, 5, 6]}),
    ]
    codec = ArrowRGFileCodec()
    payload_bytes = [
        codec.measure(table, index.identity_for(slot))
        for slot, table in enumerate(tables)
    ]
    limit = max(payload_bytes) + 8
    assert sum(payload_bytes) > limit
    caplog.set_level(logging.WARNING, logger=parquet_rg_cache_module.__name__)

    with (
        ParquetRGCache(
            root=tmp_path / "decoded",
            index=index,
            limit_bytes=limit,
            min_free_bytes=0,
        ) as first,
        ParquetRGCache(
            root=tmp_path / "decoded",
            index=index,
            limit_bytes=limit,
            min_free_bytes=0,
        ) as second,
    ):
        first.get_or_decode(slot=0, local_indices=[0], decode=lambda: tables[0])
        second.get_or_decode(slot=1, local_indices=[0], decode=lambda: tables[1])
        for _ in range(2):
            first.get_or_decode(slot=0, local_indices=[0], decode=lambda: tables[0])
            second.get_or_decode(slot=1, local_indices=[0], decode=lambda: tables[1])
        first_stats = first.stats()
        second_stats = second.stats()

    warnings = [
        record.message for record in caplog.records if "thrashing" in record.message
    ]
    assert len(warnings) == 2
    assert all("thrashing across this node" in message for message in warnings)
    assert all("had previously been evicted" in message for message in warnings)
    assert all(
        "io_options.parquet_rg_cache.limit_bytes" in message for message in warnings
    )
    assert first_stats["reloads"] >= 1
    assert first_stats["thrash_warned"] == 1
    assert second_stats["reloads"] >= 1
    assert second_stats["thrash_warned"] == 1
    assert first_stats["node_post_fill_publications"] >= 3
    assert first_stats["node_post_fill_reloads"] >= 2
    assert (
        second_stats["node_post_fill_publications"]
        == first_stats["node_post_fill_publications"]
    )
    assert (
        second_stats["node_post_fill_reloads"] == first_stats["node_post_fill_reloads"]
    )


def test_thrashing_rewarns_only_for_fresh_churn_after_interval(
    tmp_path: Path,
    monkeypatch,
    caplog,
) -> None:
    now = 0.0
    monkeypatch.setattr(
        parquet_rg_cache_module,
        "_THRASH_MIN_PUBLICATIONS_AFTER_FILL",
        3,
    )
    monkeypatch.setattr(parquet_rg_cache_module.time, "monotonic", lambda: now)
    caplog.set_level(logging.WARNING, logger=parquet_rg_cache_module.__name__)

    with ParquetRGCache(
        root=tmp_path / "decoded",
        index=_index(tmp_path / "catalog", row_group_rows=[3]),
        limit_bytes=1_000_000,
        min_free_bytes=0,
    ) as cache:
        cache._maybe_warn_thrashing(RGPublicationResult(True, 3, 2))
        assert len(caplog.records) == 1

        now = 29.0
        cache._maybe_warn_thrashing(RGPublicationResult(True, 6, 4))
        assert len(caplog.records) == 1

        now = 30.0
        cache._maybe_warn_thrashing(RGPublicationResult(True, 7, 5))
        assert len(caplog.records) == 2

        # A full window without churn rolls the baseline without warning.
        now = 60.0
        cache._maybe_warn_thrashing(RGPublicationResult(False, 10, 5))
        assert len(caplog.records) == 2

        # Churn that resumes is measured independently of that quiet window.
        now = 90.0
        cache._maybe_warn_thrashing(RGPublicationResult(True, 13, 8))

    assert len(caplog.records) == 3
    assert "3 of 3" in caplog.records[-1].message
