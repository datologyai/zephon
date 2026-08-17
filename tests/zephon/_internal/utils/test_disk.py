# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from zephon._internal.utils.disk import (
    InsufficientCacheSpaceError,
    check_cache_disk_budgets,
    check_cache_disk_space,
    device_space,
    dir_usage_bytes,
)

GiB = 1024**3


def _patch_device(total_bytes: int, free_bytes: int):
    return patch(
        "zephon._internal.utils.disk.shutil.disk_usage",
        return_value=SimpleNamespace(
            total=total_bytes,
            used=total_bytes - free_bytes,
            free=free_bytes,
        ),
    )


# ---------------------------------------------------------------------------
# device_space
# ---------------------------------------------------------------------------


def test_device_space_total_and_free(tmp_path: Path) -> None:
    with _patch_device(100 * GiB, 25 * GiB):
        assert device_space(tmp_path) == (100 * GiB, 25 * GiB)


def test_device_space_none_on_oserror(tmp_path: Path) -> None:
    with patch("zephon._internal.utils.disk.shutil.disk_usage", side_effect=OSError):
        assert device_space(tmp_path) is None


def test_device_space_walks_to_existing_ancestor(tmp_path: Path) -> None:
    seen: dict[str, Path] = {}
    real_disk_usage = shutil.disk_usage

    def spy(path: Path):
        seen["probe"] = path
        return real_disk_usage(path)

    with patch("zephon._internal.utils.disk.shutil.disk_usage", side_effect=spy):
        assert device_space(tmp_path / "missing" / "nested") is not None
    assert seen["probe"] == tmp_path


# ---------------------------------------------------------------------------
# dir_usage_bytes
# ---------------------------------------------------------------------------


def test_dir_usage_bytes_counts_nested_files(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "f1").write_bytes(b"x" * 1000)
    (tmp_path / "f2").write_bytes(b"y" * 2000)
    # st_blocks rounds up to whole filesystem blocks.
    assert dir_usage_bytes(tmp_path) >= 3000


def test_dir_usage_bytes_missing_dir(tmp_path: Path) -> None:
    assert dir_usage_bytes(tmp_path / "nope") == 0


# ---------------------------------------------------------------------------
# check_cache_disk_space
# ---------------------------------------------------------------------------


def test_check_skips_when_no_limit(tmp_path: Path) -> None:
    with patch(
        "zephon._internal.utils.disk.shutil.disk_usage", side_effect=AssertionError
    ):
        check_cache_disk_space(tmp_path, None)


def test_check_env_var_bypass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZEPHON_DISABLE_CACHE_SPACE_CHECK", "1")
    with patch(
        "zephon._internal.utils.disk.shutil.disk_usage", side_effect=AssertionError
    ):
        check_cache_disk_space(tmp_path, 10**18)


def test_check_permissive_when_disk_usage_unavailable(tmp_path: Path) -> None:
    with patch("zephon._internal.utils.disk.shutil.disk_usage", side_effect=OSError):
        check_cache_disk_space(tmp_path, 10**18)


def test_check_raises_when_limit_exceeds_usable(tmp_path: Path) -> None:
    with (
        _patch_device(100 * GiB, 10 * GiB),
        pytest.raises(InsufficientCacheSpaceError) as excinfo,
    ):
        check_cache_disk_space(tmp_path, 50 * GiB)
    err = excinfo.value
    assert err.limit_bytes == 50 * GiB
    assert err.free_bytes == 10 * GiB
    assert err.existing_bytes == 0
    assert err.device_total == 100 * GiB
    assert "ZEPHON_DISABLE_CACHE_SPACE_CHECK" in str(err)


def test_check_passes_with_headroom_no_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with (
        _patch_device(1000 * GiB, 1000 * GiB),
        caplog.at_level(logging.WARNING, logger="zephon._internal.utils.disk"),
    ):
        check_cache_disk_space(tmp_path, 100 * GiB)
    assert not caplog.records


def test_check_warns_when_headroom_below_fraction(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # 970GiB of a 1000GiB device: fits, but leaves 30GiB = 3% < 5%.
    with (
        _patch_device(1000 * GiB, 1000 * GiB),
        caplog.at_level(logging.WARNING, logger="zephon._internal.utils.disk"),
    ):
        check_cache_disk_space(tmp_path, 970 * GiB)
    assert len(caplog.records) == 1
    assert "headroom" in caplog.records[0].getMessage()


def test_check_warm_restart_counts_existing_cache(tmp_path: Path) -> None:
    # Free space alone (1GiB) is far below the limit, but the cache root
    # already holds 60GiB that counts toward the limit — must pass.
    with (
        _patch_device(100 * GiB, 1 * GiB),
        patch("zephon._internal.utils.disk.dir_usage_bytes", return_value=60 * GiB),
    ):
        check_cache_disk_space(tmp_path, 50 * GiB)


def test_check_fast_path_skips_walk(tmp_path: Path) -> None:
    with (
        _patch_device(1000 * GiB, 1000 * GiB),
        patch(
            "zephon._internal.utils.disk.dir_usage_bytes", side_effect=AssertionError
        ),
    ):
        check_cache_disk_space(tmp_path, 100 * GiB)


def test_check_existing_bytes_reuses_tally_without_walking(tmp_path: Path) -> None:
    # free (10GiB) < limit (50GiB) so the fast path is off, but a passed-in
    # tally counts toward the limit with no walk (dir_usage_bytes would assert).
    with (
        _patch_device(100 * GiB, 10 * GiB),
        patch(
            "zephon._internal.utils.disk.dir_usage_bytes", side_effect=AssertionError
        ),
    ):
        check_cache_disk_space(tmp_path, 50 * GiB, existing_bytes=45 * GiB)
        with pytest.raises(InsufficientCacheSpaceError):
            check_cache_disk_space(tmp_path, 50 * GiB, existing_bytes=1 * GiB)


def test_check_existing_bytes_loader_runs_only_after_fast_path(tmp_path: Path) -> None:
    calls = 0

    def existing_bytes() -> int:
        nonlocal calls
        calls += 1
        return 45 * GiB

    with _patch_device(100 * GiB, 100 * GiB):
        check_cache_disk_space(tmp_path, 50 * GiB, existing_bytes=existing_bytes)
    assert calls == 0

    with _patch_device(100 * GiB, 10 * GiB):
        check_cache_disk_space(tmp_path, 50 * GiB, existing_bytes=existing_bytes)
    assert calls == 1


def test_check_warn_fraction_none_suppresses_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # Headroom (5GiB of 1000GiB) is below the default fraction; the
    # per-worker mode must stay silent.
    with (
        _patch_device(1000 * GiB, 10 * GiB),
        patch("zephon._internal.utils.disk.dir_usage_bytes", return_value=50 * GiB),
        caplog.at_level(logging.WARNING, logger="zephon._internal.utils.disk"),
    ):
        check_cache_disk_space(tmp_path, 55 * GiB, warn_fraction=None)
    assert not caplog.records


def test_budgets_combine_limits_on_one_device(tmp_path: Path) -> None:
    with (
        patch("zephon._internal.utils.disk.device_id", return_value=1),
        _patch_device(100 * GiB, 12 * GiB),
        pytest.raises(InsufficientCacheSpaceError),
    ):
        check_cache_disk_budgets(
            [(tmp_path / "raw", 8 * GiB), (tmp_path / "decoded", 5 * GiB)]
        )


def test_budgets_check_different_devices_separately(tmp_path: Path) -> None:
    raw_root = (tmp_path / "raw").resolve()
    decoded_root = (tmp_path / "decoded").resolve()

    def device_id(path: Path) -> int:
        return 1 if Path(path).resolve() == raw_root else 2

    def device_space(path: Path) -> tuple[int, int]:
        free = 9 * GiB if Path(path).resolve() == raw_root else 6 * GiB
        return 100 * GiB, free

    with (
        patch("zephon._internal.utils.disk.device_id", side_effect=device_id),
        patch("zephon._internal.utils.disk.device_space", side_effect=device_space),
    ):
        check_cache_disk_budgets([(raw_root, 8 * GiB), (decoded_root, 5 * GiB)])


def test_combined_budget_fast_path_skips_cache_walk(tmp_path: Path) -> None:
    with (
        patch("zephon._internal.utils.disk.device_id", return_value=1),
        patch(
            "zephon._internal.utils.disk.device_space",
            return_value=(100 * GiB, 100 * GiB),
        ),
        patch(
            "zephon._internal.utils.disk.dir_usage_bytes",
            side_effect=AssertionError,
        ),
    ):
        check_cache_disk_budgets(
            [(tmp_path / "raw", 8 * GiB), (tmp_path / "decoded", 5 * GiB)]
        )


def test_nested_budgets_walk_only_the_outer_root(tmp_path: Path) -> None:
    raw_root = (tmp_path / "raw").resolve()
    decoded_root = raw_root / ".parquet-rg-cache"

    def dir_usage(path: Path) -> int:
        assert Path(path) == raw_root
        return 12 * GiB

    with (
        patch("zephon._internal.utils.disk.device_id", return_value=None),
        patch(
            "zephon._internal.utils.disk.device_space",
            return_value=(100 * GiB, 1 * GiB),
        ),
        patch("zephon._internal.utils.disk.dir_usage_bytes", side_effect=dir_usage),
    ):
        check_cache_disk_budgets([(raw_root, 8 * GiB), (decoded_root, 4 * GiB)])
