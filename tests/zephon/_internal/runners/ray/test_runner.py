"""Tests for Ray runner resource ownership."""

from pathlib import Path

import pytest

from tests.zephon._internal.runners._helpers import (
    _ctx_services,
    _mk_record,
    _mk_records,
    _resource_stage,
    _ResourceOp,
)

pytest.importorskip("ray")

from zephon._internal.runners.ray.runner import RemoteStageRunner

pytestmark = pytest.mark.requires_ray


@pytest.mark.parametrize("hard", [False, True])
def test_close_releases_driver_resources_and_rejects_reuse(
    tmp_path: Path, hard: bool
) -> None:
    runner = RemoteStageRunner(
        _resource_stage(_ResourceOp(tmp_path)),
        _ctx_services(),
        max_workers=1,
        stage_output_mode="stream_items",
    )
    assert list(tmp_path.glob("*.live"))
    runner.close(hard=hard)
    runner.close(hard=hard)
    assert not list(tmp_path.glob("*.live"))
    assert len(list(tmp_path.glob("*.closed"))) == 1
    with pytest.raises(RuntimeError, match="Stage runner is closed"):
        runner.run_one(_mk_record(1))
    with pytest.raises(RuntimeError, match="Stage runner is closed"):
        list(runner.run(_mk_records([1])))
