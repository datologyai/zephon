# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Tests for Ray actor resource cleanup."""

import logging
from pathlib import Path
from unittest.mock import Mock

import pytest

from tests.zephon._internal.runners._helpers import _mk_records, _ResourceOp
from zephon._internal.graph import Node

ray = pytest.importorskip("ray")

from zephon._internal.runners.ray.service import _RayActorGroup

pytestmark = pytest.mark.requires_ray


@pytest.mark.parametrize("hard", [False, True])
@pytest.mark.usefixtures("ray_init")
def test_shutdown_releases_operator_resources(tmp_path: Path, hard: bool) -> None:
    group = _RayActorGroup(
        node=Node(name="resource", op=_ResourceOp(tmp_path)),
        op_index=0,
        num_actors=1,
        stage_index=0,
        stage_name="test",
        collect_stats=False,
        ctx_services={},
    )
    group.init(num_cpus_per_actor=0.1)
    try:
        ray.get(group.actors[0].process.remote(_mk_records([1]), 0))
        assert len(list(tmp_path.glob("*.live"))) == 1
    finally:
        group.shutdown(hard=hard)
    assert not list(tmp_path.glob("*.live"))
    assert len(list(tmp_path.glob("*.closed"))) == 1


def _mock_group(tmp_path: Path) -> tuple[_RayActorGroup, Mock]:
    group = _RayActorGroup(
        node=Node(name="resource", op=_ResourceOp(tmp_path)),
        op_index=0,
        num_actors=1,
        stage_index=0,
        stage_name="test",
        collect_stats=False,
        ctx_services={},
    )
    actor = Mock()
    group.actors = [actor]
    return group, actor


@pytest.mark.parametrize("pending", [False, True])
def test_placement_failure_skips_teardown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pending: bool
) -> None:
    group, actor = _mock_group(tmp_path)
    ref = actor.get_node_id.remote.return_value
    monkeypatch.setattr(
        ray, "wait", Mock(return_value=([], [ref]) if pending else ([ref], []))
    )
    get = Mock(side_effect=ray.exceptions.RayError("unschedulable"))
    kill = Mock()
    monkeypatch.setattr(ray, "get", get)
    monkeypatch.setattr(ray, "kill", kill)
    with pytest.raises(RuntimeError, match="Actor placement (timed out|failed)"):
        group._probe_hard_placement(["missing-node"])
    actor.teardown.remote.assert_not_called()
    assert get.call_count == (0 if pending else 1)
    kill.assert_called_once()
    assert not group.actors


@pytest.mark.parametrize("timed_out", [False, True])
@pytest.mark.parametrize("hard", [False, True])
def test_shutdown_logs_expected_timeouts_without_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    timed_out: bool,
    hard: bool,
) -> None:
    caplog.set_level(logging.DEBUG, logger="zephon._internal.runners.ray.service")
    group, actor = _mock_group(tmp_path)
    error = ray.exceptions.GetTimeoutError() if timed_out else RuntimeError("failed")
    monkeypatch.setattr(ray, "get", Mock(side_effect=error))
    kill = Mock()
    monkeypatch.setattr(ray, "kill", kill)
    group.shutdown(hard=hard)
    actor.teardown.remote.assert_called_once_with(hard=hard)
    kill.assert_called_once()
    assert len(caplog.records) == 1
    assert bool(caplog.records[0].exc_info) is not timed_out
    assert caplog.records[0].levelno == (
        logging.DEBUG if hard and timed_out else logging.WARNING
    )
