# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Tests for Ray actor resource cleanup."""

from pathlib import Path

import pytest

from tests.zephon._internal.runners._helpers import _mk_records, _ResourceOp
from zephon._internal.graph import Node

ray = pytest.importorskip("ray")

from zephon._internal.runners.ray.service import _RayActorGroup

pytestmark = [pytest.mark.requires_ray, pytest.mark.usefixtures("ray_init")]


def test_shutdown_releases_operator_resources(tmp_path: Path) -> None:
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
        group.shutdown()
    assert not list(tmp_path.glob("*.live"))
    assert len(list(tmp_path.glob("*.closed"))) == 1
