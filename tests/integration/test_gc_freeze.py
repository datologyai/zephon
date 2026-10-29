# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Exercise real permanent generations without freezing pytest's interpreter."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def _run(tmp_path: Path, source: str, *args: str) -> None:
    script = tmp_path / "gc_case.py"
    script.write_text(textwrap.dedent(source))
    root = str(Path(__file__).resolve().parents[2])
    env = dict(
        os.environ, PYTHONPATH=os.pathsep.join([root, os.environ.get("PYTHONPATH", "")])
    )
    result = subprocess.run(
        [sys.executable, str(script), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("prefetch", [0, 2])
@pytest.mark.parametrize("end", ["exhaust", "close", "abandon", "error"])
def test_inline_unfreezes_and_releases_cycles(
    tmp_path: Path, prefetch: int, end: str
) -> None:
    _run(
        tmp_path,
        """
        import gc
        import sys
        import weakref

        from tests.helpers.work import FakeIndexableWorkSource, make_inmem_dataset
        from zephon import Pipeline

        class Cycle:
            def __init__(self):
                self.link = self

        prefetch, end = int(sys.argv[1]), sys.argv[2]
        enabled, thresholds = gc.isenabled(), gc.get_threshold()
        old = Cycle()
        old_ref = weakref.ref(old)
        retained_payload_ref = None
        observed_freeze = []

        def transform(payload):
            observed_freeze.append(gc.get_freeze_count() > 0)
            if end == "error" and payload["value"] == 3:
                raise RuntimeError("deliberate iteration failure")
            return payload

        ds = make_inmem_dataset("gc", [{"value": i} for i in range(8)])
        pipe = Pipeline(FakeIndexableWorkSource(ds, chunk_size=1)).map_transform(transform)
        pipe.options(mtp_mode=False, gc_freeze=True, runner="inline",
                     prefetch_batches=prefetch, auto_validation="off")
        iterator = iter(pipe)
        if end == "error":
            # Prefetch can surface an error before the first batch is yielded.
            try:
                list(iterator)
            except RuntimeError as exc:
                assert "deliberate iteration failure" in str(exc)
            else:
                raise AssertionError("expected iteration failure")
            del old
        else:
            next(iterator)
            assert gc.get_freeze_count() > 0
            # Existing frozen cycles survive collection; new cycles do not.
            old.payload = Cycle()
            retained_payload_ref = weakref.ref(old.payload)
            del old
            new = Cycle()
            new_ref = weakref.ref(new)
            del new
            gc.collect()
            assert old_ref() is not None
            assert retained_payload_ref() is not None
            assert new_ref() is None
            if end == "exhaust":
                list(iterator)
            elif end == "close":
                iterator.close()
        engine_ref = weakref.ref(pipe._engine)
        assert any(obj is pipe._engine for obj in gc.get_objects())
        assert observed_freeze and all(observed_freeze)
        # Dropping an unclosed iterator must run its teardown too.
        del iterator
        assert gc.get_freeze_count() == 0
        assert gc.isenabled() == enabled
        assert gc.get_threshold() == thresholds
        del pipe
        gc.collect()
        assert old_ref() is None
        assert retained_payload_ref is None or retained_payload_ref() is None
        assert engine_ref() is None
        """,
        str(prefetch),
        end,
    )


def test_real_overlapping_scopes_and_application_owned_freeze(tmp_path: Path) -> None:
    _run(
        tmp_path,
        """
        import gc
        import weakref
        from zephon._internal.utils.gc_freeze import GCFreeze

        class Cycle:
            def __init__(self):
                self.link = self

        old = Cycle()
        ref = weakref.ref(old)
        first, second = GCFreeze(True), GCFreeze(True)
        del old
        first.close()
        gc.collect()
        assert gc.get_freeze_count() > 0 and ref() is not None
        del second
        assert gc.get_freeze_count() == 0
        gc.collect()
        assert ref() is None

        gc.freeze()
        count = gc.get_freeze_count()
        guard = GCFreeze(True)
        guard.close()
        del guard
        assert gc.get_freeze_count() == count
        gc.unfreeze()
        """,
    )


@pytest.mark.parametrize("end", ["exhaust", "close", "abandon", "error"])
def test_mtp_releases_freeze_before_child_exit(tmp_path: Path, end: str) -> None:
    _run(
        tmp_path,
        """
        import gc
        from multiprocessing.util import Finalize
        from pathlib import Path
        import sys

        from tests.helpers.work import FakeIndexableWorkSource, make_inmem_dataset
        from zephon import Pipeline

        def record_gc_at_exit(path):
            Path(path).write_text(str(gc.get_freeze_count()))

        class ExitProbe:
            def __init__(self, path, fail):
                self.path, self.fail = path, fail
                self.registered = False

            def __call__(self, payload):
                if not self.registered:
                    Finalize(None, record_gc_at_exit, args=(self.path,), exitpriority=0)
                    self.registered = True
                assert gc.get_freeze_count() > 0
                if self.fail and payload["value"] == 3:
                    raise RuntimeError("deliberate iteration failure")
                return payload

        if __name__ == "__main__":
            end, result = sys.argv[1], sys.argv[2]
            ds = make_inmem_dataset("gc", [{"value": i} for i in range(32)])
            pipe = Pipeline(FakeIndexableWorkSource(ds, chunk_size=1))
            pipe.map_transform(ExitProbe(result, end == "error"))
            pipe.options(mtp_mode=True, runner="inline", mtp_buffer=2,
                         prefetch_batches=2, auto_validation="off")
            parent_frozen = gc.get_freeze_count()
            iterator = iter(pipe)
            process = None
            if end == "error":
                try:
                    list(iterator)
                except RuntimeError as exc:
                    assert "deliberate iteration failure" in str(exc)
                else:
                    raise AssertionError("expected iteration failure")
            else:
                next(iterator)
                process = pipe._sp._process
                if end == "exhaust":
                    list(iterator)
                elif end == "close":
                    iterator.close()
            del iterator
            assert process is None or not process.is_alive()
            assert Path(result).read_text() == "0"
            assert gc.get_freeze_count() == parent_frozen
        """,
        end,
        str(tmp_path / "freeze_at_exit.txt"),
    )
