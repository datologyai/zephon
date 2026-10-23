"""Retain a real pipeline until normal exit or an interrupt from the parent test."""

from __future__ import annotations

import argparse
import json
import sys
import weakref
from collections.abc import Iterator
from multiprocessing import shared_memory
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from zephon import Pipeline

_pipeline: Pipeline | None = None
_iterator: Iterator[Any] | None = None


class _ExitObserver:
    pass


_observer = _ExitObserver()


def _observe_cleanup(root: Path) -> None:
    ready = json.loads((root / "ready.json").read_text())
    session_path = root / "cache" / ".zephon_cache_state" / "session.json"
    session = json.loads(session_path.read_text())
    remaining_shm = []
    for name in ready["shm_names"].values():
        try:
            segment = shared_memory.SharedMemory(name=name, create=False)
        except FileNotFoundError:
            continue
        segment.close()
        remaining_shm.append(name)
    (root / "exit-observation.json").write_text(
        json.dumps({"owners": session["owners"], "remaining_shm": remaining_shm})
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("runner")
    parser.add_argument("shutdown_mode")
    args = parser.parse_args()

    # Older finalizers run last: observe Zephon cleanup before the resource
    # tracker can unlink leftover SHM after this interpreter has exited.
    weakref.finalize(_observer, _observe_cleanup, args.root)
    from zephon import Pipeline
    from zephon.io.dataset import Dataset
    from zephon.work import MixtureSpec, StaticMixtureWorkSource

    dataset = Dataset.from_path("data", str(args.root / "source"))
    work = StaticMixtureWorkSource(
        [dataset],
        mixture=MixtureSpec({"data": 1.0}),
        chunk_size=4,
        seed=1,
        shuffle_shards=False,
    )
    global _pipeline, _iterator
    _pipeline = (
        Pipeline(work)
        .prefetch(buffer_size=4, parallelism=1)
        .fetch(parallelism=1)
        .batch(microbatch_size=4, drop_last=False)
        .options(
            runner=args.runner,
            shutdown_mode=args.shutdown_mode,
            max_workers=2,
            mtp_mode=False,
            io_options={"cache": {"enabled": True, "root": str(args.root / "cache")}},
        )
    )
    _iterator = iter(_pipeline)
    assert next(_iterator)

    session_path = args.root / "cache" / ".zephon_cache_state" / "session.json"
    session = json.loads(session_path.read_text())
    for name in session["shm_names"].values():
        segment = shared_memory.SharedMemory(name=name, create=False)
        segment.close()
    ready_tmp = args.root / "ready.tmp"
    ready_tmp.write_text(json.dumps(session))
    ready_tmp.replace(args.root / "ready.json")

    # The consumer is outside next(), as when training on a yielded batch.
    # Globals retain the iterator and engine through interpreter shutdown.
    sys.stdin.readline()


if __name__ == "__main__":
    main()
