"""Cross-rank checkpoint aggregation.

The engine builds local snapshots and supplies the merge and validation logic.
A :class:`CheckpointCoordinator` exchanges those snapshots and the merged result
between participants. :class:`FileCoordinator` uses a shared ``aggregate_dir``.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeVar

from zephon._internal.checkpoint import AggregationCodec
from zephon._internal.io.storage import RouterStorageBackend
from zephon.options import DEFAULT_RUN_ID, RuntimeOptions

T = TypeVar("T")

_RELOAD_SUFFIX = re.compile(r"-(\d+)$")


@dataclass(frozen=True)
class AggregationRound:
    """One rank/worker's snapshot and the required rank and lane coverage."""

    local: dict[str, Any]
    """This rank/worker's local engine state."""

    is_leader: bool
    """Whether this process merges the round (global rank 0, worker 0)."""

    expected_ranks: frozenset[int]
    """Global ranks whose assigned lanes the leader must collect in full."""

    expected_lanes: frozenset[int]
    """Canonical lanes the collected states must cover."""


class CheckpointCoordinator(Protocol):
    """Coordinate checkpoint rounds across active rank/worker processes.

    Callers share a unique run namespace, topology, and codec settings, snapshot
    the same logical training step, and enter rounds in the same order. Each
    round has one leader; calls on an instance must not overlap.

    Implementations isolate runs, restore generations, and rounds. Publish round
    metadata, shards, and results atomically as complete values and make them
    eventually visible to their consumers. Keep shards and results immutable
    and retain them until their required consumers finish. Preserve checkpoint
    contents under the agreed codec's encoding/decoding rules.
    """

    last_round_id: str | None
    """Last joined round ID, initially None; the engine saves and restores it."""

    reload_count: int
    """Restore generation, initially zero; set by the engine before preparation."""

    def aggregate(
        self,
        round_: AggregationRound,
        merge: Callable[[list[dict[str, Any]]], dict[str, Any]],
    ) -> dict[str, Any]:
        """Return the same decoded merged state to every caller of this round.

        The leader opens a fresh round; followers join that same round. Set
        ``last_round_id`` when joining and publish one local shard per rank/worker.
        Collect this round's shards covering exactly ``expected_ranks``, each
        rank's assigned lanes from the snapshot's world mapping, and all
        ``expected_lanes``. Then call ``merge`` once on the leader and publish
        its result. Before the leader returns, every contributor must have
        decoded the result. Bound waits by the configured timeout; raise on
        failed exchange, encoding, decoding, merge, or timeout.
        """
        ...

    def prepare_restore(self) -> None:
        """Prepare an isolated namespace for the restored stream's rounds.

        Callers restore the same checkpoint on every participant. The engine has
        restored ``last_round_id`` and set ``reload_count`` to the saved count
        plus one. Preserve both values; repeated preparation with the same
        values must be safe.
        """
        ...

    def cleanup(self) -> None:
        """Idempotently release round resources whose consumers have finished.

        Preserve unconsumed results after a failed exchange. Leave other runs'
        resources intact.
        """
        ...


class FileCoordinator:
    """Checkpoint aggregation over files in a shared ``aggregate_dir``.

    The leader publishes a round ID, every rank writes its state, the leader
    lists the state files until every rank covers its assigned lanes, merges
    them and writes the merged state. Followers acknowledge successful reads
    before the leader returns and allows the result to be cleaned up.
    """

    def __init__(
        self,
        opts: RuntimeOptions,
        *,
        global_rank: int,
        world_size: int,
        plan_id: str,
        worker_info: Callable[[], tuple[int, int]],
    ) -> None:
        self._opts = opts
        self._global_rank = global_rank
        self._world_size = world_size
        self._plan_id = plan_id
        self._worker_info = worker_info
        self.reload_count = 0

        self._using_fresh_tmp = False
        if world_size == 1:
            # Single-node: auto if not provided
            base = opts.aggregate_dir or os.path.join(
                tempfile.gettempdir(), f"zephon_state_r{global_rank}"
            )
            self._using_fresh_tmp = (
                opts.aggregate_dir is None or opts.aggregate_dir == ""
            )
        else:
            # Multi-node: must be provided (shared FS or cloud storage)
            if not opts.aggregate_dir:
                raise RuntimeError(
                    "RuntimeOptions.aggregate_dir must be set for multi-node checkpoint aggregation."
                )
            base = opts.aggregate_dir

        # Initialize storage backend and aggregation codec for checkpoint I/O
        self.backend = RouterStorageBackend()
        self._codec = AggregationCodec(
            serializer=opts.aggregate_serializer,
            compressor=opts.aggregate_compressor,
        )

        # Detect cloud storage and adjust timeouts for higher latency
        base = str(base)  # Handle Path objects
        self._is_cloud = self.backend.is_cloud_path(base)
        if self._is_cloud:
            # Floor: guard against user-provided values that are too low for cloud I/O.
            self._timeout_s = max(opts.aggregate_timeout_s, 90.0)
            self._poll_base = 0.5  # 500ms base poll interval
        else:
            self._timeout_s = opts.aggregate_timeout_s
            self._poll_base = 0.05  # 50ms base poll interval
            # Resolve local paths; cloud paths stay as-is
            base = str(Path(base).resolve())

        self._base = base.rstrip("/")
        self.backend.mkdir(self._base, parents=True, exist_ok=True)
        self.last_round_id: str | None = None
        self._previous_merged_file: str | None = None
        self._warned_once_about_runid = False

    def _log(self, msg: str) -> None:
        worker_id, workers_per_rank = self._worker_info()
        print(
            f"[PR {self._global_rank}][PID {os.getpid()}][Worker {worker_id}/{workers_per_rank - 1}] {msg}",
            file=sys.stderr,
        )

    @property
    def round_file(self) -> str:
        return f"{self.agg_dir}/round.current"

    @property
    def agg_dir(self) -> str:
        def _normalize_run_dir_name(
            name: str,
        ) -> str:  # Drop a trailing "-<int>" if present
            m = _RELOAD_SUFFIX.search(name)
            return name[: m.start()] if m else name

        run_id = self._opts.run_id
        using_default = False
        # Names (not full paths) of subdirectories in _base, used to warn
        # about potential run ID collisions when using the default run ID.
        existing_subdir_names: list[str] = []

        if run_id.startswith(DEFAULT_RUN_ID):
            existing_subdir_names = self.backend.listdir(self._base)
            using_default = True
            # If users don't provide a run ID, we want to help them not shoot themselves in the foot.
            run_id = f"{run_id}-{self._opts.canonical_replicas}-{self._world_size}-{self._plan_id}"

        # Avoid having to use a new run id with every reload
        run_id = f"{run_id}-{self.reload_count}"
        current_run_dir = f"{self._base}/{run_id}"
        cur_norm = _normalize_run_dir_name(run_id)
        result = f"{current_run_dir}/.zephon_agg"

        if (
            not self._using_fresh_tmp
            and using_default
            and not self._warned_once_about_runid
            and existing_subdir_names
        ):
            existing_norms = {
                _normalize_run_dir_name(name) for name in existing_subdir_names
            }
            if existing_norms != {cur_norm}:
                # Potentially we can also only warn if the plan id or canonical replicas change since that probably really causes a semantic change but we better just tell the user early this is not the. best idea.
                print(
                    f"Warning! No run id has been supplied. This can cause issues in checkpointing if the same aggregate_dir ({self._base}) is used across multiple runs. Your current supplied directory contains data from other runs (or your world changed), which might indicate that you re-use that directory (or use it for other purposes as well). Zephon adjusted the run id to {run_id} to avoid problems, but if you choose to run exactly the same pipeline twice in the samed directory, issues might still occur without providing a run id.\n\nOffending subdirs: {existing_subdir_names}",
                    file=sys.stderr,
                )
                self._warned_once_about_runid = True

        return result

    def _state_file_path(self, round_id: str) -> str:
        wid, _ = self._worker_info()
        pid = os.getpid()
        return (
            f"{self.agg_dir}/state_r{self._global_rank}_w{wid}_p{pid}_{round_id}.ckpt"
        )

    def _merged_file_path(self, round_id: str) -> str:
        return f"{self.agg_dir}/merged_{round_id}.ckpt"

    def _list_state_files(self, round_id: str) -> list[str]:
        pattern = f"{self.agg_dir}/state_r*_w*_p*_{round_id}.ckpt"
        return self.backend.glob(pattern)

    # ---------- FS utilities ----------

    def _read_text(self, path: str) -> str | None:
        try:
            with self.backend.open(path, "r") as f:
                return str(f.read())
        except Exception:
            return None

    def _read_aggregation_state(self, path: str) -> dict | None:
        try:
            with self.backend.open(path, "rb") as f:
                raw = f.read()
        except Exception:
            return None
        try:
            return self._codec.decode(raw)
        except Exception as exc:
            self._log(f"Failed to decode aggregation state at {path}: {exc!r}")
            return None

    def _wait_until(self, pred: Callable[[], bool], timeout: float) -> bool:
        deadline = time.time() + timeout
        delay = self._poll_base
        while time.time() < deadline:
            if pred():
                return True
            time.sleep(min(delay, max(0, deadline - time.time())))
            # Exponential backoff for cloud, capped at 5s
            if self._is_cloud:
                delay = min(delay * 1.5, 5.0)
        return False

    def _wait_value(
        self,
        supplier: Callable[[], T | None],
        timeout: float,
    ) -> T | None:
        deadline = time.time() + timeout
        delay = self._poll_base
        while time.time() < deadline:
            val = supplier()
            if val is not None:
                return val
            time.sleep(min(delay, max(0, deadline - time.time())))
            if self._is_cloud:
                delay = min(delay * 1.5, 5.0)
        return None

    # ---------- Round / paths ----------

    def _publish_new_round_id(self) -> str:
        rid = str(int(time.time() * 1e9))  # monotonic-ish
        self.backend.put(self.round_file, rid.encode("utf-8"))
        self.last_round_id = rid
        return rid

    def _read_open_round_id(self) -> str | None:
        rid = (self._read_text(self.round_file) or "").strip()
        if not rid:
            return None

        # This is a practically relevant guard in case users do neither supply a run id nor a fresh aggregate directory. With this check we at least avoid stale state if it is obviously old.
        try:
            stat = self.backend.stat(self.round_file)
            age_s = time.time() - stat["mtime"]
        except OSError:
            return None  # File doesn't exist or access denied
        if age_s > self._timeout_s + 30:
            return None

        # Guard: if we already consumed this rid, skip if our own shard for
        # this rid exists. Under fork, a child can inherit last_round_id==rid
        # from the parent before it has published; in that case we must proceed.
        if self.last_round_id is not None and rid == self.last_round_id:
            try:
                my_fp = self._state_file_path(rid)
            except Exception:
                my_fp = None
            if not my_fp or self.backend.exists(my_fp):
                return None

        # only "open" if merged for this rid doesn't exist yet
        if not self.backend.exists(self._merged_file_path(rid)):
            return rid
        return None

    def _open_round_id(self, is_leader: bool) -> str:
        if is_leader:
            return self._publish_new_round_id()

        rid = self._wait_value(self._read_open_round_id, self._timeout_s)
        if rid is None:
            raise RuntimeError(f"Timeout waiting for round id at {self.round_file}")
        return rid

    def _read_states_for_round(
        self,
        round_id: str,
        *,
        seen: dict[str, dict[str, Any]] | None = None,
        covered: set[int] | None = None,
    ) -> tuple[dict[str, dict[str, Any]], set[int]]:
        """Read per-rank state files for *round_id*.

        When *seen* and *covered* are supplied, only newly appeared files are
        fetched — previously read files are skipped.  This turns repeated
        polling from O(N * polls) GETs into O(N) total.
        """
        if seen is None:
            seen = {}
        if covered is None:
            covered = set()
        for fp in self._list_state_files(round_id):
            if fp in seen:
                continue
            st = self._read_aggregation_state(fp)
            if st is None:
                continue
            seen[fp] = st
            for k in st.get("progress", {}):
                covered.add(int(k))
        return seen, covered

    def aggregate(
        self,
        round_: AggregationRound,
        merge: Callable[[list[dict[str, Any]]], dict[str, Any]],
    ) -> dict[str, Any]:
        is_leader = round_.is_leader
        round_id = self._open_round_id(is_leader)
        self.last_round_id = round_id
        my_path = self._state_file_path(round_id)
        self.backend.put(my_path, self._codec.encode(round_.local))

        merged_path = self._merged_file_path(round_id)
        if is_leader:
            expected_lanes = set(round_.expected_lanes)
            expected_ranks = set(round_.expected_ranks)
            lane_mapping = {
                int(dp_id): {int(lane) for lane in lanes}
                for dp_id, lanes in round_.local["world"]["mapping"].items()
            }
            seen: dict[str, dict[str, Any]] = {}
            covered: set[int] = set()
            rank_groups: dict[int, int] = {}
            covered_by_rank: dict[int, set[int]] = {}

            def _have_full_coverage() -> bool:
                self._read_states_for_round(round_id, seen=seen, covered=covered)
                for state in seen.values():
                    world = state["world"]
                    rank = int(world["global_rank"])
                    dp_id = int(world["dp_group_id"])
                    shard_mapping = {
                        int(group): {int(lane) for lane in lanes}
                        for group, lanes in world["mapping"].items()
                    }
                    if shard_mapping != lane_mapping or dp_id not in lane_mapping:
                        raise RuntimeError(f"Lane mapping mismatch for rank {rank}")
                    if rank_groups.setdefault(rank, dp_id) != dp_id:
                        raise RuntimeError(
                            f"DP group mismatch across rank {rank}'s shards"
                        )
                    covered_by_rank.setdefault(rank, set()).update(
                        int(lane) for lane in state["progress"]
                    )

                # Workers partition a rank's lanes into disjoint, nonempty
                # subsets. Per-rank coverage therefore waits for every active
                # worker, without counting idle workers or letting an MP peer
                # satisfy a slow worker's lane coverage.
                return (
                    set(covered_by_rank) == expected_ranks
                    and expected_lanes.issubset(covered)
                    and all(
                        covered_by_rank[rank] == lane_mapping[rank_groups[rank]]
                        for rank in expected_ranks
                    )
                )

            if not self._wait_until(_have_full_coverage, self._timeout_s):
                ranks_seen = {int(s["world"]["global_rank"]) for s in seen.values()}
                missing_ranks = sorted(expected_ranks - ranks_seen)
                missing_lanes = sorted(expected_lanes - covered)
                missing_lanes_by_rank = {
                    rank: sorted(lane_mapping[rank_groups[rank]] - lanes)
                    for rank, lanes in sorted(covered_by_rank.items())
                    if lane_mapping[rank_groups[rank]] - lanes
                }
                raise RuntimeError(
                    f"[PID {os.getpid()}] Aggregation timeout after "
                    f"{self._timeout_s}s. Missing ranks={missing_ranks}; "
                    f"missing lanes={missing_lanes}; "
                    f"missing lanes per rank={missing_lanes_by_rank}; "
                    f"files={len(self._list_state_files(round_id))}"
                )

            # States already cached from polling above.
            merged = merge(list(seen.values()))
            assert not self.backend.exists(merged_path)
            merged_bytes = self._codec.encode(merged)
            self.backend.put(merged_path, merged_bytes)
            # Roundtrip so the leader's return value is byte-identical to
            # what followers read back (json would otherwise stringify keys).
            merged = self._codec.decode(merged_bytes)
            self.backend.delete(my_path)
            # Safe: per-rank lane coverage means every active worker has read
            # round.current and published its shard before we close the round.
            self.backend.delete(self.round_file)

            # Shard paths identify every active rank/worker in this round.
            # Only retain the result for cleanup after all readers acknowledge;
            # on timeout it must remain available to delayed followers.
            ack_paths = [f"{path}.ack" for path in seen if path != my_path]
            if not self._wait_until(
                lambda: all(self.backend.exists(path) for path in ack_paths),
                self._timeout_s,
            ):
                missing = [path for path in ack_paths if not self.backend.exists(path)]
                raise RuntimeError(
                    f"[PID {os.getpid()}] Timed out after {self._timeout_s}s waiting for checkpoint acknowledgements: {missing}"
                )
            for path in ack_paths:
                self.backend.delete(path)
            if self._previous_merged_file is not None:
                self.backend.delete(self._previous_merged_file)
            self._previous_merged_file = merged_path
            return merged

        # Followers: wait for merged file
        ok = self._wait_until(lambda: self.backend.exists(merged_path), self._timeout_s)
        if not ok:
            raise RuntimeError(
                f"[PID {os.getpid()}] Timed out after {self._timeout_s}s waiting for merged checkpoint at {merged_path}"
            )
        merged = self._read_aggregation_state(merged_path)
        if merged is None:
            raise RuntimeError(f"Failed to read merged checkpoint {merged_path}")
        self.backend.put(f"{my_path}.ack", b"1")
        self.backend.delete(my_path)
        return merged

    def prepare_restore(self) -> None:
        self.backend.mkdir(self.agg_dir, parents=True, exist_ok=True)

    def cleanup(self) -> None:
        if self._previous_merged_file is not None:
            self.backend.delete(self._previous_merged_file)
            self._previous_merged_file = None
