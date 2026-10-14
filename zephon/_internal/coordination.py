"""Cross-rank checkpoint aggregation.

The engine builds local snapshots and supplies the merge and content validation
logic. The coordinator checks participant and lane coverage before merging.
A :class:`CheckpointCoordinator` exchanges those snapshots and the merged result
between participants. :class:`FileCoordinator` uses a shared ``aggregate_dir``.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeVar, cast

from zephon._internal.checkpoint import AggregationCodec
from zephon._internal.io.storage import RouterStorageBackend
from zephon.options import DEFAULT_RUN_ID, RuntimeOptions

T = TypeVar("T")

_RELOAD_SUFFIX = re.compile(r"-(\d+)$")


@dataclass(frozen=True)
class AggregationRound:
    """One rank/worker's snapshot and the required rank and lane coverage."""

    local: dict[str, Any]
    """Local engine state with ``world.global_rank``, ``world.dp_group_id``,
    ``world.mapping`` (DP group to lanes), and owned lane keys in ``progress``.
    """

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
        decoded the result; no caller barrier is needed before cleanup. A
        contributor that publishes but never acknowledges fails the leader,
        even if other followers have already returned successfully. All waits
        share one timeout budget starting at call entry; storage operations are
        not interrupted. Notify followers of leader failures before publication
        on a best-effort basis; a published result remains readable. Raise on
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
        """Release retained coordination resources that are safe to discard.

        Calls must be idempotent and leave other runs' resources intact. Never
        remove a result that a participant may still need after a failed exchange.
        This is not a guarantee to reclaim every artifact of a failed round.
        """
        ...


class _CoverageTracker:
    """Validates shards and tracks each rank's lane coverage for one round."""

    def __init__(self, round_: AggregationRound) -> None:
        self.expected_ranks = set(round_.expected_ranks)
        self.expected_lanes = set(round_.expected_lanes)
        self.lane_mapping = {
            int(dp_id): {int(lane) for lane in lanes}
            for dp_id, lanes in round_.local["world"]["mapping"].items()
        }
        self.covered: set[int] = set()
        self.dp_group_by_rank: dict[int, int] = {}
        self.covered_by_rank: dict[int, set[int]] = {}

    def add(self, location: str, state: dict[str, Any]) -> None:
        world = state["world"]
        rank = int(world["global_rank"])
        dp_id = int(world["dp_group_id"])
        shard_mapping = {
            int(group): {int(lane) for lane in lanes}
            for group, lanes in world["mapping"].items()
        }
        if shard_mapping != self.lane_mapping:
            differing_groups = sorted(
                group
                for group in shard_mapping.keys() | self.lane_mapping.keys()
                if shard_mapping.get(group) != self.lane_mapping.get(group)
            )
            raise RuntimeError(
                f"Lane mapping in rank {rank}'s shard at {location} differs from the leader's: differing DP groups={differing_groups}"
            )
        if dp_id not in self.lane_mapping:
            raise RuntimeError(
                f"Unknown DP group {dp_id} for rank {rank} at {location}: expected one of {sorted(self.lane_mapping)}"
            )
        previous_group = self.dp_group_by_rank.setdefault(rank, dp_id)
        if previous_group != dp_id:
            raise RuntimeError(
                f"DP group mismatch across rank {rank}'s shards at {location}: {previous_group} != {dp_id}"
            )
        lanes = {int(lane) for lane in state["progress"]}
        unexpected = lanes - self.lane_mapping[dp_id]
        if unexpected:
            raise RuntimeError(
                f"Unexpected lanes for rank {rank}, DP group {dp_id} at {location}: {sorted(unexpected)}"
            )
        self.covered_by_rank.setdefault(rank, set()).update(lanes)
        self.covered.update(lanes)

    def complete(self) -> bool:
        # Engine._owned_lanes / _active_workers partition a rank's lanes into
        # disjoint, nonempty subsets. Coverage waits for every active worker,
        # without counting idle workers or letting an MP peer satisfy a slow
        # worker's lane coverage.
        return (
            set(self.covered_by_rank) == self.expected_ranks
            # Also catch a DP group with no participating rank.
            and self.expected_lanes.issubset(self.covered)
            and all(
                self.covered_by_rank[rank]
                == self.lane_mapping[self.dp_group_by_rank[rank]]
                for rank in self.expected_ranks
            )
        )

    def missing(self) -> str:
        missing_lanes_by_rank = {
            rank: sorted(missing)
            for rank, lanes in sorted(self.covered_by_rank.items())
            if (missing := self.lane_mapping[self.dp_group_by_rank[rank]] - lanes)
        }
        return "; ".join(
            (
                f"Missing ranks={sorted(self.expected_ranks - self.covered_by_rank.keys())}",
                f"missing lanes={sorted(self.expected_lanes - self.covered)}",
                f"missing lanes per rank={missing_lanes_by_rank}",
            )
        )


class FileCoordinator:
    """Checkpoint aggregation over files in a shared ``aggregate_dir``.

    The leader publishes a round ID, every rank writes its state, the leader
    lists the state files until every rank covers its assigned lanes, merges
    them and writes the merged state. The leader returns only after every
    follower has acknowledged decoding it. Aggregation removes temporary round
    files as they become unnecessary, retaining the consumed merged result as
    a closed-round marker until the next successful round or a cleanup call.
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
            self._poll_base = 0.5  # 500ms base poll interval, backing off to 5s
        else:
            self._timeout_s = opts.aggregate_timeout_s
            self._poll_base = 0.05  # 50ms base poll interval
            # Resolve local paths; cloud paths stay as-is
            base = str(Path(base).resolve())
        self._poll_backoff = self._is_cloud

        self._base = base.rstrip("/")
        self.backend.mkdir(self._base, parents=True, exist_ok=True)
        self.last_round_id: str | None = None
        self._previous_result: str | None = None
        self._warned_once_about_runid = False

    def _log(self, msg: str) -> None:
        worker_id, workers_per_rank = self._worker_info()
        print(
            f"[PR {self._global_rank}][PID {os.getpid()}][Worker {worker_id}/{workers_per_rank - 1}] {msg}",
            file=sys.stderr,
        )

    def _scoped_run_id(self) -> str:
        """Run ID qualified by topology (for the default run ID) and restore count."""
        run_id = self._opts.run_id
        if run_id.startswith(DEFAULT_RUN_ID):
            # If users don't provide a run ID, we want to help them not shoot themselves in the foot.
            run_id = f"{run_id}-{self._opts.canonical_replicas}-{self._world_size}-{self._plan_id}"
        # Avoid having to use a new run id with every reload
        return f"{run_id}-{self.reload_count}"

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

        using_default = self._opts.run_id.startswith(DEFAULT_RUN_ID)
        # Names (not full paths) of subdirectories in _base, used to warn
        # about potential run ID collisions when using the default run ID.
        existing_subdir_names: list[str] = []
        if using_default:
            existing_subdir_names = self.backend.listdir(self._base)

        run_id = self._scoped_run_id()
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

    # ---------- Storage ----------

    @property
    def _announcement_location(self) -> str:
        return self.round_file

    def _shard_location(self, round_id: str) -> str:
        return self._state_file_path(round_id)

    def _result_location(self, round_id: str) -> str:
        return self._merged_file_path(round_id)

    def _failure_location(self, round_id: str) -> str:
        return f"{self.agg_dir}/failed_{round_id}.txt"

    def _ack_location(self, shard_location: str) -> str:
        return f"{shard_location}.ack"

    def _ack_name(self, shard_location: str) -> str:
        # Basenames identify participants independently of how glob
        # normalizes the directory.
        return f"{os.path.basename(shard_location)}.ack"

    def _list_ack_names(self, round_id: str) -> Iterable[str]:
        # One listing per poll avoids a HEAD for every reader.
        pattern = f"{self.agg_dir}/state_r*_w*_p*_{round_id}.ckpt.ack"
        return (os.path.basename(path) for path in self.backend.glob(pattern))

    def _list_shards(self, round_id: str) -> Iterable[str]:
        return self._list_state_files(round_id)

    def _write_announcement(self, round_id: str) -> None:
        self.backend.put(self.round_file, round_id.encode("utf-8"))

    def _read_announcement(self) -> tuple[str, float] | None:
        rid = (self._read_text(self.round_file) or "").strip()
        if not rid:
            return None
        try:
            stat = self.backend.stat(self.round_file)
        except OSError:
            return None  # File doesn't exist or access denied
        return rid, time.time() - stat["mtime"]

    def _put(self, location: str, data: bytes) -> None:
        self.backend.put(location, data)

    def _read(self, location: str) -> bytes | None:
        try:
            with self.backend.open(location, "rb") as f:
                return cast(bytes, f.read())
        except Exception:
            return None

    def _exists(self, location: str) -> bool:
        return self.backend.exists(location)

    def _delete(self, location: str) -> None:
        self.backend.delete(location)

    def _delete_best_effort(self, location: str) -> None:
        try:
            self._delete(location)
        except Exception as exc:
            self._log(f"Failed to clean up checkpoint file {location}: {exc!r}")

    def _read_text(self, location: str) -> str | None:
        raw = self._read(location)
        return None if raw is None else raw.decode("utf-8")

    def _read_aggregation_state(self, location: str) -> dict | None:
        raw = self._read(location)
        if raw is None:
            return None
        try:
            return self._codec.decode(raw)
        except Exception as exc:
            self._log(f"Failed to decode aggregation state at {location}: {exc!r}")
            return None

    def _wait_until(self, pred: Callable[[], bool], timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        delay = self._poll_base
        while True:
            if pred():
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(delay, remaining))
            if self._poll_backoff:
                delay = min(delay * 1.5, 5.0)

    def _wait_value(
        self,
        supplier: Callable[[], T | None],
        timeout: float,
    ) -> T | None:
        deadline = time.monotonic() + timeout
        delay = self._poll_base
        while True:
            val = supplier()
            if val is not None:
                return val
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            time.sleep(min(delay, remaining))
            if self._poll_backoff:
                delay = min(delay * 1.5, 5.0)

    # ---------- Rounds ----------

    def _publish_new_round_id(self) -> str:
        rid = str(int(time.time() * 1e9))  # monotonic-ish
        self._write_announcement(rid)
        self.last_round_id = rid
        return rid

    def _read_open_round_id(self) -> str | None:
        announcement = self._read_announcement()
        if announcement is None:
            return None
        rid, age_s = announcement

        # This is a practically relevant guard in case users do neither supply a run id nor a fresh aggregate directory. With this check we at least avoid stale state if it is obviously old.
        if age_s > self._timeout_s + 30:
            return None

        # Guard: if we already consumed this rid, skip if our own shard for
        # this rid exists. Under fork, a child can inherit last_round_id==rid
        # from the parent before it has published; in that case we must proceed.
        if self.last_round_id is not None and rid == self.last_round_id:
            try:
                my_shard = self._shard_location(rid)
            except Exception:
                my_shard = None
            if not my_shard or self._exists(my_shard):
                return None

        # Only "open" if the result for this rid doesn't exist yet.
        if not self._exists(self._result_location(rid)):
            return rid
        return None

    def _open_round_id(self, is_leader: bool, deadline: float) -> str:
        if is_leader:
            return self._publish_new_round_id()

        rid = self._wait_value(
            self._read_open_round_id, max(0.0, deadline - time.monotonic())
        )
        if rid is None:
            raise self._timeout_error(
                deadline, f"round id at {self._announcement_location}"
            )
        return rid

    def _timeout_error(self, deadline: float, waiting_for: str) -> RuntimeError:
        """Report the time actually spent, which may differ from the budget.

        Waits share one budget per ``aggregate()`` call, and blocking storage or
        merge work is not interrupted, so a timed-out wait may have run for less
        than the budget while the whole call ran for longer.
        """
        elapsed = time.monotonic() - (deadline - self._timeout_s)
        budget = f"aggregate_timeout_s budget: {self._timeout_s}s"
        return RuntimeError(
            f"[PID {os.getpid()}] Timed out after {elapsed:.1f}s ({budget}) waiting for {waiting_for}"
        )

    def _read_states_for_round(
        self,
        round_id: str,
        *,
        seen: dict[str, dict[str, Any]],
    ) -> Iterator[tuple[str, dict[str, Any]]]:
        """Cache and yield only newly read shards so each is validated once."""
        for location in self._list_shards(round_id):
            if location in seen:
                continue
            st = self._read_aggregation_state(location)
            if st is None:
                continue
            seen[location] = st
            yield location, st

    def aggregate(
        self,
        round_: AggregationRound,
        merge: Callable[[list[dict[str, Any]]], dict[str, Any]],
    ) -> dict[str, Any]:
        deadline = time.monotonic() + self._timeout_s
        round_id = self._open_round_id(round_.is_leader, deadline)
        self.last_round_id = round_id
        if round_.is_leader:
            return self._lead(round_id, round_, merge, deadline)
        return self._follow(round_id, round_, deadline)

    def _lead(
        self,
        round_id: str,
        round_: AggregationRound,
        merge: Callable[[list[dict[str, Any]]], dict[str, Any]],
        deadline: float,
    ) -> dict[str, Any]:
        my_shard = self._shard_location(round_id)
        result = self._result_location(round_id)
        seen: dict[str, dict[str, Any]] = {}
        try:
            self._put(my_shard, self._codec.encode(round_.local))
            coverage = _CoverageTracker(round_)

            def _have_full_coverage() -> bool:
                for location, state in self._read_states_for_round(round_id, seen=seen):
                    coverage.add(location, state)
                return coverage.complete()

            if not self._wait_until(
                _have_full_coverage, max(0.0, deadline - time.monotonic())
            ):
                raise self._timeout_error(
                    deadline,
                    f"checkpoint shards: {coverage.missing()}; shards={len(seen)}",
                )

            # States already cached from polling above.
            merged = merge(list(seen.values()))
            assert not self._exists(result)
            merged_bytes = self._codec.encode(merged)
            self._put(result, merged_bytes)
        except Exception as exc:
            # Only announce failures before publication. After publication,
            # delayed followers must still be able to consume the result.
            try:
                self._put(
                    self._failure_location(round_id),
                    f"{type(exc).__name__}: {exc}".encode("utf-8"),
                )
            except Exception as notify_error:
                self._log(f"Failed to publish checkpoint failure: {notify_error!r}")
            raise
        # Roundtrip so the leader's return value is byte-identical to
        # what followers read back (json would otherwise stringify keys).
        merged = self._codec.decode(merged_bytes)
        self._delete_best_effort(my_shard)
        # Safe: per-rank lane coverage means every active worker has read the
        # announcement and published its shard before we close the round.
        self._delete(self._announcement_location)
        self._wait_for_acks(round_id, seen, my_shard, result, deadline)
        if self._previous_result is not None:
            self._delete_best_effort(self._previous_result)
        # Retain the consumed result as a closed-round marker: a follower
        # starting the next round may still have read the old announcement.
        self._previous_result = result
        return merged

    def _wait_for_acks(
        self,
        round_id: str,
        shards: Iterable[str],
        my_shard: str,
        result: str,
        deadline: float,
    ) -> None:
        own_ack = self._ack_name(my_shard)
        ack_locations = {
            self._ack_name(shard): self._ack_location(shard)
            for shard in shards
            if self._ack_name(shard) != own_ack
        }
        pending = set(ack_locations)
        acknowledged: dict[str, str] = {}

        def _all_acknowledged() -> bool:
            for name in pending.intersection(self._list_ack_names(round_id)):
                pending.remove(name)
                acknowledged[name] = ack_locations[name]
            return not pending

        try:
            if not self._wait_until(
                _all_acknowledged, max(0.0, deadline - time.monotonic())
            ):
                # Don't advance _previous_result: cleanup must preserve this
                # result for followers that have not decoded it yet.
                details = (
                    f"missing={len(pending)}",
                    f"first 10={sorted(pending)[:10]}",
                    f"Merged result retained at {result}",
                )
                raise self._timeout_error(
                    deadline,
                    f"checkpoint acknowledgements: {'; '.join(details)}",
                )
        finally:
            for location in acknowledged.values():
                self._delete_best_effort(location)

    def _follow(
        self, round_id: str, round_: AggregationRound, deadline: float
    ) -> dict[str, Any]:
        my_shard = self._shard_location(round_id)
        result = self._result_location(round_id)
        failure = self._failure_location(round_id)
        self._put(my_shard, self._codec.encode(round_.local))

        # Followers prefer a published result; a failure notification never
        # revokes it. Failure markers are isolated by the same round namespace.
        def _result_ready() -> bool:
            if self._exists(result):
                return True
            message = self._read_text(failure)
            if message is not None:
                raise RuntimeError(
                    f"Checkpoint round {round_id} failed on leader: {message}"
                )
            return False

        if not self._wait_until(_result_ready, max(0.0, deadline - time.monotonic())):
            raise self._timeout_error(deadline, f"merged checkpoint at {result}")
        merged = self._read_aggregation_state(result)
        if merged is None:
            raise RuntimeError(f"Failed to read merged checkpoint {result}")
        self._delete_best_effort(my_shard)
        self._put(self._ack_location(my_shard), b"")
        return merged

    def prepare_restore(self) -> None:
        self.backend.mkdir(self.agg_dir, parents=True, exist_ok=True)

    def cleanup(self) -> None:
        """Delete the retained result of the last fully acknowledged round.

        The engine calls this during teardown. It does not scan for failed-round
        artifacts or retry earlier best-effort deletions. Those may require
        manual cleanup once every participant has stopped; in particular, an
        unacknowledged result must remain available to delayed readers.
        """
        if self._previous_result is not None:
            self._delete(self._previous_result)
            self._previous_result = None
