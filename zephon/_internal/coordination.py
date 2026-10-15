"""Cross-rank checkpoint aggregation.

The engine builds local snapshots and supplies the merge and content validation
logic. The coordinator checks participant and lane coverage before merging.
A :class:`CheckpointCoordinator` exchanges those snapshots and the merged result
between participants. :class:`RoundCoordinator` implements the round protocol
over storage that subclasses provide; :class:`FileCoordinator` uses a shared
``aggregate_dir`` and :class:`KVCoordinator` the key-value store that
``aggregate_store`` names.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeVar, cast

from zephon._internal.checkpoint import AggregationCodec
from zephon._internal.io.storage import RouterStorageBackend
from zephon._internal.kvstore import (
    KVStore,
    open_kv_store,
    read_staged_value,
    stage_value,
)
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

    def check_ready(self) -> None:
        """Raise if this participant can't use its coordination storage.

        The engine calls this when a participant is built, if it will take part
        in rounds, so a misconfigured setup fails before training starts rather
        than at the first checkpoint.
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


class RoundCoordinator(ABC):
    """Checkpoint rounds over a storage backend that subclasses provide.

    The leader announces a round ID, every active rank/worker publishes its
    state shard, and the leader collects shards until every rank covers its
    assigned lanes. It then merges them, publishes the result, and returns only
    after every follower has acknowledged decoding it. If the leader fails
    before publishing, it publishes a failure notice so followers don't wait
    out the timeout. Temporary round data is removed as it becomes unnecessary;
    the consumed result is retained as a closed-round marker until the next
    successful round or a cleanup call.

    Subclasses map round artifacts to storage locations and implement reading,
    writing, deleting, and discovering them. They may also override how the
    result and a failure notice are published, waited for, and read, for
    example to keep both in one conditionally written value.
    """

    def __init__(
        self,
        opts: RuntimeOptions,
        *,
        global_rank: int,
        world_size: int,
        plan_id: str,
        worker_info: Callable[[], tuple[int, int]],
        timeout_s: float,
        poll_base_s: float,
        poll_backoff: bool,
    ) -> None:
        self._opts = opts
        self._global_rank = global_rank
        self._world_size = world_size
        self._plan_id = plan_id
        self._worker_info = worker_info
        self._timeout_s = timeout_s
        self._poll_base = poll_base_s
        self._poll_backoff = poll_backoff
        self._codec = AggregationCodec(
            serializer=opts.aggregate_serializer,
            compressor=opts.aggregate_compressor,
        )
        self.last_round_id: str | None = None
        self.reload_count = 0
        self._previous_result: str | None = None
        self._previous_result_pid: int | None = None

    # ---------- Storage, implemented by subclasses ----------

    @property
    @abstractmethod
    def _announcement_location(self) -> str:
        """Where the leader announces the open round."""

    @abstractmethod
    def _shard_location(self, round_id: str) -> str:
        """Where this process publishes its state shard for ``round_id``."""

    @abstractmethod
    def _result_location(self, round_id: str) -> str: ...

    @abstractmethod
    def _failure_location(self, round_id: str) -> str: ...

    @abstractmethod
    def _ack_location(self, shard_location: str) -> str:
        """Where the publisher of ``shard_location`` acknowledges the result."""

    @abstractmethod
    def _ack_name(self, shard_location: str) -> str:
        """A name identifying that publisher's ack, unique within the round."""

    @abstractmethod
    def _acknowledged(self, round_id: str, pending: set[str]) -> set[str]:
        """Return the names in ``pending`` whose acks have been published."""

    @abstractmethod
    def _shard_rank(self, shard_location: str) -> int:
        """The global rank of the participant that published ``shard_location``."""

    @abstractmethod
    def _list_shards(self, round_id: str) -> Iterable[str]:
        """Locations of the shards published so far for ``round_id``."""

    @abstractmethod
    def _write_announcement(self, round_id: str) -> None: ...

    @abstractmethod
    def _read_announcement(self) -> tuple[str, float] | None:
        """Return the announced round ID and its age in seconds, if any."""

    @abstractmethod
    def _put(self, location: str, data: bytes) -> None: ...

    @abstractmethod
    def _read(self, location: str) -> bytes | None:
        """Return the value at ``location``, or None if it can't be read."""

    @abstractmethod
    def _exists(self, location: str) -> bool: ...

    @abstractmethod
    def _delete(self, location: str) -> None: ...

    # ---------- Storage with defaults that subclasses may override ----------

    def _publish_shard(self, round_id: str, data: bytes) -> None:
        """Publish this process's shard where ``_list_shards`` will find it."""
        self._put(self._shard_location(round_id), data)

    def _read_many(self, locations: list[str]) -> list[bytes | None]:
        """Read several values; override to batch them."""
        return [self._read(location) for location in locations]

    def _wait_for_announcement(self, timeout: float) -> str | None:
        """Poll until a joinable round is announced; return None on timeout."""
        return self._wait_value(self._read_open_round_id, timeout)

    def _close_round(self, round_id: str) -> None:
        """Stop new participants from joining once every shard is in."""
        self._delete(self._announcement_location)

    # ---------- Round outcome, overridable as a unit ----------

    def _publish_result(self, round_id: str, data: bytes) -> None:
        result = self._result_location(round_id)
        assert not self._exists(result)
        self._put(result, data)

    def _publish_failure(self, round_id: str, message: str) -> None:
        self._put(self._failure_location(round_id), message.encode("utf-8"))

    def _wait_for_outcome(self, round_id: str, timeout: float) -> bool:
        """Wait for the round's result; return False on timeout.

        Raise if the leader reported a failure instead. A published result
        takes precedence over a failure notice, which never revokes it.
        """
        result = self._result_location(round_id)
        failure = self._failure_location(round_id)

        def _result_ready() -> bool:
            if self._exists(result):
                return True
            message = self._read_text(failure)
            if message is not None:
                raise RuntimeError(
                    f"Checkpoint round {round_id} failed on leader: {message}"
                )
            return False

        return self._wait_until(_result_ready, timeout)

    def _read_result(self, round_id: str) -> dict | None:
        return self._read_aggregation_state(self._result_location(round_id))

    # ---------- Shared helpers ----------

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

    def _delete_best_effort(self, location: str) -> None:
        try:
            self._delete(location)
        except Exception as exc:
            self._log(f"Failed to clean up checkpoint file {location}: {exc!r}")

    def _read_text(self, location: str) -> str | None:
        raw = self._read(location)
        return None if raw is None else raw.decode("utf-8")

    def _read_aggregation_state(self, location: str) -> dict | None:
        return self._decode_state(location, self._read(location))

    def _decode_state(self, location: str, raw: bytes | None) -> dict | None:
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

        rid = self._wait_for_announcement(max(0.0, deadline - time.monotonic()))
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
        own_name: str,
    ) -> Iterator[tuple[str, dict[str, Any]]]:
        """Cache and yield only newly read shards so each is validated once.

        Skips the leader's own shard by participant name rather than location,
        since a listing may spell the location differently.
        """
        new = [
            loc
            for loc in self._list_shards(round_id)
            if loc not in seen and self._ack_name(loc) != own_name
        ]
        for location, raw in zip(new, self._read_many(new)):
            st = self._decode_state(location, raw)
            if st is None:
                continue
            publisher = self._shard_rank(location)
            if int(st["world"]["global_rank"]) != publisher:
                raise RuntimeError(
                    f"Shard at {location} carries global_rank={st['world']['global_rank']} but was published by rank {publisher}"
                )
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
            encoded = self._codec.encode(round_.local)
            self._publish_shard(round_id, encoded)
            coverage = _CoverageTracker(round_)
            # Use the leader's own shard without reading it back, keeping the
            # codec roundtrip so it matches what followers' shards decode to.
            seen[my_shard] = self._codec.decode(encoded)
            coverage.add(my_shard, seen[my_shard])
            own_name = self._ack_name(my_shard)

            def _have_full_coverage() -> bool:
                for location, state in self._read_states_for_round(
                    round_id, seen=seen, own_name=own_name
                ):
                    coverage.add(location, state)
                return coverage.complete()

            if not self._wait_until(
                _have_full_coverage, max(0.0, deadline - time.monotonic())
            ):
                raise self._timeout_error(
                    deadline,
                    f"checkpoint shards: {coverage.missing()}; shards={len(seen)}",
                )

            # Safe: per-rank lane coverage means every active worker has read
            # the announcement and published its shard. Closing the round
            # before publishing means no follower can still be reading the
            # announcement while it's deleted.
            self._close_round(round_id)
            # States already cached from polling above.
            merged = merge(list(seen.values()))
            merged_bytes = self._codec.encode(merged)
            self._publish_result(round_id, merged_bytes)
        except Exception as exc:
            # Only announce failures before publication. After publication,
            # delayed followers must still be able to consume the result.
            try:
                self._publish_failure(round_id, f"{type(exc).__name__}: {exc}")
            except Exception as notify_error:
                self._log(f"Failed to publish checkpoint failure: {notify_error!r}")
            raise
        # Roundtrip so the leader's return value is byte-identical to
        # what followers read back (json would otherwise stringify keys).
        merged = self._codec.decode(merged_bytes)
        self._delete_best_effort(my_shard)
        self._wait_for_acks(round_id, seen, my_shard, result, deadline)
        if self._previous_result is not None:
            self._delete_best_effort(self._previous_result)
        # Retain the consumed result as a closed-round marker: a follower
        # starting the next round may still have read the old announcement.
        self._previous_result = result
        self._previous_result_pid = os.getpid()
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
            for name in self._acknowledged(round_id, pending):
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
        self._publish_shard(round_id, self._codec.encode(round_.local))
        if not self._wait_for_outcome(round_id, max(0.0, deadline - time.monotonic())):
            raise self._timeout_error(deadline, f"merged checkpoint at {result}")
        merged = self._read_result(round_id)
        if merged is None:
            raise RuntimeError(f"Failed to read merged checkpoint {result}")
        self._delete_best_effort(my_shard)
        self._put(self._ack_location(my_shard), b"ok")
        return merged

    def check_ready(self) -> None:
        """Nothing to check beyond what construction already did."""

    def prepare_restore(self) -> None:
        """Nothing to prepare: the round namespace includes ``reload_count``."""

    def cleanup(self) -> None:
        """Delete the retained result of the last fully acknowledged round.

        The engine calls this during teardown. It does not scan for failed-round
        artifacts or retry earlier best-effort deletions. Those may require
        manual cleanup once every participant has stopped; in particular, an
        unacknowledged result must remain available to delayed readers.
        """
        # A forked child inherits this record but never owned the result.
        if (
            self._previous_result is not None
            and self._previous_result_pid == os.getpid()
        ):
            self._delete(self._previous_result)
        self._previous_result = None


class FileCoordinator(RoundCoordinator):
    """Checkpoint rounds over files in a shared ``aggregate_dir``.

    Shards are discovered by listing the round's state files, whose names
    include the rank, DataLoader worker, and PID of their publisher.
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
                    "RuntimeOptions.aggregate_dir or RuntimeOptions.aggregate_store must be set for multi-node checkpoint aggregation."
                )
            base = opts.aggregate_dir

        self.backend = RouterStorageBackend()

        # Detect cloud storage and adjust timeouts for higher latency
        base = str(base)  # Handle Path objects
        self._is_cloud = self.backend.is_cloud_path(base)
        if self._is_cloud:
            # Floor: guard against user-provided values that are too low for cloud I/O.
            timeout_s = max(opts.aggregate_timeout_s, 90.0)
            poll_base_s = 0.5  # 500ms base poll interval, backing off to 5s
        else:
            timeout_s = opts.aggregate_timeout_s
            poll_base_s = 0.05  # 50ms base poll interval
            # Resolve local paths; cloud paths stay as-is
            base = str(Path(base).resolve())

        super().__init__(
            opts,
            global_rank=global_rank,
            world_size=world_size,
            plan_id=plan_id,
            worker_info=worker_info,
            timeout_s=timeout_s,
            poll_base_s=poll_base_s,
            poll_backoff=self._is_cloud,
        )
        self._base = base.rstrip("/")
        self.backend.mkdir(self._base, parents=True, exist_ok=True)
        self._warned_once_about_runid = False

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

    # ---------- RoundCoordinator storage ----------

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

    def _acknowledged(self, round_id: str, pending: set[str]) -> set[str]:
        # One listing per poll avoids a HEAD for every reader.
        pattern = f"{self.agg_dir}/state_r*_w*_p*_{round_id}.ckpt.ack"
        return pending & {os.path.basename(path) for path in self.backend.glob(pattern)}

    def _shard_rank(self, shard_location: str) -> int:
        return int(os.path.basename(shard_location).split("_")[1][1:])

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

    def prepare_restore(self) -> None:
        self.backend.mkdir(self.agg_dir, parents=True, exist_ok=True)


class KVCoordinator(RoundCoordinator):
    """Checkpoint rounds over a :class:`~zephon._internal.kvstore.KVStore`.

    Keys live under ``zephon/{run_id}-{reload_count}``. Restoring the same
    checkpoint twice reuses that namespace, so isolation between such
    attempts is best effort, as with ``aggregate_dir``. Per round:

    - ``round``: the open round ID and a timestamp.
    - ``{round}/state/{participant}``: each participant's shard, where a
      participant is one rank, DataLoader worker, and process.
    - ``{round}/participants``: participant IDs, appended after each shard is
      written, so the leader finds shards without listing keys.
    - ``{round}/outcome``: the merged result or the leader's failure, written
      conditionally so a later failure can't replace a published result.
    - ``{round}/ack/{participant}``: each follower's acknowledgement.

    Shards, acknowledgements and the result are stored as manifests (see
    :func:`~zephon._internal.kvstore.stage_value`): a value too large for one
    write is split into chunks that are written first and deleted with it.
    """

    _RESULT = b"R"
    _FAILURE = b"F"
    # Far below any write limit, even at four UTF-8 bytes per character.
    _MAX_FAILURE_CHARS = 64 * 1024

    def __init__(
        self,
        opts: RuntimeOptions,
        store: KVStore,
        *,
        global_rank: int,
        world_size: int,
        plan_id: str,
        worker_info: Callable[[], tuple[int, int]],
    ) -> None:
        super().__init__(
            opts,
            global_rank=global_rank,
            world_size=world_size,
            plan_id=plan_id,
            worker_info=worker_info,
            timeout_s=opts.aggregate_timeout_s,
            poll_base_s=store.poll_interval_s,
            poll_backoff=False,
        )
        self._store = store
        # Chunk keys of the values this process published, deleted with them.
        self._chunks: dict[str, list[str]] = {}

    @property
    def _namespace(self) -> str:
        return f"zephon/{self._scoped_run_id()}"

    @property
    def _participant(self) -> str:
        worker_id, _ = self._worker_info()
        return f"r{self._global_rank}_w{worker_id}_p{os.getpid()}"

    def _participants_key(self, round_id: str) -> str:
        return f"{self._namespace}/{round_id}/participants"

    def _state_key(self, round_id: str, participant: str) -> str:
        return f"{self._namespace}/{round_id}/state/{participant}"

    def check_ready(self) -> None:
        self._store.check_ready()

    # ---------- RoundCoordinator storage ----------

    @property
    def _announcement_location(self) -> str:
        return f"{self._namespace}/round"

    def _shard_location(self, round_id: str) -> str:
        return self._state_key(round_id, self._participant)

    def _result_location(self, round_id: str) -> str:
        return f"{self._namespace}/{round_id}/outcome"

    def _failure_location(self, round_id: str) -> str:
        return self._result_location(round_id)

    def _ack_location(self, shard_location: str) -> str:
        round_dir, _, participant = shard_location.rpartition("/state/")
        return f"{round_dir}/ack/{participant}"

    def _ack_name(self, shard_location: str) -> str:
        return shard_location.rpartition("/state/")[2]

    def _acknowledged(self, round_id: str, pending: set[str]) -> set[str]:
        keys = {name: f"{self._namespace}/{round_id}/ack/{name}" for name in pending}
        if self._store.wait(list(keys.values()), 0):
            return set(pending)
        return {name for name, key in keys.items() if self._store.exists(key)}

    def _shard_rank(self, shard_location: str) -> int:
        return int(self._ack_name(shard_location).split("_")[0][1:])

    def _list_shards(self, round_id: str) -> list[str]:
        raw = self._store.get(self._participants_key(round_id))
        participants = raw.decode("utf-8").split() if raw else []
        return [self._state_key(round_id, p) for p in participants]

    def _publish_shard(self, round_id: str, data: bytes) -> None:
        self._put(self._shard_location(round_id), data)
        # Appended on the same connection after the shard, so a leader that
        # sees this participant can already read its shard.
        self._store.append(
            self._participants_key(round_id), f"{self._participant}\n".encode()
        )

    def _close_round(self, round_id: str) -> None:
        super()._close_round(round_id)
        self._delete_best_effort(self._participants_key(round_id))

    def _write_announcement(self, round_id: str) -> None:
        payload = json.dumps({"round": round_id, "ts": time.time()})
        self._store.set(self._announcement_location, payload.encode("utf-8"))

    def _read_announcement(self) -> tuple[str, float] | None:
        raw = self._store.get(self._announcement_location)
        if raw is None:
            return None
        payload = json.loads(raw)
        return payload["round"], time.time() - payload["ts"]

    def _put(self, location: str, data: bytes) -> None:
        staged = stage_value(self._store, location, data)
        self._store.set(location, staged.manifest)
        self._chunks[location] = staged.chunk_keys

    def _read(self, location: str) -> bytes | None:
        return self._read_many([location])[0]

    def _read_many(self, locations: list[str]) -> list[bytes | None]:
        return [
            None if manifest is None else read_staged_value(self._store, loc, manifest)
            for loc, manifest in zip(locations, self._store.get_many(locations))
        ]

    def _exists(self, location: str) -> bool:
        return self._store.exists(location)

    def _delete(self, location: str) -> None:
        # Manifest first, so no reader finds it with chunks missing.
        self._store.delete(location)
        for chunk_key in self._chunks.pop(location, ()):
            self._store.delete(chunk_key)

    # ---------- Native waits and a single outcome value ----------

    def _wait_for_announcement(self, timeout: float) -> str | None:
        deadline = time.monotonic() + timeout
        while (rid := self._read_open_round_id()) is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            if not self._store.exists(self._announcement_location):
                self._store.wait([self._announcement_location], remaining)
            else:
                # Stale or already joined; wait for the leader to replace it.
                time.sleep(min(self._poll_base, remaining))
        return rid

    def _publish_result(self, round_id: str, data: bytes) -> None:
        outcome_key = self._result_location(round_id)
        staged = stage_value(self._store, outcome_key, data)
        outcome = self._RESULT + staged.manifest
        # The manifest's write ID is unique, so a match means this call won.
        if self._store.set_if_absent(outcome_key, outcome) != outcome:
            for chunk_key in staged.chunk_keys:
                self._delete_best_effort(chunk_key)
            raise RuntimeError(
                f"Round {round_id} already has an outcome at {outcome_key}"
            )
        self._chunks[outcome_key] = staged.chunk_keys

    def _publish_failure(self, round_id: str, message: str) -> None:
        # Truncate by character so the notice stays valid UTF-8.
        notice = message[: self._MAX_FAILURE_CHARS].encode("utf-8")
        self._store.set_if_absent(
            self._result_location(round_id), self._FAILURE + notice
        )

    def _wait_for_outcome(self, round_id: str, timeout: float) -> bool:
        outcome_key = self._result_location(round_id)
        if not self._store.wait([outcome_key], timeout):
            return False
        outcome = self._store.get(outcome_key) or b""
        if outcome[:1] == self._FAILURE:
            raise RuntimeError(
                f"Checkpoint round {round_id} failed on leader: {outcome[1:].decode('utf-8')}"
            )
        return True

    def _read_result(self, round_id: str) -> dict | None:
        outcome_key = self._result_location(round_id)
        outcome = self._store.get(outcome_key)
        if not outcome or outcome[:1] != self._RESULT:
            return None
        return self._decode_state(
            outcome_key, read_staged_value(self._store, outcome_key, outcome[1:])
        )


def make_coordinator(
    opts: RuntimeOptions,
    *,
    global_rank: int,
    world_size: int,
    plan_id: str,
    worker_info: Callable[[], tuple[int, int]],
) -> CheckpointCoordinator:
    """Build the coordinator that ``aggregate_dir``/``aggregate_store`` select."""
    if opts.aggregate_store:
        if opts.aggregate_dir:
            raise ValueError(
                "Set at most one of RuntimeOptions.aggregate_dir and "
                "RuntimeOptions.aggregate_store."
            )
        return KVCoordinator(
            opts,
            open_kv_store(opts.aggregate_store, timeout_s=opts.aggregate_timeout_s),
            global_rank=global_rank,
            world_size=world_size,
            plan_id=plan_id,
            worker_info=worker_info,
        )
    return FileCoordinator(
        opts,
        global_rank=global_rank,
        world_size=world_size,
        plan_id=plan_id,
        worker_info=worker_info,
    )
