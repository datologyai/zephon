# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Runtime execution engine that wires plans to concrete stage runners.

Canonical scheduling and mapping strategies
-------------------------------------------
The engine builds a canonical schedule over a fixed number of logical data
parallel replicas (``canonical_replicas``). At runtime, those replicas are
mapped to the set of physical ranks in this run via a mapping strategy. This
decoupling enables elastic continuation: we can resume with a different number
of ranks while keeping the canonical order.
Two built-in mapping strategies are supported via ``RuntimeOptions.mapping_strategy``:
- "contiguous" (default)
  - Assigns each rank a contiguous block of canonical replicas.
    Example: canonical_replicas=8, num_ranks=3 → rank→lanes:
      r0: [0,1,2], r1: [3,4,5], r2: [6,7]
  - Why: maximizes locality for chunk/block caches and minimizes cross-rank
    interleaving. Useful when shards/blocks benefit from staying together.
- "interleaved"
  - Assigns replica i to rank (i % num_ranks). Example with 8 and 3 →
    r0: [0,3,6], r1: [1,4,7], r2: [2,5]
  - Why: balances progress more evenly across ranks and often reduces tail
    variance. Helpful when ranks scale up/down between phases or when shards
    are heterogeneous.
Both strategies are deterministic and equivalent with respect to the canonical
order; only the per-rank partitioning differs. BOTH strategies will lead to identical
global batches the mapping strategy only decides which rank processes which parts.
The engine also inserts a lane-merging distributor at the tail only when a rank owns multiple replicas
and batching is present, so 1:1 rank↔replica runs remain a clean, single-lane dataflow.
"""

import atexit
import json
import math
import os
import re
import sys
import tempfile
import time
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Literal, TypeVar

T = TypeVar("T")

from zephon.core.constants import (
    ChunkId,
    EngineSample,
    LaneId,
    LanePtr,
    SampleBatch,
    SampleId,
    SampleRecord,
)
from zephon.core.graph import Plan
from zephon.core.world import World
from zephon.io.options import StoreOptions
from zephon.runners.threads import ThreadStageRunner
from zephon.work import (
    MixtureReadConfig,
    WorkSource,
)
from zephon.work.base import WorkChunk


def inside_torch_worker() -> bool:
    """Return True if the current process is a PyTorch DataLoader worker."""
    try:
        from torch.utils.data import get_worker_info

        return get_worker_info() is not None
    except Exception:
        return False


def get_torch_worker_info() -> tuple[int, int]:
    """Return (worker_id, workers_per_rank) if torch is available; else (0,1)."""
    try:
        from torch.utils.data import get_worker_info

        info = get_worker_info()
        if info is None:
            return 0, 1
        return int(info.id), int(info.num_workers)
    except Exception:
        return 0, 1


def _extract_lane_id(item: SampleRecord | SampleBatch) -> int:
    if isinstance(item, SampleRecord):
        return int(item.meta.lane_id)
    assert isinstance(item, SampleBatch)
    lids = item.lane_ids
    if not lids:
        raise RuntimeError("Empty batch has no lane_id")
    # Pipeline guarantees one lane per batch at the tail
    return int(lids[0])


def _call_engine_clean_merged(engine: "Engine"):
    """This is a helper for a really strange observation, described below.

    In irregular frequencies, we run into
     File "/Users/mboether/dev/zephon/zephon/core/engine.py",
     line 202, in __init__ atexit.register(self._clean_merged)
     ^^^^^^^^^^^^^^^^^^ AttributeError: 'Engine' object has no attribute '_clean_merged'
    Errors. This safe-guards against that.
    """
    try:
        # only look up the attribute at exit time, not at registration time
        cm = getattr(engine, "_clean_merged", None)
        if callable(cm):
            cm()
    except Exception:
        pass


DEFAULT_RUN_ID = "default_run_id"
_RELOAD_SUFFIX = re.compile(r"-(\d+)$")


@dataclass
class RuntimeOptions:
    """User-tunable knobs that influence how the engine constructs runners."""

    runner: str | None = None  # Default runner. Typically auto-inferred.
    run_id: str = DEFAULT_RUN_ID
    per_stage_runner: dict[int, str] = field(
        default_factory=dict
    )  # Manual override for runner per-stage. Mostly useful for debugging and advanced usage.
    allow_subprocess_in_worker: bool = False  # TODO(MaxiBoether): Implement this.
    mp_context: Any = None  # TODO(MaxiBoether): Do we need this?
    worker_allocation: Literal[
        "fit_to_ops", "per_stage_fixed", "global", "autotune"
    ] = "fit_to_ops"
    stage_weighting: Literal["equal", "by_declared_parallelism"] = (
        "by_declared_parallelism"
    )
    max_workers: int = (
        8  # max_workers per stage OR global, depending on worker_allocation.
    )
    deterministic: bool = True
    prefetch_batches: int | None = None
    default_stage_prefetch: int = 0
    per_stage_prefetch: dict[int, int] = field(default_factory=dict)
    mixture_config: MixtureReadConfig | None = None
    io_options: StoreOptions = field(default_factory=StoreOptions)
    # Expert knob:
    # Keep latency-based flush in deterministic mode when True unless a stage contains
    # a batch-shape sensitive operator (in which case we auto-disable it for that stage).
    # When False, latency flush is always disabled in deterministic mode.
    allow_latency_flush_in_deterministic: bool = True
    # Canonical number of replicas (logical DP). If None, derives from num_ranks (TODO(MaxiBoether): this breaks if num_ranks > dp. Check when supporting 3D parallelism.)
    canonical_replicas: int | None = None
    # How to map canonical replicas to physical ranks:
    # - 'contiguous': ranks own contiguous blocks of replicas (locality-friendly)
    # - 'interleaved': replicas are round-robin across ranks (balanced progress)
    mapping_strategy: Literal["contiguous", "interleaved"] | None = None
    # TODO(MaxiBoether): This should be obtained from the current environment (e.g., torchtitan). Maybe not part of RuntimeOptions but rather part of init? Runtime options describe logical options of pipeline.
    num_ranks: int = 1
    physical_rank: int = 0
    # Autotune placeholders (intentionally not implemented yet)
    autotune_config: dict[str, Any] | None = None  # e.g., {"target_util": 0.3, ...}
    # Where all workers/ranks dump their local state. Must be shared (e.g., NFS) for multi-node.
    aggregate_dir: str | None = None
    # How long to wait for all contributors and for the merged file.
    aggregate_timeout_s: float = 30.0


class Engine:
    """Bind a `Plan` to concrete runners and orchestrate streaming execution."""

    def __init__(self, plan: Plan, opts: RuntimeOptions, work: WorkSource) -> None:
        """Initialize stage runners and prepare to stream work items."""
        self._plan = plan
        base_ctx: dict[str, Any] = {
            "datasets_by_id": work.datasets_by_id,
            "io_options": opts.io_options,
        }
        self._ctx = base_ctx
        self._opts = opts
        self._world = self._build_world()
        self.inflight_chunks_per_lane: dict[LaneId, dict[ChunkId, Any]] = defaultdict(
            dict
        )
        self._lane_progress: dict[LaneId, LanePtr] = defaultdict(LanePtr)
        # When non-None, we are replaying; drop outputs for that lane until we pass this pointer.
        self._replay_until: dict[LaneId, LanePtr] | None = None
        self._replay_seen: dict[LaneId, LanePtr] | None = None

        self._lane_next_cid: dict[LaneId, int] = defaultdict(int)
        self._warned_once_about_runid = False
        self._checkpoint_reload_count = 0

        self._work = work
        self._lane_ws: dict[LaneId, WorkSource] = {}
        for lane in self._world.lanes_for_rank[self._world.physical_rank]:
            self._lane_ws[lane] = self._work.clone_for_lane(
                lane, canonical_replicas=self._world.canonical_replicas
            )

        # Stage runners are stored heterogeneously.
        self._runners: list[ThreadStageRunner] = []
        self._build_runners()

        self._agg_timeout_s = self._opts.aggregate_timeout_s
        self._using_fresh_tmp = False
        if self._world.num_ranks == 1:
            # Single-node: auto if not provided
            base = self._opts.aggregate_dir or os.path.join(
                tempfile.gettempdir(), f"zephon_state_r{self._world.physical_rank}"
            )
            self._using_fresh_tmp = (
                self._opts.aggregate_dir is None or self._opts.aggregate_dir == ""
            )
        else:
            # Multi-node: must be provided (shared FS)
            if not self._opts.aggregate_dir:
                raise RuntimeError(
                    "RuntimeOptions.aggregate_dir must be set for multi-node checkpoint aggregation."
                )
            base = self._opts.aggregate_dir

        self._agg_base = Path(base).resolve()
        self._agg_base.mkdir(parents=True, exist_ok=True)

        self._agg_dir.mkdir(parents=True, exist_ok=True)
        self._last_round_id: str | None = None
        self._previous_merged_file: Path | None = None

        atexit.register(_call_engine_clean_merged, self)

        bs = self._plan.batch_size_hint
        if bs is not None:
            cs = self._work.chunk_size_hint()
            assert cs is not None, (
                "No chunk size hint for current work source, determinism breaks potentially"
            )
            if (bs % cs != 0) and (cs % bs != 0):
                print(
                    "Warning! Your chunk and batch size don't align. If you plan to checkpoint and continue from a checkpoint, this currently breaks. Consider making the batch size a multiple/divisor of chunk size for now.",
                    file=sys.stderr,
                )

    @property
    def _round_file(self) -> Path:
        return self._agg_dir / "round.current"

    @property
    def _agg_dir(self) -> Path:
        def _normalize_run_dir_name(
            name: str,
        ) -> str:  # Drop a trailing "-<int>" if present
            m = _RELOAD_SUFFIX.search(name)
            return name[: m.start()] if m else name

        run_id = self._opts.run_id
        using_default = False
        if run_id.startswith(DEFAULT_RUN_ID):
            agg_dir_entries = [
                Path(e.path).resolve() for e in os.scandir(self._agg_base) if e.is_dir()
            ]
            using_default = True
            # If users don't provide a run ID, we want to help them not shoot themselves in the foot.
            run_id = f"{run_id}-{self._opts.canonical_replicas}-{self._world.num_ranks}-{self._plan.plan_id}"

        # Avoid having to use a new run id with every reload
        run_id = f"{run_id}-{self._checkpoint_reload_count}"
        current_run_dir = (self._agg_base / run_id).resolve()
        cur_norm = _normalize_run_dir_name(current_run_dir.name)
        result = current_run_dir / ".zephon_agg"

        if (
            not self._using_fresh_tmp
            and using_default
            and not self._warned_once_about_runid
            and agg_dir_entries
        ):
            if {_normalize_run_dir_name(p.name) for p in agg_dir_entries} != {cur_norm}:
                # Potentially we can also only warn if the plan id or canonical replicas change since that probably really causes a semantic change but we better just tell the user early this is not the. best idea.
                print(
                    f"Warning! No run id has been supplied. This can cause issues in checkpointing if the same aggregate_dir ({self._agg_base}) is used across multiple runs. Your current supplied directory contains data from other runs (or your world changed), which might indicate that you re-use that directory (or use it for other purposes as well). Zephon adjusted the run id to {run_id} to avoid problems, but if you choose to run exactly the same pipeline twice in the samed directory, issues might still occur without providing a run id.\n\nOffending subdirs: {agg_dir_entries}",
                    file=sys.stderr,
                )
                self._warned_once_about_runid = True

        # print(f"DEBUG: aggregation directory is {result}", file=sys.stderr)

        return result

    def __del__(self) -> None:
        try:
            self._clean_merged()
        except Exception:
            pass

    def _clean_merged(self) -> None:
        if self._previous_merged_file is not None:
            self._previous_merged_file.unlink(missing_ok=True)
            self._previous_merged_file = None

    def __getstate__(self):
        # We rather fail explicitly here for now to avoid problems with runners.
        raise RuntimeError("Engine must not be pickled; build it inside a worker.")

    def _build_world(self) -> World:
        worker_id, workers_per_rank = get_torch_worker_info()
        # Always build a canonical schedule. If canonical_replicas is unspecified,
        # default it to the number of physical DP ranks (>=1).
        num_ranks = max(1, int(self._opts.num_ranks))
        canonical_replicas = (
            int(self._opts.canonical_replicas)
            if self._opts.canonical_replicas is not None
            else num_ranks
        )
        strategy = (self._opts.mapping_strategy or "contiguous").lower()
        # Build mapping according to strategy
        mapping: dict[int, list[int]] = {}
        if strategy == "interleaved":
            for r in range(num_ranks):
                mapping[r] = [
                    lane for lane in range(canonical_replicas) if lane % num_ranks == r
                ]
        else:  # contiguous
            base = canonical_replicas // num_ranks
            rem = canonical_replicas % num_ranks
            start = 0
            for r in range(num_ranks):
                count = base + (1 if r < rem else 0)
                lanes = list(range(start, start + count))
                mapping[r] = lanes
                start += count

        return World(
            canonical_replicas=canonical_replicas,
            worker_id=worker_id,
            workers_per_rank=workers_per_rank,
            physical_rank=self._opts.physical_rank,
            num_ranks=num_ranks,
            lanes_for_rank=mapping,
        )

    def explain(self) -> str:
        """ASCII execution graph showing where buffers exist and their sizes.

        Visualizes:
        - Per-op input queues inside stages (as -[in_q=Q]-> between ops)
        - Stage output queue (as --[stage_out=Q]-->)
        - Boundary prefetch between stages (as ==[prefetch=Q]==>)
        - Final pipeline prefetch to the consumer
        """
        lines: list[str] = []

        mode = self._opts.worker_allocation
        if mode == "global":
            lines.append(
                f"Allocation=global total={self._opts.max_workers} weighting={self._opts.stage_weighting}"
            )
        elif mode == "per_stage_fixed":
            lines.append(
                f"Allocation=per_stage_fixed per_stage={self._opts.max_workers}"
            )
        else:
            lines.append("Allocation=fit_to_ops (cap=sum(node.parallelism), min 1/op)")

        for idx, (stage, runner) in enumerate(zip(self._plan.stages, self._runners)):
            runner_kind = "unknown"
            op_in_q: int | None = None
            stage_prefetch: int | None = None
            cap: int | None = None

            if isinstance(runner, ThreadStageRunner):  # pyright: ignore[reportUnnecessaryIsInstance]
                runner_kind = "threads"
                op_in_q = getattr(runner, "_queue_capacity", None)
                stage_prefetch = getattr(runner, "_prefetch_capacity", None)
                cap = getattr(runner, "_max_workers", None)  # <- show cap

            # Header with placement and runner only (buffers are shown inline)
            lines.append(
                f"Stage[{idx}] place={stage.placement} runner={runner_kind} cap={cap}"
            )

            # Inside-stage ops with input buffers between them
            nodes = [f"{nd.name}@p{nd.parallelism}" for nd in stage.nodes]
            if not nodes:
                lines.append("  [empty stage]")
            else:
                if len(nodes) == 1:
                    lines.append(f"  {nodes[0]}")
                else:
                    if op_in_q is not None:
                        connector = f" -[in_q={op_in_q}]-> "
                    else:
                        connector = " -> "
                    lines.append("  " + connector.join(nodes))

            # Stage output queue capacity follows ThreadStageRunner logic
            # out_capacity = max(1, stage_prefetch or op_in_q)
            out_q: int | None = None
            if op_in_q is not None:
                sp = stage_prefetch or 0
                out_q = max(1, sp or op_in_q)

            # Boundary: to next stage or pipeline end
            if out_q is not None:
                lines.append(f"  --[stage_out={out_q}]-->")
            else:
                lines.append("  --[stage_out=?]-->")

            is_last = idx + 1 == len(self._runners)
            if not is_last:
                if stage_prefetch and stage_prefetch > 0:
                    lines.append(f"  ==[prefetch={stage_prefetch}]==> Stage[{idx + 1}]")
                else:
                    lines.append(f"  ==> Stage[{idx + 1}]")
            else:
                final_prefetch = self._opts.prefetch_batches or 0
                if final_prefetch > 0:
                    lines.append(
                        f"  ==[final_prefetch={final_prefetch}]==> pipeline_end"
                    )
                else:
                    lines.append("  ==> pipeline_end")

        return "\n".join(lines)

    @staticmethod
    def _apportion(total: int, weights: list[int]) -> list[int]:
        """Split `total` into integer parts proportional to `weights`.

        Uses largest-remainder (Hamilton) method; guarantees sum(parts) == total.
        If all weights are non-positive, falls back to an equal split.
        """
        if total <= 0 or not weights:
            return [0] * len(weights)
        wpos = [max(0, int(w)) for w in weights]
        wsum = sum(wpos)
        n = len(wpos)
        if wsum == 0:
            base = total // n
            parts = [base] * n
            for i in range(total - base * n):
                parts[i] += 1
            return parts
        quotas = [total * (w / wsum) for w in wpos]
        floors = [int(math.floor(q)) for q in quotas]
        remaining = total - sum(floors)
        order = sorted(range(n), key=lambda i: (quotas[i] - floors[i]), reverse=True)
        for i in range(remaining):
            floors[order[i]] += 1
        return floors

    def _build_runners(self) -> None:
        """Instantiate per-stage runners according to placement and options."""
        inside_worker = inside_torch_worker()
        default_runner = self._opts.runner or "auto"
        mode = self._opts.worker_allocation
        num_stages = len(self._plan.stages)
        if mode == "autotune":
            raise NotImplementedError(
                "worker_allocation='autotune' is reserved for future auto-tuning. "
                + "Use 'per_stage_fixed' or 'global' for now."
            )

        if mode == "global":
            total = self._opts.max_workers
            if total < num_stages:
                warnings.warn(
                    f"[zephon] max_workers_total={total} < number of stages={num_stages}; "
                    + f"bumping to {num_stages} (1 thread per stage).",
                    RuntimeWarning,
                    stacklevel=2,
                )
                total = num_stages

            if self._opts.stage_weighting == "equal":
                weights = [1 for _ in range(num_stages)]
            else:  # by_declared_parallelism
                weights = [
                    max(
                        1,
                        sum(max(1, (nd.parallelism or 1)) for nd in stage.nodes),
                    )
                    for stage in self._plan.stages
                ]
            per_stage_caps = self._apportion(total, weights)
        elif mode == "per_stage_fixed":
            # per_stage_fixed: same cap for each stage
            cap = self._opts.max_workers
            per_stage_caps = [cap for _ in range(num_stages)]
        else:
            per_stage_caps = [
                max(1, sum(max(1, (nd.parallelism or 1)) for nd in stage.nodes))
                for stage in self._plan.stages
            ]

        for idx, stage in enumerate(self._plan.stages):
            cap_for_stage = per_stage_caps[idx]

            prefetch = self._opts.per_stage_prefetch.get(
                idx, self._opts.default_stage_prefetch
            )
            chosen = self._opts.per_stage_runner.get(idx, default_runner)
            if chosen == "auto":
                if inside_worker and not self._opts.allow_subprocess_in_worker:
                    chosen = "threads"
                elif stage.placement == "remote":
                    chosen = "remote"
                else:
                    chosen = "threads"
            if (
                inside_worker
                and chosen == "process"
                and not self._opts.allow_subprocess_in_worker
            ):
                chosen = "threads"
            if chosen == "threads":
                allow_latency = self._opts.allow_latency_flush_in_deterministic
                if self._opts.deterministic:
                    has_sensitive = any(
                        getattr(nd.op.traits(), "batch_shape_sensitive", False)
                        for nd in stage.nodes
                    )
                    if has_sensitive and allow_latency:
                        print(
                            "Deterministic mode: disabling time-based flush for Stage[%d] due to batch-shape sensitive op.",
                            idx,
                        )
                        allow_latency = False
                self._runners.append(
                    ThreadStageRunner(
                        stage,
                        self._ctx,
                        cap_for_stage,
                        prefetch_capacity=prefetch,
                        deterministic=self._opts.deterministic,
                        allow_latency_flush_in_deterministic=allow_latency,
                    )
                )
            elif chosen == "remote":
                raise NotImplementedError("Remote workers are not yet implemented.")
            elif chosen == "process":
                raise NotImplementedError("ProcessStageRunner not yet implemented.")
            else:
                raise ValueError(f"Unknown runner '{chosen}'")

    def _lane_stream(self, lane_id: LaneId) -> Iterator[EngineSample]:
        """Yield EngineSamples for a single lane, fetching chunks lazily.

        Maintains inflight_chunks_per_lane[lane_id][chunk_id] -> chunk_obj.
        """
        inflight_lane = self.inflight_chunks_per_lane.setdefault(lane_id, {})
        target = None
        if self._replay_until is not None:
            target = self._replay_until.get(lane_id)

        # Phase 1: replay restored inflight chunks first (ascending chunk_id)
        for cid in sorted(inflight_lane.keys()):
            chunk = inflight_lane[cid]
            # Drop strictly older chunks; skip the prefix on the pointer chunk
            if target is not None:
                if int(cid) < int(target.chunk_id):
                    continue
            for sample_id in chunk:
                # Note that we yield the _entire_ chunk here. This can break with elastic continuation in case a batch is cross-chunk boundaries.
                yield (sample_id, lane_id, int(cid))

        # Phase 2: fetch new chunks and assign stable per-lane ids
        ws = self._lane_ws[lane_id]
        while True:
            chunk = ws.next_chunk()
            if chunk is None:
                return  # lane exhausted

            cid = int(self._lane_next_cid[lane_id])
            self._lane_next_cid[lane_id] = cid + 1
            inflight_lane[cid] = chunk

            for sample_id in chunk:
                yield (sample_id, lane_id, cid)

    def _active_workers(self, num_workers: int, lanes_all: list[int]) -> int:
        L = len(lanes_all)
        a = min(num_workers, L)
        while a > 1 and (L % a) != 0:
            a -= 1
        return a  # at least 1

    def _source_stream(self) -> Iterator[EngineSample]:
        """Yield sample identifiers from the backing work source."""
        lanes_all = self._world.lanes_for_rank[
            self._world.physical_rank
        ]  # canonical order
        worker_id, workers_per_rank = get_torch_worker_info()
        active = self._active_workers(workers_per_rank, lanes_all)

        if worker_id >= active:
            return  # this worker is idle / no lanes assigned

        owned = [
            lane for idx, lane in enumerate(lanes_all) if (idx % active) == worker_id
        ]

        # Build one generator per lane
        gens: dict[LaneId, Iterator[EngineSample]] = {
            lane: self._lane_stream(lane) for lane in owned
        }

        # Round-robin over the lane generators
        rr = deque(gens.items())  # holds (lane_id, gen)
        while rr:
            lane_id, gen = rr.popleft()
            try:
                yield next(gen)  # emit one sample from this lane
                rr.append((lane_id, gen))  # put it back for fair round-robin
            except StopIteration:
                # This lane is finished; don't re-append
                pass

    def _lane_rr_iter(
        self,
        upstream: Iterable[SampleRecord | SampleBatch],
    ) -> Iterator[SampleRecord | SampleBatch]:
        """Unbounded per-lane tail mux with round-robin emission.

        - Always drain upstream into per-lane deques.
        - Always try to emit in round-robin across lanes owned by this rank.
        - If the current target lane is empty, keep pulling from upstream until it isn't,
          or until upstream ends. Warn as buffers grow large.
        """
        warn_threshold = 10000

        # Derive the lanes owned by THIS DataLoader worker (same logic as _source_stream)
        lanes_all = self._world.lanes_for_rank[self._world.physical_rank]
        worker_id, workers_per_rank = get_torch_worker_info()
        active = self._active_workers(workers_per_rank, lanes_all)

        # Idle worker: upstream will be empty, but we still drain it to let the
        # ThreadStageRunner stop cleanly (propagate stop token, join threads).
        if worker_id >= active:
            upstr = list(upstream)
            assert len(upstr) == 0
            return

        lanes = [
            lane for idx, lane in enumerate(lanes_all) if (idx % active) == worker_id
        ]
        if len(lanes) <= 1:  # Simple case: only one owned lane
            yield from upstream
            return

        buffers: dict[int, deque[SampleRecord | SampleBatch]] = {
            lane: deque() for lane in lanes
        }
        it = iter(upstream)
        upstream_ended = False
        idx = 0

        # soft warning thresholds (double each time they’re tripped)
        per_lane_next_warn: dict[int, int] = {}
        total_next_warn = warn_threshold
        lane_warn_base = warn_threshold

        def warn_after_enqueue(lane: int) -> None:
            nonlocal total_next_warn
            if lane_warn_base:
                nxt = per_lane_next_warn.get(lane, lane_warn_base)
                sz = len(buffers[lane])
                if sz >= nxt:
                    warnings.warn(
                        f"[zephon] Tail RR buffer for lane {lane} reached {sz} items.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    per_lane_next_warn[lane] = max(nxt * 2, nxt + 1)
            if total_next_warn:
                total_sz = sum(len(q) for q in buffers.values())
                if total_sz >= total_next_warn:
                    warnings.warn(
                        f"[zephon] Total Tail RR buffered items reached {total_sz}.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    total_next_warn = max(total_next_warn * 2, total_next_warn + 1)

        while not upstream_ended:
            lane = lanes[idx]
            # Block-fill until current lane has something or upstream ends.
            while not buffers[lane]:
                try:
                    item = next(it)  # blocks until data or StopIteration
                except StopIteration:
                    upstream_ended = True
                    break
                got_lane = _extract_lane_id(item)
                assert got_lane in buffers
                buffers[got_lane].append(item)
                warn_after_enqueue(got_lane)

            if upstream_ended:
                break  # move to drain regime

            # Emit exactly one from the current lane, then advance RR pointer.
            yield buffers[lane].popleft()
            idx = (idx + 1) % len(lanes)

        # -------- DRAIN REGIME: upstream ended, flush everything in RR ----------
        while any(buffers[l] for l in lanes):
            # Find next non-empty lane starting at idx (skip empties).
            rotated = 0
            emitted = False
            while rotated < len(lanes):
                lane = lanes[idx]
                if buffers[lane]:
                    yield buffers[lane].popleft()
                    idx = (idx + 1) % len(lanes)
                    emitted = True
                    break
                idx = (idx + 1) % len(lanes)
                rotated += 1
            if not emitted:
                # All empty (defensive; the outer while should break next iteration).
                break

    def build_iter(self) -> Iterator[SampleRecord | SampleBatch]:
        """Return an iterator that threads the work stream through all stages."""
        source_iter = self._source_stream()

        # Construct overall pipeline by chaining runners
        stream_iter = self._runners[0].run(source_iter)
        for runner in self._runners[1:]:
            stream_iter = runner.run(stream_iter)

        stream_iter = self._lane_rr_iter(stream_iter)

        yield from stream_iter

    def notify(
        self, lane_id: int, max_chunk_id: int, max_chunk_samples: list[SampleId]
    ) -> bool:
        """Return True if the *whole* item should be yielded to the consumer.

        Also:
        - Evict inflight chunks with cid < max_chunk_id for this lane.
        - Update per-lane training position (chunk_id, offset).
        - While in replay mode, drop items until we strictly pass the saved pointer.
        """
        # 1) Evict older inflight chunks for this lane.
        inflight_lane = self.inflight_chunks_per_lane[lane_id]
        for cid in list(inflight_lane.keys()):
            if cid < max_chunk_id:
                inflight_lane.pop(cid, None)
        add_k = len(max_chunk_samples)

        # 2) Replay gate
        if self._replay_until is not None and lane_id in self._replay_until:
            target = self._replay_until[lane_id]

            if max_chunk_id < target.chunk_id:
                # Before the checkpoint chunk: drop (don't touch lane_progress)
                return False

            if max_chunk_id == target.chunk_id:
                # Compare against a fresh, 0-based replay cursor
                rp = 0
                if self._replay_seen is not None:
                    rp = self._replay_seen.get(
                        lane_id, LanePtr(target.chunk_id, 0)
                    ).offset
                projected = rp + add_k

                if projected <= target.offset:
                    # Still before/at the saved pointer: advance replay cursor only
                    if self._replay_seen is not None:
                        self._replay_seen[lane_id] = LanePtr(target.chunk_id, projected)
                    return False

                # We cross the saved pointer on this item: switch to live
                if self._replay_seen is not None:
                    self._replay_seen.pop(lane_id, None)
                self._replay_until.pop(lane_id, None)
                if not self._replay_until:
                    self._replay_until = None

            else:  # max_chunk_id > target.chunk_id
                # Past the saved chunk: switch to live immediately
                self._replay_until.pop(lane_id, None)
                if self._replay_seen is not None:
                    self._replay_seen.pop(lane_id, None)
                if not self._replay_until:
                    self._replay_until = None
                # fall through to accept

        # 3) Live mode (or just crossed pointer): accept and advance persisted progress
        cur = self._lane_progress[lane_id]
        seen_offset = cur.offset if (cur.chunk_id == max_chunk_id) else 0
        self._lane_progress[lane_id] = LanePtr(max_chunk_id, seen_offset + add_k)
        return True

    def eval_one(self, sample: SampleId | EngineSample) -> Any:
        """Synchronously evaluate a single element through every stage runner."""
        value: Any = sample
        for runner in self._runners:
            value = runner.run_one(value)
        return value

    def close(self) -> None:
        """Close all stage runners, suppressing teardown errors."""
        for runner in self._runners:
            try:
                runner.close()
            except Exception:
                pass

    def _state_dict_local(self) -> dict[str, Any]:
        """Serializable snapshot of engine runtime state (no plan/op state)."""
        lanes_all = self._world.lanes_for_rank[self._world.physical_rank]
        worker_id, workers_per_rank = get_torch_worker_info()
        active = self._active_workers(workers_per_rank, lanes_all)

        owned = {
            lane for idx, lane in enumerate(lanes_all) if (idx % active) == worker_id
        }
        # print(f"node {self._world.physical_rank}/{self._world.num_ranks} w{worker_id}/{workers_per_rank} owns {len(owned)} lanes.")
        if worker_id >= active:
            assert len(owned) == 0

        for purge_candidate_str in [
            "_lane_ws",
            "inflight_chunks_per_lane",
            "_lane_progress",
            "_lane_next_cid",
        ]:
            # print(f"length of {purge_candidate_str} is {len(getattr(self, purge_candidate_str))}")

            for lane in list(getattr(self, purge_candidate_str)):
                if lane not in owned:
                    del getattr(self, purge_candidate_str)[lane]

        for lane in owned:
            self._lane_ws.setdefault(
                lane,
                self._work.clone_for_lane(
                    lane, canonical_replicas=self._world.canonical_replicas
                ),
            )
            self.inflight_chunks_per_lane.setdefault(lane, {})
            if lane not in self._lane_progress:
                self._lane_progress[lane] = LanePtr(0, 0)
            if lane not in self._lane_next_cid:
                self._lane_next_cid[lane] = 0

        inflight: dict[int, dict[int, dict[str, Any]]] = {}
        for lane, by_chunk in self.inflight_chunks_per_lane.items():
            inflight[int(lane)] = {
                int(cid): ch.state_dict() for cid, ch in by_chunk.items()
            }

        progress = {
            int(l): {"chunk_id": int(p.chunk_id), "offset": int(p.offset)}
            for l, p in self._lane_progress.items()
        }

        world = {
            "canonical_replicas": int(self._world.canonical_replicas),
            "num_ranks": int(self._world.num_ranks),
            "physical_rank": int(self._world.physical_rank),
            "mapping": {
                int(r): [int(x) for x in lanes]
                for r, lanes in self._world.lanes_for_rank.items()
            },
        }
        for lane, by_chunk in self.inflight_chunks_per_lane.items():
            if by_chunk:
                mx = max(by_chunk)
                assert self._lane_next_cid[lane] == mx + 1
                if self._lane_next_cid[lane] < mx + 1:
                    # Either bump silently or assert in debug
                    self._lane_next_cid[lane] = mx + 1  # or: assert False

        lane_next = {int(l): int(n) for l, n in self._lane_next_cid.items()}
        lane_ws_state = {int(l): self._lane_ws[l].state_dict() for l in self._lane_ws}

        return {
            "version": 1,
            "world": world,
            "inflight": inflight,
            "progress": progress,
            "lane_next_cid": lane_next,
            "work_source": self._work.state_dict(),
            "lane_ws_state": lane_ws_state,
            "last_round_id": self._last_round_id,
            "checkpoint_reload_count": self._checkpoint_reload_count,
        }

    # ---------- FS utilities ----------
    def _atomic_write_text(self, path: Path, text: str) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "w") as f:
            f.write(text)
        os.replace(tmp, path)

    def _atomic_write_json(self, path: Path, payload: dict) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, path)

    def _read_text(self, path: Path) -> str | None:
        try:
            with open(path, "r") as f:
                return f.read()
        except Exception:
            return None

    def _read_json(self, path: Path) -> dict | None:
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return None

    def _wait_until(
        self, pred: Callable[[], bool], timeout: float, poll: float = 0.05
    ) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if pred():
                return True
            time.sleep(poll)
        return False

    # ---------- Round / paths ----------

    def _publish_new_round_id(self) -> str:
        rid = str(int(time.time() * 1e9))  # monotonic-ish
        self._atomic_write_text(self._round_file, rid)
        self._last_round_id = rid
        return rid

    def _read_open_round_id(self) -> str | None:
        rid = (self._read_text(self._round_file) or "").strip()
        if not rid:
            return None

        # This is a practically relevant guard in case users do neither supply a run id nor a fresh aggregate directory. With this check we at least avoid stale state if it is obviously old.
        try:
            age_s = time.time() - self._round_file.stat().st_mtime
        except OSError:
            return None  # If we can't stat it, treat as missing/invalid
        if age_s > 120:
            return None

        # Guard: if we already consumed this rid, skip if our own shard for
        # this rid exists. Under fork, a child can inherit _last_round_id==rid
        # from the parent before it has published; in that case we must proceed.
        if self._last_round_id is not None and rid == self._last_round_id:
            try:
                my_fp = self._state_file_path(rid)
            except Exception:
                my_fp = None
            if not my_fp or my_fp.exists():
                return None

        # only “open” if merged for this rid doesn’t exist yet
        if not self._merged_file_path(rid).exists():
            return rid
        return None

    def _wait_value(
        self, supplier: Callable[[], T | None], timeout: float, poll: float = 0.05
    ) -> T | None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            val = supplier()
            if val is not None:
                return val
            time.sleep(poll)
        return None

    def _open_round_id(self, is_leader: bool) -> str:
        if is_leader:
            return self._publish_new_round_id()

        rid = self._wait_value(self._read_open_round_id, self._agg_timeout_s)
        if rid is None:
            raise RuntimeError(f"Timeout waiting for round id at {self._round_file}")
        return rid

    def _state_file_path(self, round_id: str) -> Path:
        wid, _ = get_torch_worker_info()
        pid = os.getpid()
        return (
            self._agg_dir
            / f"state_r{self._world.physical_rank}_w{wid}_p{pid}_{round_id}.json"
        )

    def _merged_file_path(self, round_id: str) -> Path:
        return self._agg_dir / f"merged_{round_id}.json"

    def _list_state_files(self, round_id: str) -> list[Path]:
        return list(self._agg_dir.glob(f"state_r*_w*_{round_id}.json"))

    def _read_states_for_round(
        self, round_id: str, printt: bool = False
    ) -> tuple[list[dict], set[int]]:
        states: list[dict] = []
        covered: set[int] = set()
        for fp in self._list_state_files(round_id):
            st = self._read_json(fp)
            if st is None:
                if printt:
                    print(f"{fp.name} covers nothing!")
                continue
            states.append(st)
            for k in st.get("progress", {}).keys():
                if printt:
                    print(f"{fp.name} covers lane {k}!")
                covered.add(int(k))
        return states, covered

    def state_dict(self) -> dict[str, Any]:
        # Fast path: single node & single active worker → just return local
        local = self._state_dict_local()
        lanes_all = self._world.lanes_for_rank[self._world.physical_rank]
        worker_id, workers_per_rank = get_torch_worker_info()
        active_here = self._active_workers(workers_per_rank, lanes_all)
        if self._world.num_ranks == 1 and active_here == 1:
            return local

        # Round setup
        is_leader = self._world.physical_rank == 0 and worker_id == 0
        round_id = self._open_round_id(is_leader)
        self._last_round_id = round_id
        my_path = self._state_file_path(round_id)
        self._atomic_write_json(my_path, local)  # atomic publish

        merged_path = self._merged_file_path(round_id)
        if is_leader:
            # Wait for complete lane coverage
            expected_lanes = set(range(int(self._world.canonical_replicas)))

            def _have_full_coverage() -> bool:
                states, covered = self._read_states_for_round(round_id)
                return bool(states) and expected_lanes.issubset(covered)

            if not self._wait_until(_have_full_coverage, self._agg_timeout_s):
                states, covered = self._read_states_for_round(round_id)
                missing = sorted(expected_lanes - covered)
                raise RuntimeError(
                    f"Aggregation timeout after {self._agg_timeout_s}s. "
                    + f"Missing lanes={missing}; files={len(self._list_state_files(round_id))}"
                )

            # Merge and publish
            states, _ = self._read_states_for_round(round_id, printt=False)
            merged = self._merge_state_dicts(states)
            assert not merged_path.exists()
            self._atomic_write_json(merged_path, merged)
            my_path.unlink(missing_ok=True)
            if self._previous_merged_file is not None:
                self._previous_merged_file.unlink(missing_ok=True)
            self._previous_merged_file = merged_path
            self._round_file.unlink(
                missing_ok=True
            )  # can also clean this up since we know everybody consumed it.
            return merged

        # Followers: wait for merged file
        ok = self._wait_until(lambda: merged_path.exists(), self._agg_timeout_s)
        if not ok:
            raise RuntimeError(
                f"Timed out after {self._agg_timeout_s}s waiting for merged checkpoint at {merged_path}"
            )
        merged = self._read_json(merged_path)
        my_path.unlink(missing_ok=True)
        if merged is None:
            raise RuntimeError(f"Failed to read merged checkpoint {merged_path}")
        return merged

    def _merge_state_dicts(self, states: list[dict[str, Any]]) -> dict[str, Any]:
        assert states
        C = int(states[0]["world"]["canonical_replicas"])
        for s in states[1:]:
            if int(s["world"]["canonical_replicas"]) != C:
                raise RuntimeError("canonical_replicas mismatch")

        merged_num_ranks: int | None = None
        for s in states:
            w = s.get("world", {})
            if "num_ranks" in w and w["num_ranks"] is not None:
                n = int(w["num_ranks"])
                if merged_num_ranks is None:
                    merged_num_ranks = n
                elif merged_num_ranks != n:
                    raise RuntimeError("num_ranks mismatch across state shards")

        inflight, progress, lane_next, lane_ws_state = {}, {}, {}, {}
        for st in states:
            for lane_s, by_chunk in st.get("inflight", {}).items():
                lane = int(lane_s)
                if lane in inflight:
                    raise RuntimeError(f"duplicate inflight for lane {lane}")
                inflight[lane] = {
                    int(cid): payload for cid, payload in by_chunk.items()
                }

            for lane_s, p in st.get("progress", {}).items():
                lane = int(lane_s)
                if lane in progress:
                    raise RuntimeError(f"duplicate progress for lane {lane}")
                progress[lane] = {
                    "chunk_id": int(p["chunk_id"]),
                    "offset": int(p["offset"]),
                }

            for lane_s, nxt in st.get("lane_next_cid", {}).items():
                lane = int(lane_s)
                if lane in lane_next:
                    raise RuntimeError(f"duplicate lane_next_cid for lane {lane}")
                lane_next[lane] = int(nxt)

            for lane_s, ws in st.get("lane_ws_state", {}).items():
                lane = int(lane_s)
                if lane in lane_ws_state:
                    raise RuntimeError(f"duplicate lane_ws_state for lane {lane}")
                lane_ws_state[lane] = ws

        work_config = next(
            (s.get("work_config") for s in states if s.get("work_config")), None
        )
        last_round_ids = {s.get("last_round_id") for s in states}
        assert len(last_round_ids) == 1

        checkpoint_reload_counts = {s.get("checkpoint_reload_count") for s in states}
        assert len(checkpoint_reload_counts) == 1

        return {
            "version": 1,
            "world": {"canonical_replicas": C, "num_ranks": merged_num_ranks},
            "inflight": inflight,
            "progress": progress,
            "lane_next_cid": lane_next,
            "work_config": work_config,
            "lane_ws_state": lane_ws_state,
            "last_round_id": list(last_round_ids)[0],
            "checkpoint_reload_count": list(checkpoint_reload_counts)[0],
        }

    def load_state_dict(self, state: dict[str, Any], *, replay: bool = True) -> None:
        """Restore engine & WorkSource; enter replay mode if 'replay' is True."""
        if int(state["world"]["canonical_replicas"]) != self._world.canonical_replicas:
            raise RuntimeError("canonical_replicas changed; migration required")

        bs = self._plan.batch_size_hint
        if bs is not None:
            cs = self._work.chunk_size_hint()
            assert cs is not None, (
                "No chunk size hint for current work source, determinism breaks potentially"
            )
            if (bs % cs != 0) and (cs % bs != 0):
                # TODO(MaxiBoether): Currently our notify logic only works with batching if batching does not go across chunk boundaries.
                # Imagine a batch containing samples cross a chunk boundary. We would elastically continue with the highest chunk id within that batch
                # So we would repeat all of the samples from that chunk that we have already seen within the previous batch, because the Batch operator
                # buffers start empty. An example:

                # a batch contains samples from chunks 3 and chunk 4 (so cross chunks).
                # let it look like this c3s7 c3s8 c4s0 c4s1 c4s2 c4s3 for a batch size of 6.
                # now if we see this batch, we will discard chunk 3 from the inflight chunks.
                # during continuation, what now happens is that the source starts to replay c4 from the start.
                # since the batching buffers are empty it would build this batch: c4s0 c4s1 c4s2 c4s3 c4s4 c4s5 and this batch would be accepted by notify because c4s5 increases our pointer.
                # this cannot happen if chunk and batch boundaries are aligned.
                #
                # The long term solution is to ingest a filter op into the graph in case of elastic resumption, and let that op
                # on the sample (pre-batch) level filter out everything before the last seen item. This also requires us to have unique sample IDs even in
                # case an operator has a 1:n sample mapping.
                raise RuntimeError(
                    "Correct resumption requires identical num_ranks or "
                    + "compatible batch/chunk sizes (one must divide the other). "
                    + f"batch_size={bs}, chunk_size={cs}"
                )

        # This is FOR ALL WORKERS on that node. So we restore a bit more than we have to because we cannot be certain whether load_state_dict is called before workers are instantiated or not.

        owned = set(self._world.lanes_for_rank[self._world.physical_rank])
        base_ws = self._work
        # TODO(MaxiBoether): in the future support sometihng like this
        # if state.get("work_config") and hasattr(type(self._work), "from_config"):
        #    try:
        # ensure identity parity if user passed a mismatched base WorkSource
        #        base_ws = type(self._work).from_config(state["work_config"])
        #    except Exception:
        #        pass

        # (Re)build lane WorkSources for currently owned lanes
        self._lane_ws.clear()
        for lane in owned:
            self._lane_ws[lane] = base_ws.clone_for_lane(
                lane, canonical_replicas=self._world.canonical_replicas
            )

        # Restore inflight & pointers
        self.inflight_chunks_per_lane.clear()
        inflight_all = state.get("inflight", {})
        for lane_s, by_chunk in inflight_all.items():
            lane = int(lane_s)
            if lane not in owned:
                continue
            self.inflight_chunks_per_lane[lane] = {}
            for cid_s, payload in by_chunk.items():
                cid = int(cid_s)
                self.inflight_chunks_per_lane[lane][cid] = WorkChunk.from_state(payload)

        self._lane_progress.clear()
        progress_all = state.get("progress", {})
        for lane in owned:
            p = progress_all.get(str(lane)) or progress_all.get(int(lane))
            if p is not None:
                self._lane_progress[lane] = LanePtr(
                    int(p["chunk_id"]), int(p["offset"])
                )
            else:
                raise RuntimeError(
                    f"Lane {lane} does not have progress state in the checkpoint!"
                )

        self._lane_next_cid.clear()
        lane_next_all = state.get("lane_next_cid", {})
        for lane in owned:
            saved_next = lane_next_all.get(str(lane)) or lane_next_all.get(int(lane))
            if saved_next is None:
                raise RuntimeError(
                    f"Lane {lane} does not have next cid in the checkpoint"
                )
            self._lane_next_cid[lane] = int(saved_next)

        # Restore lane WS state for owned lanes
        lane_ws_state_all = state.get("lane_ws_state", {})
        for lane, ws in self._lane_ws.items():
            st = lane_ws_state_all.get(str(lane)) or lane_ws_state_all.get(int(lane))
            if st is not None:
                ws.load_state_dict(dict(st))
            else:
                raise RuntimeError(
                    f"Lane {lane} does not have WorkSource state in the checkpoint"
                )

        self._last_round_id = state["last_round_id"]
        self._checkpoint_reload_count = state["checkpoint_reload_count"] + 1
        self._agg_dir.mkdir(parents=True, exist_ok=True)

        # Enter replay so we drop until we pass the saved pointers (if requested)
        if replay:
            # Keep the checkpoint pointer separate:
            self._replay_until = {
                l: LanePtr(v.chunk_id, v.offset) for l, v in self._lane_progress.items()
            }
            # Start a fresh replay cursor at offset 0 for each lane's target chunk
            self._replay_seen = {
                l: LanePtr(v.chunk_id, 0) for l, v in self._lane_progress.items()
            }
        else:
            self._replay_until = None
            self._replay_seen = None
