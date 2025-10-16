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

import warnings
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Literal, Optional

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


@dataclass
class RuntimeOptions:
    """User-tunable knobs that influence how the engine constructs runners."""

    runner: Optional[str] = None
    allow_subprocess_in_worker: bool = False  # TODO(MaxiBoether): Implement this.
    mp_context: Any = None
    max_workers: int = 8  # How does this get configured? The fundamental problem is that stages are implicitly created. So hwo should a user define how many workers to give each stage beforehand?
    per_stage_runner: dict[int, str] = field(default_factory=dict)
    deterministic: bool = False
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

        self._work = work
        # Stage runners are stored heterogeneously.
        self._runners: list[ThreadStageRunner] = []
        self._build_runners()

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
        for idx, (stage, runner) in enumerate(zip(self._plan.stages, self._runners)):
            runner_kind = "unknown"
            op_in_q: int | None = None
            stage_prefetch: int | None = None

            if isinstance(runner, ThreadStageRunner):  # pyright: ignore[reportUnnecessaryIsInstance]
                runner_kind = "threads"
                op_in_q = getattr(runner, "_queue_capacity", None)
                stage_prefetch = getattr(runner, "_prefetch_capacity", None)

            # Header with placement and runner only (buffers are shown inline)
            lines.append(f"Stage[{idx}] place={stage.placement} runner={runner_kind}")

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

    def _build_runners(self) -> None:
        """Instantiate per-stage runners according to placement and options."""
        inside_worker = inside_torch_worker()
        default_runner = self._opts.runner or "auto"
        for idx, stage in enumerate(self._plan.stages):
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
                        self._opts.max_workers,
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

    def _lane_stream(
        self, lane_id: LaneId, worker_id: int, workers_per_rank: int
    ) -> Iterator[EngineSample]:
        """Yield EngineSamples for a single lane, fetching chunks lazily.

        Maintains inflight_chunks_per_lane[lane_id][chunk_id] -> chunk_obj.
        """
        inflight_lane = self.inflight_chunks_per_lane.setdefault(lane_id, {})

        # Phase 1: replay restored inflight chunks first (ascending chunk_id)
        for cid in sorted(inflight_lane.keys()):
            chunk = inflight_lane[cid]
            for sample_id in chunk:
                yield (sample_id, lane_id, int(cid))

        # Phase 2: fetch new chunks and assign stable per-lane ids
        while True:
            chunk = self._work.next_chunk_for(
                lane_id,
                worker_id=worker_id,
                workers_per_rank=workers_per_rank,
                canonical_replicas=self._world.canonical_replicas,
            )
            if chunk is None:
                return  # lane exhausted

            cid = int(self._lane_next_cid[lane_id])
            self._lane_next_cid[lane_id] = cid + 1
            inflight_lane[cid] = chunk

            for sample_id in chunk:
                yield (sample_id, lane_id, cid)

    def _source_stream(self) -> Iterator[EngineSample]:
        """Yield sample identifiers from the backing work source."""
        lanes = self._world.lanes_for_rank[self._world.physical_rank]
        worker_id, workers_per_rank = get_torch_worker_info()

        # Build one generator per lane
        gens: dict[LaneId, Iterator[EngineSample]] = {
            lane: self._lane_stream(lane, worker_id, workers_per_rank) for lane in lanes
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

        lanes = self._world.lanes_for_rank[self._world.physical_rank]
        if len(lanes) <= 1:  # Simple case:
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

    def eval_one(self, sample_id: SampleId) -> Any:
        """Synchronously evaluate a single element through every stage runner."""
        value: Any = sample_id
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

    def state_dict(self) -> dict[str, Any]:
        """Serializable snapshot of engine runtime state (no plan/op state)."""
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

        lane_next = {int(l): int(n) for l, n in self._lane_next_cid.items()}

        return {
            "version": 1,
            "world": world,
            "inflight": inflight,
            "progress": progress,
            "lane_next_cid": lane_next,
            "work_source": self._work.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any], *, replay: bool = True) -> None:
        """Restore engine & WorkSource; enter replay mode if 'replay' is True."""
        # Restore WorkSource first (so the chunk stream reproduces deterministically)
        ws_state = state.get("work_source")
        if ws_state is None:
            raise RuntimeError("Missing work_source state in checkpoint")
        self._work.load_state_dict(ws_state)

        # Restore inflight chunks registry
        self.inflight_chunks_per_lane.clear()
        for lane_s, by_chunk in state.get("inflight", {}).items():
            lane = int(lane_s)
            self.inflight_chunks_per_lane[lane] = {}
            for cid_s, payload in by_chunk.items():
                cid = int(cid_s)
                self.inflight_chunks_per_lane[lane][cid] = WorkChunk.from_state(payload)

        # Progress (last yielded pointer)
        self._lane_progress.clear()
        for lane_s, p in state.get("progress", {}).items():
            lane = int(lane_s)
            self._lane_progress[lane] = LanePtr(int(p["chunk_id"]), int(p["offset"]))

        self._lane_next_cid.clear()
        saved_next = state.get("lane_next_cid", {})
        for lane_s, nxt in saved_next.items():
            self._lane_next_cid[int(lane_s)] = int(nxt)

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
