# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Pure compile-time execution descriptor for pipeline runner configuration.

``RuntimeSpec`` captures every scheduling decision the Engine will make
(runner types, worker counts, queue depths) **without** allocating any live
resources.  It is computed from a ``Plan`` and ``RuntimeOptions`` via the
pure function :func:`resolve_runtime_spec`.

This separation enables:
- ``Pipeline.explain()`` without building an Engine.
- Programmatic introspection of the execution plan before iteration starts.
- Passing pre-computed decisions to the Engine, avoiding redundant computation.
"""

from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

from zephon._internal.utils.ipc import DEFAULT_IPC_BUFFER_BYTES, DEFAULT_IPC_TRANSPORT
from zephon._internal.utils.shm_coalesce import (
    DEFAULT_SHM_COALESCE_MAX_SIZE,
    DEFAULT_SHM_MAX_RETAINED_RATIO,
    DEFAULT_SHM_MIN_BUFFER_SIZE,
    DEFAULT_SHM_MIN_RECLAIM_BYTES,
    DEFAULT_SHM_MIN_REUSE_SIZE,
    PayloadMemoryPolicy,
)
from zephon.options import IpcTransport

if TYPE_CHECKING:
    from zephon._internal.graph import Plan, Stage
    from zephon.options import RuntimeOptions


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StageRuntimeSpec:
    """Fully resolved execution configuration for a single pipeline stage."""

    stage_index: int
    runner_type: str  # "threads" | "process" | "inline"
    worker_cap: int
    queue_capacity: int
    prefetch_capacity: int
    output_mode: str  # "microbatches" | "stream_items"
    allow_latency_flush: bool
    coalesce_tensors: bool
    shm_min_size: int
    shm_min_buffer_size: int = DEFAULT_SHM_MIN_BUFFER_SIZE
    shm_min_reuse_size: int = DEFAULT_SHM_MIN_REUSE_SIZE
    shm_max_retained_ratio: float | None = DEFAULT_SHM_MAX_RETAINED_RATIO
    shm_min_reclaim_bytes: int = DEFAULT_SHM_MIN_RECLAIM_BYTES
    shm_coalesce_max_size: int | None = DEFAULT_SHM_COALESCE_MAX_SIZE
    # See RuntimeOptions.max_worker_retries; ignored by non-process runners.
    max_worker_retries: int = 0
    # See RuntimeOptions.ipc_transport / ipc_buffer_bytes; process runners only.
    ipc_transport: IpcTransport = DEFAULT_IPC_TRANSPORT
    ipc_buffer_bytes: int = DEFAULT_IPC_BUFFER_BYTES


@dataclass(frozen=True, slots=True)
class RuntimeSpec:
    """Complete compile-time execution specification.

    Produced by :func:`resolve_runtime_spec`.  Frozen and picklable.
    """

    stages: tuple[StageRuntimeSpec, ...]
    # Allocation description
    worker_allocation: str
    max_workers: int
    # Pipeline-level flags
    preserves_cursor_order: bool
    final_prefetch: int
    deterministic: bool

    def explain(self, plan: Plan) -> str:
        """ASCII execution graph matching ``Engine.explain()`` format.

        Reads from :class:`StageRuntimeSpec` fields instead of live runner
        attributes, so no Engine is required.
        """
        lines: list[str] = []

        # Allocation header
        mode = self.worker_allocation
        if mode == "global":
            lines.append(f"Allocation=global total={self.max_workers}")
        elif mode == "per_stage_fixed":
            lines.append(f"Allocation=per_stage_fixed per_stage={self.max_workers}")
        else:
            lines.append("Allocation=fit_to_ops (cap=sum(node.parallelism), min 1/op)")

        # Bookkeeping header
        bookkeeping = (
            "simple chunk-watermark (preserves_cursor_order=True)"
            if self.preserves_cursor_order
            else "contributor-aware (packing/shuffle-safe)"
        )
        lines.append(f"Bookkeeping={bookkeeping}")

        # Per-stage detail
        for spec in self.stages:
            stage = plan.stages[spec.stage_index]

            header = (
                f"Stage[{spec.stage_index}] place={stage.placement} "
                f"runner={spec.runner_type} cap={spec.worker_cap} "
                f"mode={spec.output_mode}"
            )
            lines.append(header)

            # Ops with in-queue connectors
            nodes = [f"{nd.name}@p{nd.parallelism}" for nd in stage.nodes]
            if not nodes:
                lines.append("  [empty stage]")
            elif len(nodes) == 1:
                lines.append(f"  {nodes[0]}")
            else:
                if spec.runner_type != "inline":
                    connector = f" -[in_q={spec.queue_capacity}]-> "
                else:
                    connector = " -> "
                lines.append("  " + connector.join(nodes))

            # Stage output queue
            if spec.runner_type != "inline":
                sp = spec.prefetch_capacity or 0
                out_q = max(1, sp or spec.queue_capacity)
                lines.append(f"  --[stage_out={out_q}]-->")
            elif spec.prefetch_capacity and spec.prefetch_capacity > 0:
                lines.append(f"  --[stage_out={max(1, spec.prefetch_capacity)}]-->")
            else:
                lines.append("  --[stage_out=?]-->")

            # Boundary arrow
            is_last = spec.stage_index == len(self.stages) - 1
            if not is_last:
                if spec.prefetch_capacity and spec.prefetch_capacity > 0:
                    lines.append(
                        f"  ==[prefetch={spec.prefetch_capacity}]==> "
                        f"Stage[{spec.stage_index + 1}]"
                    )
                else:
                    lines.append(f"  ==> Stage[{spec.stage_index + 1}]")
            else:
                if self.final_prefetch > 0:
                    lines.append(
                        f"  ==[final_prefetch={self.final_prefetch}]==> pipeline_end"
                    )
                else:
                    lines.append("  ==> pipeline_end")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Auto-derivation of numeric tuning knobs
# ---------------------------------------------------------------------------
#
# Each `resolve_*` helper materializes a `RuntimeOptions` field that may be
# left as ``None`` (signalling "pick a sensible default"). Keeping the
# derivation logic in one place lets the rest of the engine and the consumer
# Pipeline API read uniform integer values without re-implementing fallbacks.


_MAX_WORKERS_FLOOR = 4
_MAX_WORKERS_CEIL = 16


def _resolve_max_workers(opts: RuntimeOptions) -> int:
    """Materialize ``opts.max_workers`` to a concrete int.

    When unset, derives from ``os.cpu_count()`` clamped to ``[4, 16]``.

    The resolved value is used in two distinct ways depending on
    ``worker_allocation``:

    * ``per_stage_fixed`` / ``global``: directly caps per-stage / total worker
      counts.
    * ``fit_to_ops`` (default): NOT used for worker counts — those come from
      each op's declared parallelism. The resolved number only feeds the
      buffer-sizing cascade (``prefetch_batches``, ``op_queue_capacity``,
      ``mtp_buffer``, ``default_stage_prefetch``), so the clamp keeps those
      buffers reasonable.

    """
    if opts.max_workers is not None:
        return opts.max_workers
    cpus = os.cpu_count() or _MAX_WORKERS_FLOOR
    return min(_MAX_WORKERS_CEIL, max(_MAX_WORKERS_FLOOR, cpus))


def resolve_prefetch_batches(opts: RuntimeOptions) -> int:
    """Materialize ``opts.prefetch_batches`` to a concrete int.

    When unset, derives from the resolved ``max_workers`` so the consumer-side
    tail buffer scales with the number of producers in flight.
    """
    if opts.prefetch_batches is not None:
        return opts.prefetch_batches
    return max(8, 2 * _resolve_max_workers(opts))


def _resolve_default_stage_prefetch(opts: RuntimeOptions, *, runner_type: str) -> int:
    """Materialize ``opts.default_stage_prefetch`` to a concrete int.

    When unset, scales with the resolved ``prefetch_batches``.
    Floors of 4 (process) and 8 (threads/inline) guarantee meaningful
    pipelining even on small machines.
    """
    if opts.default_stage_prefetch is not None:
        return opts.default_stage_prefetch
    prefetch = resolve_prefetch_batches(opts)
    if runner_type == "process":
        return max(4, prefetch // 4)
    return max(8, prefetch // 2)


def _resolve_op_queue_capacity(opts: RuntimeOptions) -> int:
    """Materialize ``opts.op_queue_capacity`` to a concrete int.

    Scales with the resolved prefetch depth so backpressure stays consistent.
    """
    if opts.op_queue_capacity is not None:
        return opts.op_queue_capacity
    return max(8, resolve_prefetch_batches(opts))


def resolve_mtp_buffer(opts: RuntimeOptions) -> int:
    """Materialize ``opts.mtp_buffer`` to a concrete int.

    MTP only matters when ``opts.mtp_mode`` is True. Default keeps the IPC
    queue at roughly half the tail prefetch depth.
    """
    if opts.mtp_buffer is not None:
        return opts.mtp_buffer
    return max(4, resolve_prefetch_batches(opts) // 2)


# ---------------------------------------------------------------------------
# Pure helper functions (extracted from Engine)
# ---------------------------------------------------------------------------


def stage_parallelism(stage: Stage) -> int:
    """Compute effective parallelism for a stage.

    Sums declared parallelism of all non-ReplayFilter nodes.  ReplayFilter
    parallelism is tracked separately and used as a floor.
    """
    from zephon._internal.ops.replay_filter import ReplayFilter

    total = 0
    filter_parallelism = 0
    for nd in stage.nodes:
        dop = max(1, (nd.parallelism or 1))
        if isinstance(nd.op, ReplayFilter):
            filter_parallelism = max(filter_parallelism, dop)
            continue
        total += dop
    if total == 0:
        total = filter_parallelism
    else:
        total = max(total, filter_parallelism)
    return max(1, total)


def apportion(total: int, weights: list[int]) -> list[int]:
    """Split *total* into integer parts proportional to *weights*.

    Uses largest-remainder (Hamilton) method; guarantees ``sum(parts) == total``.
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
    order = sorted(range(n), key=lambda i: quotas[i] - floors[i], reverse=True)
    for i in range(remaining):
        floors[order[i]] += 1
    return floors


# ---------------------------------------------------------------------------
# Main resolver
# ---------------------------------------------------------------------------


def resolve_runtime_spec(
    plan: Plan,
    opts: RuntimeOptions,
    *,
    inside_worker: bool = False,
) -> RuntimeSpec:
    """Compute full execution specification from Plan + RuntimeOptions.

    This is a **pure function** — no side effects, no resource allocation.
    The returned :class:`RuntimeSpec` describes everything the Engine needs
    to instantiate runners.

    Parameters
    ----------
    plan:
        The execution plan (stages, nodes, metadata).
    opts:
        ``RuntimeOptions`` instance.
    inside_worker:
        Whether the Engine will run inside a PyTorch DataLoader worker.
        Affects runner type selection (``process`` → ``threads`` demotion).
    """
    PayloadMemoryPolicy(
        shm_min_size=opts.shm_min_size,
        min_buffer_size=opts.shm_min_buffer_size,
        min_reuse_size=opts.shm_min_reuse_size,
        coalesce=opts.coalesce_tensors,
        max_retained_ratio=opts.shm_max_retained_ratio,
        min_reclaim_bytes=opts.shm_min_reclaim_bytes,
        coalesce_max_size=opts.shm_coalesce_max_size,
    )
    num_stages = len(plan.stages)
    mode = opts.worker_allocation

    if mode == "autotune":
        raise NotImplementedError(
            "worker_allocation='autotune' is reserved for future auto-tuning. "
            "Use 'per_stage_fixed' or 'global' for now."
        )

    # Materialize all numeric tuning knobs up front so the rest of the
    # resolver and downstream RuntimeSpec carry concrete ints, never None.
    max_workers = _resolve_max_workers(opts)
    op_queue_capacity = _resolve_op_queue_capacity(opts)
    final_prefetch = resolve_prefetch_batches(opts)

    # --- Worker count computation ---
    if mode == "global":
        total = max_workers
        if total < num_stages:
            warnings.warn(
                f"[zephon] max_workers_total={total} < number of stages={num_stages}; "
                f"bumping to {num_stages} (1 thread per stage).",
                RuntimeWarning,
                stacklevel=2,
            )
            # Bumped path: give every stage exactly 1 worker. Weight-proportional
            # apportion would still produce zero-worker stages for small-parallelism
            # ops (e.g. [2,5,1] under total=3 ⇒ [1,2,0]), which deadlocks the
            # pipeline. The warning above promises "1 thread per stage", so honor it.
            per_stage_caps = [1] * num_stages
        else:
            weights = [stage_parallelism(s) for s in plan.stages]
            per_stage_caps = apportion(total, weights)
    elif mode == "per_stage_fixed":
        per_stage_caps = [max_workers for _ in range(num_stages)]
    else:  # fit_to_ops (default)
        per_stage_caps = [stage_parallelism(s) for s in plan.stages]

    # --- Per-stage runner selection ---
    default_runner = opts.runner or "auto"
    stage_specs: list[StageRuntimeSpec] = []

    for idx, stg in enumerate(plan.stages):
        is_last = idx == num_stages - 1
        cap = per_stage_caps[idx]

        # Runner type selection. Inside a PyTorch DataLoader worker, process
        # runners are demoted to threads (subprocesses cannot fork further).
        chosen = opts.per_stage_runner.get(idx, default_runner)
        if chosen == "auto":
            if inside_worker:
                chosen = "threads"
            elif stg.placement == "remote":
                chosen = "remote"
            else:
                chosen = "threads"
        if inside_worker and chosen == "process":
            chosen = "threads"

        # Runner hint overrides from Planner
        hint = getattr(stg, "runner_hint", None)
        if hint == "inline":
            chosen = "inline"
        elif hint == "threads":
            chosen = "threads"

        # Default stage prefetch depends on the resolved runner type, so
        # compute it after the runner is selected.
        prefetch = opts.per_stage_prefetch.get(
            idx,
            _resolve_default_stage_prefetch(opts, runner_type=chosen),
        )

        # Latency flush in deterministic mode
        allow_latency = opts.allow_latency_flush_in_deterministic
        if opts.deterministic:
            has_sensitive = any(
                getattr(nd.op.traits(), "batch_shape_sensitive", False)
                for nd in stg.nodes
            )
            if has_sensitive and allow_latency:
                allow_latency = False

        output_mode = "stream_items" if is_last else "microbatches"

        stage_specs.append(
            StageRuntimeSpec(
                stage_index=idx,
                runner_type=chosen,
                worker_cap=cap,
                queue_capacity=op_queue_capacity,
                prefetch_capacity=prefetch,
                output_mode=output_mode,
                allow_latency_flush=allow_latency,
                coalesce_tensors=opts.coalesce_tensors,
                shm_min_size=opts.shm_min_size,
                shm_min_buffer_size=opts.shm_min_buffer_size,
                shm_min_reuse_size=opts.shm_min_reuse_size,
                shm_max_retained_ratio=opts.shm_max_retained_ratio,
                shm_min_reclaim_bytes=opts.shm_min_reclaim_bytes,
                shm_coalesce_max_size=opts.shm_coalesce_max_size,
                max_worker_retries=opts.max_worker_retries,
                ipc_transport=opts.ipc_transport,
                ipc_buffer_bytes=opts.ipc_buffer_bytes,
            )
        )

    return RuntimeSpec(
        stages=tuple(stage_specs),
        worker_allocation=mode,
        max_workers=max_workers,
        preserves_cursor_order=plan.preserves_cursor_order,
        final_prefetch=final_prefetch,
        deterministic=opts.deterministic,
    )
