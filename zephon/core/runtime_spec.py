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
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zephon.core.engine import RuntimeOptions
    from zephon.core.graph import Plan, Stage


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


@dataclass(frozen=True, slots=True)
class RuntimeSpec:
    """Complete compile-time execution specification.

    Produced by :func:`resolve_runtime_spec`.  Frozen and picklable.
    """

    stages: tuple[StageRuntimeSpec, ...]
    # Allocation description
    worker_allocation: str
    max_workers: int
    stage_weighting: str
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
            lines.append(
                f"Allocation=global total={self.max_workers} weighting={self.stage_weighting}"
            )
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
# Pure helper functions (extracted from Engine)
# ---------------------------------------------------------------------------


def stage_parallelism(stage: Stage) -> int:
    """Compute effective parallelism for a stage.

    Sums declared parallelism of all non-ReplayFilter nodes.  ReplayFilter
    parallelism is tracked separately and used as a floor.
    """
    from zephon.ops.replay_filter import ReplayFilter

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
    num_stages = len(plan.stages)
    mode = opts.worker_allocation

    if mode == "autotune":
        raise NotImplementedError(
            "worker_allocation='autotune' is reserved for future auto-tuning. "
            "Use 'per_stage_fixed' or 'global' for now."
        )

    # --- Worker count computation ---
    if mode == "global":
        total = opts.max_workers
        if total < num_stages:
            warnings.warn(
                f"[zephon] max_workers_total={total} < number of stages={num_stages}; "
                f"bumping to {num_stages} (1 thread per stage).",
                RuntimeWarning,
                stacklevel=2,
            )
            total = num_stages
        if opts.stage_weighting == "equal":
            weights = [1 for _ in range(num_stages)]
        else:
            weights = [stage_parallelism(s) for s in plan.stages]
        per_stage_caps = apportion(total, weights)
    elif mode == "per_stage_fixed":
        per_stage_caps = [opts.max_workers for _ in range(num_stages)]
    else:  # fit_to_ops (default)
        per_stage_caps = [stage_parallelism(s) for s in plan.stages]

    # --- Per-stage runner selection ---
    default_runner = opts.runner or "auto"
    stage_specs: list[StageRuntimeSpec] = []

    for idx, stg in enumerate(plan.stages):
        is_last = idx == num_stages - 1
        cap = per_stage_caps[idx]
        prefetch = opts.per_stage_prefetch.get(idx, opts.default_stage_prefetch)

        # Runner type selection
        chosen = opts.per_stage_runner.get(idx, default_runner)
        if chosen == "auto":
            if inside_worker and not opts.allow_mtp_in_worker:
                chosen = "threads"
            elif stg.placement == "remote":
                chosen = "remote"
            else:
                chosen = "threads"
        if inside_worker and chosen == "process" and not opts.allow_mtp_in_worker:
            chosen = "threads"

        # Runner hint overrides from Planner
        hint = getattr(stg, "runner_hint", None)
        if hint == "inline":
            chosen = "inline"
        elif hint == "threads":
            chosen = "threads"

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
                queue_capacity=opts.op_queue_capacity,
                prefetch_capacity=prefetch,
                output_mode=output_mode,
                allow_latency_flush=allow_latency,
            )
        )

    return RuntimeSpec(
        stages=tuple(stage_specs),
        worker_allocation=mode,
        max_workers=opts.max_workers,
        stage_weighting=opts.stage_weighting,
        preserves_cursor_order=plan.preserves_cursor_order,
        final_prefetch=opts.prefetch_batches or 0,
        deterministic=opts.deterministic,
    )
