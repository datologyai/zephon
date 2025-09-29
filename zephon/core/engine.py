# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Runtime execution engine that wires plans to concrete stage runners."""

from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Optional

from zephon.core.constants import Element, SampleId
from zephon.core.graph import Plan
from zephon.runners.threads import ThreadStageRunner
from zephon.work import WorkSource


def inside_torch_worker() -> bool:
    """Return True if the current process is a PyTorch DataLoader worker."""
    try:
        from torch.utils.data import get_worker_info

        return get_worker_info() is not None
    except Exception:
        return False


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


class Engine:
    """Bind a `Plan` to concrete runners and orchestrate streaming execution."""

    def __init__(
        self, plan: Plan, ctx: dict[str, Any], opts: RuntimeOptions, work: WorkSource
    ) -> None:
        """Initialize stage runners and prepare to stream work items."""
        self._plan = plan
        self._ctx = ctx
        self._opts = opts
        if self._opts.deterministic:
            # TODO(MaxiBoether): there most likely is a difference between determinism (same setting) and elastic scalability.
            # Determinism == same ordering given I run my setup exactly the same way (reliable multithreading, no elastic scaling)
            # Elastic scalability: determinism plus concept of canonical nodes to ensure determinism across configs!
            raise NotImplementedError("Deterministic mode not implemented yet.")

        self._work = work
        self._runners: list[Any] = []
        self._build_runners()

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

            if isinstance(runner, ThreadStageRunner):
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
                self._runners.append(
                    ThreadStageRunner(
                        stage,
                        self._ctx,
                        self._opts.max_workers,
                        prefetch_capacity=prefetch,
                    )
                )
            elif chosen == "remote":
                raise NotImplementedError("Remote workers are not yet implemented.")
            elif chosen == "process":
                raise NotImplementedError("ProcessStageRunner not yet implemented.")
            else:
                raise ValueError(f"Unknown runner '{chosen}'")

    def _source_stream(self) -> Iterator[Element]:
        """Yield sample identifiers from the backing work source."""
        while True:
            chunk = self._work.next_chunk()
            if chunk is None:
                break
            for sample_id in chunk.sample_ids:
                yield sample_id

    def build_iter(self) -> Iterator[Element]:
        """Return an iterator that threads the work stream through all stages."""
        # TODO(MaxiBoether): Change the way we iterate over the chunks to respect (implicit) mixtures and maybe implement sub-chunk concept for parallel data fetching.
        stream_iter: Iterable[Element] = self._source_stream()
        for runner in self._runners:  # Construct overall pipeline by chaining runners
            stream_iter = runner.run(stream_iter)
        for element in stream_iter:
            yield element

    def eval_one(self, sample_id: SampleId) -> Element:
        """Synchronously evaluate a single element through every stage runner."""
        value: Element = sample_id
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
