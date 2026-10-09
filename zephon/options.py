# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Runtime options for ``Pipeline.options()`` (``RuntimeOptions``, ``IpcTransport``)."""

import multiprocessing as mp
import warnings
from dataclasses import dataclass, field
from typing import Any, Literal

from zephon._internal.utils.ipc import (
    DEFAULT_IPC_BUFFER_BYTES,
    DEFAULT_IPC_TRANSPORT,
    DEFAULT_MTP_BUFFER_BYTES,
)
from zephon._internal.utils.shm_coalesce import (
    DEFAULT_SHM_COMPACT_ABOVE_RATIO,
    DEFAULT_SHM_COMPACT_MIN_SAVINGS_BYTES,
    DEFAULT_SHM_MAX_COALESCED_BYTES,
    DEFAULT_SHM_MIN_FORWARD_BYTES,
    DEFAULT_SHM_MIN_ITEM_BYTES,
    DEFAULT_SHM_MIN_NEW_ALLOCATION_BYTES,
)
from zephon.io.options import StoreOptions
from zephon.observability import ExecutionTrackingMode, MetricsSinkConfig
from zephon.work.base import MixtureReadConfig

IpcTransport = Literal["socketpair", "pipe"]


DEFAULT_RUN_ID = "default_run_id"


_SHM_OPTION_ALIASES = {
    "shm_min_size": "shm_min_item_bytes",
    "coalesce_tensors": "shm_coalesce",
}


def _normalize_shm_options(hints: dict[str, Any]) -> dict[str, Any]:
    """Resolve legacy keyword names consistently at both configuration entry points."""
    result = hints.copy()
    for old, new in _SHM_OPTION_ALIASES.items():
        if old not in result:
            continue
        value = result.pop(old)
        if value is None:
            continue
        if result.get(new) is not None and result[new] != value:
            raise ValueError(f"Conflicting pipeline options: {old} and {new}")
        warnings.warn(
            f"{old} is deprecated; use {new} instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        result[new] = value
    return result


@dataclass
class RuntimeOptions:
    """User-tunable knobs that influence how the engine constructs runners.

    Numeric tuning knobs default to ``None``, resulting in sensible
    defaults via  ``resolve_*`` helpers that form a dependency cascade.
    Each resolver calls the ones above it when its own field is unset::

        max_workers              ← os.cpu_count(), clamped [4, 16]
            └── prefetch_batches             = max(8, 2 × max_workers)
                    ├── op_queue_capacity        = max(8, prefetch_batches)
                    ├── mtp_buffer               = max(4, prefetch_batches // 2)
                    │       └── mtp_prefetch         = max(4, mtp_buffer // 10)
                    └── default_stage_prefetch   = max(4, prefetch_batches // 4)  [process]
                                                   max(8, prefetch_batches // 2)  [threads/inline]

    For example, ``max_workers=24`` naturally bumps the
    prefetch / queue / MTP buffers without the user having to set them too,
    while ``prefetch_batches=64`` overrides the tail buffer alone and lets
    the rest auto-derive from it.

    By default, ``max_workers`` is derived from the host's CPU count,
    but is clamped to a range of ``[4, 16]``. For any large deployment
    this will hit the upper limit, but the clamp allows Zephon to run
    reasonably on small machines as well.
    """

    runner: str | None = None  # Default runner. Typically auto-inferred.
    run_id: str = DEFAULT_RUN_ID
    per_stage_runner: dict[int, str] = field(
        default_factory=dict
    )  # Manual override for runner per-stage. Mostly useful for debugging and advanced usage.
    mp_context: Any = mp.get_context("spawn")
    # "autotune" is reserved for a future probe-based tuner; passing it today
    # raises NotImplementedError in resolve_runtime_spec.
    worker_allocation: Literal[
        "fit_to_ops", "per_stage_fixed", "global", "autotune"
    ] = "fit_to_ops"
    # max_workers per stage OR global, depending on worker_allocation.
    # None = auto-derived from os.cpu_count(), clamped to [4, 16].
    max_workers: int | None = None
    deterministic: bool = True
    # Consumer-side prefetch depth at the pipeline tail.
    # None = auto-derived from resolved max_workers (max(8, 2 * max_workers)).
    prefetch_batches: int | None = None
    # Inter-stage prefetch buffer depth.
    # None = auto-derived (4 for thread/inline runners, 2 for process runners).
    default_stage_prefetch: int | None = None
    per_stage_prefetch: dict[int, int] = field(default_factory=dict)
    # Maximum in-flight items in the queue between ops within a stage.
    # None = auto-derived from resolved prefetch_batches (max(8, prefetch_batches)).
    op_queue_capacity: int | None = None
    # Transport under process-runner and MTP IPC queues. "socketpair"
    # (default) is exempt from the shared per-UID pipe budget and honors
    # the *_buffer_bytes requests below; "pipe" is stock mp.Queue.
    ipc_transport: IpcTransport = DEFAULT_IPC_TRANSPORT
    # Kernel buffer request per process-runner IPC queue (socketpair only).
    # Best-effort: Linux clamps to 2 * net.core.wmem_max (416 KiB stock),
    # macOS to kern.ipc.maxsockbuf.
    ipc_buffer_bytes: int = DEFAULT_IPC_BUFFER_BYTES
    mixture_config: MixtureReadConfig | None = None
    io_options: StoreOptions = field(default_factory=StoreOptions)
    # Expert knob:
    # Keep latency-based flush in deterministic mode when True unless a stage contains
    # a batch-shape sensitive operator (in which case we auto-disable it for that stage).
    # When False, latency flush is always disabled in deterministic mode.
    allow_latency_flush_in_deterministic: bool = True

    # === Epoch flush ===
    # Flush sentinel cadence: inject a flush sentinel every K chunks per lane.
    # Forces history-dependent accumulators (preserves_cursor_order=False) to flush,
    # creating clean epoch boundaries for deterministic replay after eviction.
    # None = auto (8 for non-monotonic pipelines, 0 for monotone).
    # Explicitly setting 0 for non-monotonic pipelines is an error.
    flush_every_k_chunks: int | None = None

    # === Shutdown ===
    # "graceful" (default) waits generously for threads/processes to finish.
    # "hard" slashes all join timeouts for fast exit (useful for benchmarks).
    shutdown_mode: Literal["graceful", "hard"] = "graceful"

    # === Auto-validation ===
    # Controls how Pipeline.__iter__ handles the auto-validation harness.
    # "strict" (default) raises ValidationError on any error-severity issue.
    # "warn" runs validation and surfaces the full report via warnings.warn
    # but does not raise — escape hatch for cases where the validator's
    # generic checks produce a false positive against a user op.
    # "off" skips the validator entirely; reserved for last-resort overrides.
    auto_validation: Literal["strict", "warn", "off"] = "strict"

    # === MTP Mode (GIL isolation) ===
    # When True, the Engine runs in a non-daemon subprocess for GIL isolation.
    # The main process only dequeues finished batches via IPC.
    mtp_mode: bool = False
    # Bounded IPC queue depth for MTP mode.
    # None = auto-derived from resolved prefetch_batches (max(4, prefetch_batches // 2)).
    mtp_buffer: int | None = None
    # Same as ipc_buffer_bytes but for the MTP data queue; larger so the
    # subprocess feeder can serialize ahead of consumer demand instead of
    # stalling on the subprocess GIL at every next().  Stock Linux clamps
    # this to 2 * net.core.wmem_max (~416 KiB); raise that sysctl for the
    # full run-ahead.
    mtp_buffer_bytes: int = DEFAULT_MTP_BUFFER_BYTES
    # Size (items) of the main-process buffer that a low-priority thread
    # prefetches from the MTP data queue, hiding IPC recv + unpickle from
    # next().  None = auto (max(4, mtp_buffer // 10)); 0 disables the thread.
    mtp_prefetch: int | None = None
    # Automatically capture a checkpoint from the MTP subprocess after normal
    # iteration completion.  Set to False when multi-rank aggregation is not
    # available (e.g. no aggregate_dir, or ranks run sequentially rather than
    # in parallel).
    mtp_auto_checkpoint: bool = True

    # === Global Coordination ===
    # Total number of ranks (GPUs) in the distributed job.
    world_size: int = 1
    # Unique identifier for this rank (0 to world_size-1).
    global_rank: int = 0

    # === Data Partitioning ===
    # Number of data parallel groups (data partitions).
    # Defaults to world_size (1D parallelism) if not specified.
    dp_degree: int | None = None
    # Which data partition this rank reads (0 to dp_degree-1).
    # Defaults to global_rank (1D parallelism) if not specified.
    dp_group_id: int | None = None

    # === Logical Parallelism ===
    # Number of canonical lanes (for elasticity). Defaults to dp_degree.
    canonical_replicas: int | None = None
    # How to map canonical replicas to dp groups:
    # - 'contiguous': dp groups own contiguous blocks of replicas (locality-friendly)
    # - 'interleaved': replicas are round-robin across dp groups (balanced progress)
    mapping_strategy: Literal["contiguous", "interleaved"] | None = None

    # === IPC serialization ===
    # Disable Zephon SHM preparation in both process directions. Ordinary
    # multiprocessing reducers still apply; Torch may still use shared storage.
    shm_enabled: bool = True
    # Coalesce eligible payloads in a microbatch by dtype into SHM buffers
    # before serialization. Reduces POSIX SHM segments from N to K; coalesced
    # groups are split at shm_max_coalesced_bytes. Only consumed by process runners.
    # Disabling coalescing still applies transport thresholds and view compaction.
    shm_coalesce: bool | None = None  # None resolves to True at construction.
    # Minimum item size eligible for fresh SHM; shared views are grouped first.
    shm_min_item_bytes: int | None = (
        None  # None resolves to the default at construction.
    )
    # Minimum useful bytes per new allocation in a message.
    shm_min_new_allocation_bytes: int = DEFAULT_SHM_MIN_NEW_ALLOCATION_BYTES
    # Minimum useful bytes per existing shared allocation in a message.
    shm_min_forward_bytes: int = DEFAULT_SHM_MIN_FORWARD_BYTES
    # Copy a shared view out when both ratio and absolute savings exceed these
    # limits. None disables copying views out; other references may delay freeing.
    shm_compact_above_ratio: float | None = DEFAULT_SHM_COMPACT_ABOVE_RATIO
    shm_compact_min_savings_bytes: int = DEFAULT_SHM_COMPACT_MIN_SAVINGS_BYTES
    # Bound each coalesced allocation; larger individual values get their own.
    # None permits unlimited coalescing. Compacted views follow the same rule.
    shm_max_coalesced_bytes: int | None = DEFAULT_SHM_MAX_COALESCED_BYTES

    # Deprecated keyword aliases, normalized at construction. Conflicting old
    # and new names are rejected, including when the new value is its default.
    coalesce_tensors: bool | None = None
    shm_min_size: int | None = None

    # ProcessStageRunner only.  Re-dispatches per crashed seq before
    # giving up; 0 disables.  In non-deterministic mode an exhausted
    # seq is dropped instead of failing the pipeline.
    max_worker_retries: int = 3

    # Where all workers/ranks dump their local state. For multi-node, must be a shared filesystem
    # (e.g., NFS) or cloud storage (s3://bucket/path or gs://bucket/path).
    aggregate_dir: str | None = None
    # How long to wait for all contributors and for the merged file.
    aggregate_timeout_s: float = 180.0
    # Serialization format for intermediate aggregation files ("json" or "msgpack").
    aggregate_serializer: str = "msgpack"
    # Compression for intermediate aggregation files ("none" or "zstd").
    aggregate_compressor: str = "zstd"
    # Observability controls.
    execution_tracking: ExecutionTrackingMode = ExecutionTrackingMode.OFF
    metrics_sink_config: MetricsSinkConfig | None = None

    def __post_init__(self) -> None:
        normalized = _normalize_shm_options(
            {
                "shm_coalesce": self.shm_coalesce,
                "shm_min_item_bytes": self.shm_min_item_bytes,
                "coalesce_tensors": self.coalesce_tensors,
                "shm_min_size": self.shm_min_size,
            }
        )
        self.shm_coalesce = (
            True if normalized["shm_coalesce"] is None else normalized["shm_coalesce"]
        )
        self.shm_min_item_bytes = (
            DEFAULT_SHM_MIN_ITEM_BYTES
            if normalized["shm_min_item_bytes"] is None
            else normalized["shm_min_item_bytes"]
        )
        self.coalesce_tensors = None
        self.shm_min_size = None


__all__ = [
    "IpcTransport",
    "RuntimeOptions",
]
