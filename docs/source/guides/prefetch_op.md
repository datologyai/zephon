# Prefetch Operator Guide

The `PrefetchOp` reduces data loading latency by downloading shards to the local cache before they're needed by the fetch operator.
It looks ahead in the sample stream, identifies which shards will be accessed soon, and triggers downloads in parallel to minimize fetch latency.

## Quick Start

Add prefetch to your pipeline before decode operations:

```python
from zephon import Pipeline
from zephon.work import StaticMixtureWorkSource

pipeline = (
    Pipeline(work_source)
    .prefetch(buffer_size=1024)  # Uses default parallelism=4
    .decode_text()
    .tokenize(tokenizer_id="gpt2", field="text")
    .batch(32)
)
```

The prefetch operator is automatically inserted before the implicit fetch operator in the pipeline.

## The Problem: FetchOp Blocking on Downloads

Zephon pipelines overlap work across operators and stages. The planner may fuse
Fetch, Decode, and Tokenize into one stage or split them across stages; use
`pipeline.explain()` to inspect the compiled plan for a particular pipeline.
However, **within the FetchOp stage**, downloads can still block processing.

Example: Processing 3 batches. Download time = 100ms per batch of shards. Full pipeline processing time (Fetch + Decode + Tokenize) = 100ms per batch.

**Without Prefetch:**

```
FetchOp must download each batch's shards before it can read:

Batch 1:
  0-100ms:  Download Shards A-D (blocks FetchOp)
  100-120ms: Read samples from shards → emit batch 1

Batch 2:
  120-220ms: Download Shards E-H (blocks FetchOp)
  220-240ms: Read samples from shards → emit batch 2

Batch 3:
  240-340ms: Download Shards I-L (blocks FetchOp)
  340-360ms: Read samples from shards → emit batch 3

Timeline (FetchOp stage):
  0        100  120      220  240      340  360ms
  [Download][R][Download][R][Download][R]
      ↑ Blocked!    ↑ Blocked!   ↑ Blocked!

Total time: 360ms
Bottleneck: Download latency blocks each batch
```

**With Prefetch:**

```
PrefetchOp downloads shards BEFORE FetchOp needs them:

Background (PrefetchOp):
  0-100ms:   Download Shards A-D
  100-200ms: Download Shards E-H
  200-300ms: Download Shards I-L

Foreground (FetchOp):
  0-100ms:   Wait for warmup (initial download)
  100-120ms: Read Shards A-D (already cached!) → emit batch 1

  Meanwhile pipeline processes batch 1 (Decode, Tokenize: 100ms)

  200-220ms: Read Shards E-H (already cached!) → emit batch 2

  Meanwhile pipeline processes batch 2 (100ms)

  300-320ms: Read Shards I-L (already cached!) → emit batch 3

Timeline:
  Prefetch: [Download A-D][Download E-H][Download I-L]
  FetchOp:  [Wait warmup][R]          [R]          [R]
  Pipeline:              [Process B1] [Process B2] [Process B3]

Total time: 320ms (FetchOp done emitting all batches)
Speedup: 360ms → 320ms = 1.1x faster
Benefit: FetchOp never blocks - shards are always ready
```

**Key Insight:**
- **Without prefetch**: FetchOp repeatedly blocks on downloads (serial bottleneck)
- **With prefetch**: FetchOp always reads from cache (no blocking)
- Prefetch keeps cache warm while rest of pipeline processes batches
- **Works when**: Pipeline processing time per batch ≥ Download time per batch
- If processing is too fast, pipeline still waits for downloads

## Architecture Overview

### Pipeline Structure

```
                                    ┌─────────────────────────┐
                                    │   Background Threads    │
                                    │                         │
                                    │  Thread 1: Download S1  │
                                    │  Thread 2: Download S2  │
                                    │  Thread 3: Download S3  │
                                    │  Thread 4: Download S4  │
┌──────────────┐                    └──────────┬──────────────┘
│              │                               │
│  WorkSource  │                               ▼
│              │                    ┌────────────────────────┐
└──────┬───────┘                    │   Local Disk Cache     │
       │                            │                        │
       │ Emits:                     │  /tmp/zephon-cache/    │
       │ (dataset_id,               │    shard_0.bin         │
       │  shard_id,                 │    shard_1.bin         │
       │  sample_idx)               │    shard_2.bin         │
       │                            │    ...                 │
       ▼                            └──────────▲─────────────┘
┌─────────────────┐                           │
│  PrefetchOp     │                           │
│                 │                           │
│  ┌───────────┐  │                           │
│  │ Lookahead │  │  Triggers downloads       │
│  │  Buffer   │  │◄──────────────────────────┘
│  │ (Large)   │  │
│  │           │  │
│  │ S1 S1 S2  │  │  Scans buffer:
│  │ S2 S3 S4  │  │  "S3 coming up, prefetch!"
│  │ S4 S5 S6  │  │
│  └───────────┘  │
│                 │
│  Yields samples │
│  (unchanged)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   FetchOp       │
│                 │  Reads from cache
│  open(shard_1)  │◄─────────────────
│     CACHED!     │  (instant!)
└────────┬────────┘
         │
         ▼
   [Rest of Pipeline]
   DecodeText → Tokenize → Batch
```

### Key Components

1. **Lookahead Buffer**: Buffers samples using `CountingAccumulator` to provide visibility into upcoming samples
2. **Prefetch Scheduler**: Scans buffer to identify unique `(dataset_id, shard_id)` tuples needing download
3. **Stage Parallelism**: Multiple worker threads download shards concurrently using Zephon's stage parallelism
   - Each worker processes a batch of samples
   - Downloads are **blocking/synchronous** within each worker thread (calls `resolver.resolve()` and waits)
   - Parallelism comes from multiple workers processing different batches simultaneously
   - Not async/non-blocking - relies on stage-level concurrency
4. **Determinism Guarantee**: Downloads don't affect sample order - samples yielded in exact input order

## Configuration

### Parameters

```python
pipeline.prefetch(
    buffer_size=1024,           # Lookahead window size
    parallelism=4,              # Number of concurrent download workers (default: 4)
)
```

**buffer_size**: How many samples to buffer for lookahead
- Larger → more prefetch opportunities, more memory
- Typical: 512-2048 samples
- Memory usage: ~64 KB per 1024 samples (negligible)
- All shards appearing in the buffer are prefetched
- **Important**: Prefetch buffer should be **much larger** than fetch buffer for effective prefetching
  - Prefetch buffer (e.g., 1024 samples) provides lookahead for downloading shards
  - Fetch buffer (e.g., 64 samples default) processes samples for extraction
  - The gap between them creates the prefetch window

**parallelism**: Number of concurrent download workers
- Default: 4 (good balance for most workloads)
- Typical: 2-8 workers (balance between throughput and rate limits)
- Higher values increase download throughput but may hit bandwidth or rate limits
- Uses Zephon's stage parallelism (no separate thread pool)
- Downloads are **blocking** within each worker thread (synchronous calls to resolve())

## When to Use Prefetch

### High Impact ✅

- **Remote storage (S3, GCS)**: High download latency (50-500ms per shard)
- **Many shards with frequent transitions**: More opportunities to prefetch ahead
- **Sequential or predictable access patterns**: Lookahead can predict future needs
- **Processing time ≥ download time**: Downloads can be fully hidden behind processing

### Low Impact ⚠️

- **Local storage**: Already fast (no network latency to hide)
- **Random access patterns**: Limited prediction capability within lookahead window
- **Already cached data**: Prefetch is no-op (but also no harm)
- **Very fast processing (processing << download time)**: Pipeline still waits for downloads

## Testing and Observability

### Metrics Collection

Enable node-level execution tracking, then read the prefetch timing snapshot
from the pipeline (see `zephon.observability.PrefetchTimingSummary`):

```python
from zephon.observability import ExecutionTrackingMode

pipeline = pipeline.options(execution_tracking=ExecutionTrackingMode.NODES)

for batch in pipeline:
    ...  # train / consume

prefetch = pipeline.prefetch_timing_snapshot()
for stage_index, totals in prefetch.stages.items():
    print(f"Stage {stage_index}:")
    print(f"  Prefetch requests: {totals.prefetch_requests}")
    print(f"  Succeeded: {totals.prefetch_succeeded}")
    print(f"  Failed: {totals.prefetch_failed}")
```

Snapshots may also be taken mid-run (e.g. from a monitoring thread); they
return a consistent clone of the counters accumulated so far, and `None`
when `execution_tracking` is off.

MTP snapshots are not yet available while observability support is still in
development.

### Cache Performance

Monitor cache hit/miss rates via the fetch timing snapshot. Per-stage breakdowns
in `FetchStageSummary.shard_totals` are keyed by `(dataset_id, shard_id)`:

```python
fetch = pipeline.fetch_timing_snapshot()
for stage in fetch.iter_stages():
    totals = stage.totals
    print(f"Cache hit rate: {totals.cache_hit_ratio * 100:.1f}%")
    print(f"  Hits: {totals.cache_hits}, misses: {totals.cache_misses}")
```

## Complete Example

See `examples/run_with_prefetch.py` in the repository for a complete working example demonstrating:
- Pipeline construction with prefetch
- Configuration for different scenarios
- Performance comparison with/without prefetch
- Metrics collection and analysis
