# Inspecting and Tuning Pipelines

Once you have defined the `Dataset`, `WorkSource`, and the set of `Pipeline` operators
that you want to apply during data preparation, the next thing to focus on is ensuring
that the `Pipeline` can prepare the data quickly enough to keep the training loop supplied
with batches. A data loading system that spends too much time downloading shards or
transforming records leaves the very expensive training hardware waiting around for its
next input, so it is critical to understand how the data pipeline is performing and
identify any bottlenecks in the data preparation process so as to keep the GPUs hot.

To avoid so-called data stalls — the GPUs being idle because they are not supplied fast
enough with input tensors — the data loader must produce these tensors faster than the GPU
can consume them. In simple pipelines (e.g., our fetch-and-batch example) this is
typically not a problem. However, when operations such as image decoding are introduced,
this can, depending on the compute requirements, become a bottleneck; academic researchers
have investigated how to avoid such data stalls since the late 2010s (see, e.g., Mohan et
al., [Analyzing and Mitigating Data Stalls in DNN Training](https://arxiv.org/abs/2007.06775),
VLDB 2021).

Zephon provides default execution settings that are a reasonable starting point for most
workloads. This page explains how to inspect those settings, measure and understand where
a `Pipeline` is spending its time during data loading, and adjust the execution parameters
to ensure that the data pipeline does not become a bottleneck.

**Measuring Where The Time Goes.** Before starting an actual training run, run the
`Pipeline` with representative data on hardware that matches the specifications you expect
to use during training. Starting a particular `Pipeline` run may require downloading
shards, initializing the tokenizer, and filling buffers or bins, so it is useful to
distinguish the initial startup time for a `Pipeline` from the steady-state time required
to deliver each incremental batch of data to the training loop.

To help debug performance issues, call `Pipeline.enable_observability()` before iteration
begins, and then periodically inspect the metrics that are collected while the `Pipeline`
is running:

```{literalinclude} ../../../examples/guide/pipelines/enable_observability.py
:language: python
:caption: examples/guide/pipelines/enable_observability.py
```

The `metrics_snapshot` method reports on the activity within the `Pipeline`, while the
`fetch_timing_snapshot` method provides additional information about the time spent
fetching and reading the underlying data samples. If you have enabled shard prefetching
for datasets stored in remote storage locations, the `prefetch_timing_snapshot` method
provides similar metrics about the prefetch operations. Keep in mind that tracking these
metrics introduces a fair amount of overhead (especially when using the `process` runner),
so leave observability off for production runs.

The purpose of these measurements is to identify which part of the `Pipeline` run is
limiting the throughput. High fetch times suggest that we need to look at the time spent
downloading shards, opening files, and reading records, while large amounts of time spent
inside of individual transformations indicate that we may need to optimize their
implementation or provide them with additional computing resources. Remember, we do not
need to bring more compute resources to bear unless the `Pipeline` is causing the training
loop to stall — increasing the concurrency and compute resources of a `Pipeline` beyond
what the training pipeline needs increases resource usage without a benefit to our overall
throughput.

:::{warning}
We did not have the opportunity yet to invest as much in the observability infrastructure
as we would have liked to. There might be IPC overhead when you use the process runner. We
appreciate contributions around this area and are also excited to look into this more as
the project matures.
:::

**Fetching Remote Data.** When training on datasets that are kept in object storage, the
initial fetch operation the `Pipeline` implicitly performs may need to wait for the shard
to be downloaded (and possibly decompressed) before the requested records can be read. The
caching settings described in
[Storage Backends and Shard Cache](../datasets/storage_backends_and_shard_cache.md)
control where downloaded shards live and how much local storage is available for them. The
Pipeline's `prefetch` operation complements that cache by looking ahead at what samples
will be needed soon and downloading those shards before the `fetch` operator needs to open
them for reading:

```{literalinclude} ../../../examples/guide/pipelines/prefetch_shards.py
:language: python
:caption: examples/guide/pipelines/prefetch_shards.py
```

The `buffer_size` is measured in samples (as opposed to e.g. tokens), while the
`parallelism` setting on the `prefetch` operator controls the number of concurrent
prefetch workers. Using a larger value for these settings can allow the pipeline to
download shards further ahead of the `fetch` operator, but the downside of this is that it
is possible for the extra shards to put pressure on the size of the `Dataset` cache and
cause shard evictions that can be counterproductive for throughput. Prefetching is most
useful when downloading and decompressing shards from remote stores contributes
meaningfully to fetch latency; it is unnecessary when all of the training data can easily
fit in fast local storage.

**Adjusting Processing and Prefetching Training Batches.** Every `Pipeline` operator
except for `batch` exposes a `parallelism` argument that can be used to increase the
amount of compute resources devoted to them in a way that is appropriate for the operator:

```{literalinclude} ../../../examples/guide/pipelines/operator_parallelism.py
:language: python
:caption: examples/guide/pipelines/operator_parallelism.py
```

You can also control how far the `Pipeline` prepares output batches ahead of the training
loop:

```{literalinclude} ../../../examples/guide/pipelines/prefetch_batches.py
:language: python
:caption: examples/guide/pipelines/prefetch_batches.py
```

Unlike shard prefetching, this setting controls buffering the Pipeline's outputs.
Additional buffering can absorb short variations in data preparation time, but like other
buffers, it consumes additional memory and cannot compensate when the data pipeline
produces batches more slowly than the model can consume them.

As a general approach to measuring and improving execution performance, change one setting
at a time and compare the throughput and resource usage over a representative run. Keep
the sample preparation, shuffle, mixture, and packing settings identical while you are
doing this, since those settings impact the training data itself.


## Process payload memory

Process stages use one policy for supported CPU tensors, NumPy arrays, bytes,
memoryviews, bytearrays, and homogeneous numeric lists. Configure it with
`Pipeline.options(...)`:

| Option | Default | Effect |
|---|---:|---|
| `shm_min_item_bytes` | 4 KiB | Minimum item size eligible for a fresh SHM allocation. Smaller private values travel inline. Existing shared views are considered together first. |
| `shm_min_new_allocation_bytes` | 2 MiB | Minimum useful bytes per new allocation in the message. Smaller groups travel inline. |
| `shm_min_forward_bytes` | 128 KiB | Minimum useful bytes per existing shared allocation in the message. Smaller groups travel inline. This considers the combined views of that allocation, not each view separately. |
| `shm_coalesce` | `True` | Coalesce eligible values of compatible types/dtypes together. Applies to private values and compacted views alike. |
| `shm_max_coalesced_bytes` | 16 MiB | Maximum coalesced allocation size. Larger individual values still use SHM, in separate allocations. `None` removes the limit. |
| `shm_compact_above_ratio` | 8 | Copy shared views into compact storage when their backing allocation is more than this many times larger than their combined useful bytes in the message. `None` disables view compaction. |
| `shm_compact_min_savings_bytes` | 16 MiB | View compaction also requires at least this many potential savings: backing bytes minus useful bytes in this message. |

`shm_min_size` and `coalesce_tensors` remain accepted as keyword aliases for
`shm_min_item_bytes` and `shm_coalesce`.

Disabling coalescing leaves the transport thresholds and view compaction active.
Private values and compacted views follow the same path: apply the item floor,
form compatible groups up to the cap, then apply the new-allocation minimum.
For example, two compacted 1 MiB views can share one 2 MiB destination.
With a 512 KiB allocation minimum, two 256 KiB views take the same shared path.
Each is copied straight from its source view into that final allocation.
A 64 MiB batch of eligible 1 MiB values forms four 16 MiB allocations. A single
64 MiB value gets its own allocation; the cap never splits an individual item.
Array values, shapes and dtypes, and container structure are preserved; copying
a strided view can change its strides and contiguity. Unsupported array dtypes
and tensor layouts retain their normal serialization behavior.
Bytes, memoryviews and bytearrays selected for SHM use the same bytes-like
representation. On Python 3.12+, operators receive SHM-backed buffer views,
including after lazy payload resolution. This preserves the contents and buffer
interface, rather than every method of the original type. Converting these
views with `bytes(value)` makes a copy. Python 3.10 and 3.11 use the existing
copying `bytes` subclass fallback. Inline bytearrays remain bytearrays.

View compaction considers all supported nonempty views of the same allocation in the
message. A batch that still uses most of a slab will keep it. Overlapping views
are counted separately, which can conservatively skip a useful compaction.
Other samples or pending commands may still reference the original allocation,
delaying its release. The policy does not track references outside the message,
repack the contents of a slab, or revisit samples in shuffle buffers.
If a destination containing compacted views cannot allocate SHM, those views
keep their original backing. Private values in that group travel inline. This
avoids waiting for space held by the very views being compacted.

Existing shared views are grouped by backing allocation first. A group below
`shm_min_forward_bytes` travels inline; otherwise it keeps its backing unless
both compaction conditions hold. Views selected for compaction join the same
fresh-allocation path as private values, including the per-item floor and cap.
Compaction requires
**both** the relative and absolute conditions:
`backing_bytes > useful_bytes * shm_compact_above_ratio` and
`backing_bytes - useful_bytes >= shm_compact_min_savings_bytes`. The relative condition
limits copying large values for modest savings; the absolute condition avoids
allocating many small buffers for little potential benefit.

The default compaction minimum equals the coalescing cap, so unchanged views
of newly coalesced buffers are not immediately copied out again. Raising the
cap or lowering the compaction minimum can make those views qualify; tune the
settings together. Compaction can still be useful for large individual values
that were given their own allocations.

The transport minimums serve different purposes. A per-value floor avoids
copying and describing thousands of tiny entries, even if their total is large.
Passing that floor alone does not make a new allocation worthwhile. The
allocation floor amortizes setup and handle transfer across enough data:
64 eligible 32 KiB values can share a 2 MiB allocation, while a single 32 KiB
value stays inline. Existing shared allocations use the forwarding minimum;
forwarding a handle can be worthwhile even when allocating a new buffer is not.
The defaults balance measurements on Linux and macOS, under pickle protocols
4 and 5. New 512 KiB and 1 MiB allocations often lost to inline transport on
Linux; 2 MiB improved the balance. The 4 KiB item floor allows large batches
of small values to coalesce while avoiding measured Torch regressions at
1-2 KiB per item. Neither threshold is a universal crossover. In particular,
macOS can benefit from smaller new allocations. Forwarding many separate
128 KiB allocations can lose to inline transport on Linux, but raising the
common forwarding minimum would slow those cases substantially on macOS.

Inline values in a microbatch travel in one queue message,
not one message per value. Benchmark representative batch sizes when tuning.

These settings apply to process stages in both directions. They do not configure
the final MTP queue. Inline Torch values use a Zephon transport descriptor; Torch's
global multiprocessing reducer is unchanged.
