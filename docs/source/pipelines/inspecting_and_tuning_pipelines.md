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

Process stages send supported CPU payloads (Torch tensors, NumPy arrays,
bytes-like values and homogeneous numeric lists) either inline in the queue
message or through shared memory (SHM). Values that are large enough are copied
once into shared buffers, coalesced by compatible type/dtype. Views of existing
shared buffers can be forwarded by handle. When a message uses only a small
part of a large buffer, those views can be copied out ("compacted") to reduce
how much memory a retained sample keeps alive.

| Option | Default | Effect |
|---|---:|---|
| `shm_enabled` | `True` | `False` disables Zephon's SHM preparation. Ordinary multiprocessing pickling still applies, so Torch may still share tensor storage. |
| `shm_min_item_bytes` | 4 KiB | Minimum item size eligible for fresh SHM. Existing shared views are considered together first. |
| `shm_min_new_allocation_bytes` | 1 MiB | Create a shared buffer only if its group reaches this size; smaller groups travel inline. |
| `shm_min_forward_bytes` | 128 KiB | Forward an existing shared buffer only if the message uses at least this much of it; otherwise send those views inline. |
| `shm_coalesce` | `True` | Combine compatible values into common buffers. `False` considers each value separately; transport thresholds still apply. |
| `shm_max_coalesced_bytes` | 64 MiB | Upper bound for a coalesced buffer (`None`: no bound). Larger single values remain unsplit, subject to the transport minimum. |
| `shm_compact_above_ratio` | 8 | Compact views when their buffer is more than this many times larger than the bytes the message uses (`None` disables compaction) … |
| `shm_compact_min_savings_bytes` | 16 MiB | … and compaction would save at least this many bytes of backing. Other references can keep the original buffer alive. |

The best thresholds depend on payload types, batch sizes and the host, so
benchmark representative batches before changing them. Lowering
`shm_max_coalesced_bytes` reduces how much memory one retained sample can keep
alive, at the cost of more allocations. Compaction applies at process transport
boundaries; it does not revisit samples already held in shuffle buffers.

On Python 3.12+, bytes-like values sent through SHM arrive as read-only,
zero-copy buffer views; on 3.10 and 3.11 they arrive as `bytes` subclasses.
A `bytearray` sent through SHM therefore loses its mutability. Inline bytearrays
remain bytearrays. `shm_min_size` and `coalesce_tensors` are deprecated aliases
for `shm_min_item_bytes` and `shm_coalesce`. These options apply in both process-stage
directions and do not configure the final queue to the training process (MTP).
