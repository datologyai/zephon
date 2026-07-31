# (Elastic) Determinism

Zephon can guarantee that a pipeline produces the exact same sequence of
training batches across runs and restarts.  With elastic determinism, this
extends to changes in the number of GPUs: the global set of training samples
per step is preserved even when the hardware topology changes.  This page
explains how the determinism guarantees work, what makes them hold across
different execution configurations, and how elastic determinism lets you
scale GPU counts without changing the data order.

---

## Determinism by Induction

Zephon's determinism guarantee is **compositional**: each part of the
pipeline is individually deterministic, so the pipeline as a whole is
deterministic.

1. The {py:class}`~zephon.work.WorkSource` produces pointers in a fixed order
   for a given seed.
2. Each operator in the pipeline is deterministic given its inputs.
3. By induction, the entire pipeline produces the same output for the same
   configuration and checkpoint state.

This means you can checkpoint mid-training, restart on different hardware, and
get exactly the same global sample order, provided the pipeline
configuration (seed, operators, mixture weights, etc.) stays the same.

Determinism is enabled by default and can be turned off via
{py:meth}`Pipeline.options(deterministic=False) <zephon.Pipeline.options>`.
When disabled, runners gain more scheduling flexibility (forwarding results in
completion order rather than input order), which can reduce latency at the
cost of reproducibility.

### How it works

Each operator stage is executed deterministically by its runner.  In
deterministic mode, the runner tags every micro-batch (a group of samples
dispatched together) with a monotonically increasing sequence number before
dispatching it to workers.  Results are reordered by this number before being
forwarded downstream, so the output is identical to a single-threaded
execution regardless of how many workers run in parallel.

Operators that need cross-invocation state, e.g., packing bins or batch buffers,
keep that state in **accumulators** that run in a single thread on the
runner's pump thread (the serial thread that receives all upstream elements
in order).  The accumulator decides batch boundaries; workers only execute
stateless `process_many` calls.  This ensures, for example, that packing
decisions do not depend on how work is distributed across workers.

A related pattern is **data-derived pseudo-randomness**: operators like
``ShuffleBuffer`` derive their RNG seed from the identity
of the elements in each batch (dataset, shard, sample offset, lineage) rather
than from an incrementing counter.  The same set of elements always produces
the same permutation, regardless of restart boundaries or batch grouping.

For a deeper treatment of accumulators and the operator execution model, see
[Accumulators and Operators](accumulators_operators.md).

### Execution parameters do not change data order

A key design goal in Zephon is that **execution parameters are pure
performance knobs**.  Changing the runner type (threads vs. processes),
operator parallelism, queue depths, or prefetch settings never alters the
sequence of training samples the model sees.  This separation follows
directly from the accumulator design and sequence-number reordering
described above.  You can freely tune performance without worrying about
invalidating a previous run's data order or breaking checkpoint resumption.

---

## Elastic Determinism

**Elastic determinism** goes beyond fixing the output data order: you can
change the number of GPUs between runs and still get the same global data
order.  This matters in practice because often we want to prototype on a
small cluster and later scale to more GPUs, or resume on different hardware
after a preemption.  Without elastic determinism, every topology change
produces a different training run, making it impossible to compare results or
continue a run faithfully.

```{note}
Elastic determinism requires deterministic mode (`deterministic=True`, the
default).  Without it, per-lane ordering is not guaranteed and the tail
multiplexer cannot reconstruct a consistent global batch.
```

### The problem

In a naively distributed pipeline, data is partitioned directly across the
physical GPU count.  If you trained on 8 GPUs and resume on 4, each GPU's
data stream changes completely.
The common solution is to introduce a layer of indirection between the
logical data order and the physical hardware.
[Mosaic Streaming](https://docs.mosaicml.com/projects/streaming/en/latest/distributed_training/elastic_determinism.html)
solves this with **canonical nodes**: a fixed number of logical
partitions that are mapped onto physical nodes at runtime.  By keeping the
number of canonical nodes constant, the global sample order is preserved
even when the number of GPUs changes.  Streaming also requires a constant
global batch size across runs.
[Megatron Energon](https://nvidia.github.io/Megatron-Energon/advanced/repro_scaling.html)
offers a similar feature but requires that the global number of workers
(`world_size * num_workers`) stays constant across runs, coupling
the parallelism configuration to the data order.

In both systems, execution settings that should be pure performance knobs
can influence the data ordering.  Energon explicitly notes this constraint;
Mosaic Streaming works around it with a clever partition algorithm, but
the coupling requires careful engineering to avoid (see
[Transitioning from Streaming](../transitioning.md) for the full story).
As described [above](#execution-parameters-do-not-change-data-order),
Zephon avoids this coupling entirely.

### Lanes: Zephon's approach

Zephon uses **lanes**, i.e., logical, deterministic sub-streams of the global
sample order.  Lanes are the natural equivalent of Streaming's canonical
nodes for an iterable pipeline architecture.

```{note}
For simplicity, this section says "GPU" to mean "data-parallel group."
In setups with tensor or pipeline parallelism, multiple GPUs share the same
data stream. See [Distributed Training](distributed_training.md) for
details.
```

The number of lanes is controlled by `canonical_replicas` in
{py:meth}`Pipeline.options() <zephon.Pipeline.options>`.  If you do not
set `canonical_replicas`, it defaults to `dp_degree`, i.e., one lane per GPU,
no elastic determinism.  To enable it, set `canonical_replicas` higher than
`dp_degree`:

```python
pipeline = pipeline.options(
    dp_degree=4,            # current run: 4 GPUs
    canonical_replicas=32,  # plan for up to 32 GPUs
    # ...
)
```

`canonical_replicas` is an **upper bound** on the number of data-parallel
ranks you can use for this training run.  Each lane is an independent data
stream, and each GPU must own at least one lane. `dp_degree` can never
exceed `canonical_replicas`.  If you set `canonical_replicas=32`, you can
scale anywhere from 1 to 32 GPUs, but not to 64.  Choose the value to cover
the largest cluster size you realistically expect to scale to.

Each GPU is then responsible for one or more lanes.  In this example, each
of the 4 GPUs owns 8 lanes.  If you later resume on 8 GPUs, each GPU owns
4 lanes; on 32 GPUs, each owns exactly 1.  Crucially, the per-lane sample
streams are identical in all three cases; only the assignment of lanes to
GPUs changes.

```
canonical_replicas = 8, dp_degree = 2

GPU 0 owns lanes [0, 1, 2, 3]      GPU 1 owns lanes [4, 5, 6, 7]

Resume with dp_degree = 4:

GPU 0: [0, 1]   GPU 1: [2, 3]   GPU 2: [4, 5]   GPU 3: [6, 7]

Resume with dp_degree = 8:

GPU 0: [0]  GPU 1: [1]  GPU 2: [2]  GPU 3: [3]  GPU 4: [4]  ...
```

### How samples flow through lanes

Unlike systems that physically separate data into per-lane streams at the
source, Zephon keeps samples **unseparated** as they flow through the
pipeline.  Each sample carries a `lane_id` in its metadata, but operators
process samples from all lanes together.  This avoids duplicating operator
state and lets the pipeline benefit from shared parallelism across lanes.

At the **source**, the engine feeds samples into the pipeline in fair
round-robin across the owned lanes (one sample from lane 0, one from
lane 1, and so on).  This round-robin input ensures roughly even progress
per lane.

Operators that maintain per-element state need to be **lane-aware** so they
do not mix data across lanes.  Packing, for example, maintains per-lane bins
so that sequences from lane 0 are never packed together with sequences from
lane 3.  Batching similarly fills per-lane buffers.  Stateless operators
(maps, tokenization, filtering) do not need to care about lanes at all.

### The output multiplexer

Because samples from different lanes are interleaved through the pipeline,
and operators like packing or filtering can change the relative rate at which
lanes produce output, the pipeline's raw output stream does not emit lanes in
a predictable order.  A GPU that owns lanes [0, 1, 2, 3] might see batches
arrive as lane 2, lane 0, lane 2, lane 1, ...

The engine's **tail multiplexer** (`_lane_rr_iter`) fixes this.  It sits
at the very end of the pipeline and buffers output into per-lane queues,
then emits batches in strict round-robin order across the owned lanes:

```
Raw pipeline output (arrival order varies by processing time):

  ... lane2  lane0  lane2  lane1  lane3  lane0  lane1  lane3 ...
         │      │      │      │      │      │      │      │
         v      v      v      v      v      v      v      v
  ┌──────────────────────────────────────────────────────────────┐
  │                    Per-lane buffers                          │
  │                                                              │
  │  lane 0: [batch, batch, ...]                                 │
  │  lane 1: [batch, batch, ...]                                 │
  │  lane 2: [batch, batch, ...]                                 │
  │  lane 3: [batch, batch, ...]                                 │
  └───────────────────────┬──────────────────────────────────────┘
                          │
                          v  strict round-robin emission
  Training loop sees:  lane0  lane1  lane2  lane3  lane0  lane1 ...
```

### What is preserved across topology changes

The guarantee rests on two layers:

1. **Per-lane progress** is the source of truth.  Each lane independently
   tracks how far it has advanced (chunk ID and offset).  This state is
   durable, topology-independent, and checkpointed.  It guarantees that no
   sample is ever skipped or produced twice, regardless of how the topology
   changes.

2. **The round-robin pointer** determines which lane the tail multiplexer
   emits from next on a given GPU.  It is a physical-topology detail: it is
   keyed by the current rank, worker, and lane set.  When the topology
   stays the same, the pointer is restored exactly from the checkpoint and
   the per-GPU output is sample-by-sample identical to an uninterrupted run.
   When the topology changes, the old key no longer matches and the engine
   recomputes a starting point from per-lane progress.

For non-monotonic pipelines (those with packing, shuffling, or mixture
correction), the checkpoint additionally stores **epoch boundary positions**
and **inflight chunk state** so that flush sentinels can be re-injected at
the correct points during replay.  See
[Checkpointing](checkpointing.md#epoch-based-eviction-packing-and-shuffling) for the full model.

Checkpoints should be taken between global training steps, at which point
every lane has contributed exactly once per round-robin cycle.  All lanes
are at the same frontier, so the recomputed pointer always starts at the
same position regardless of topology --- the tiebreaker deterministically
picks the lowest lane ID.

The **global training step** (the union of per-lane batches across all
GPUs) therefore contains the same set of samples across topology changes.
What can differ is the assignment of lanes to GPUs: with 8 GPUs
(given `canonical_replicas=8`) each GPU processes one lane's batch, while
with 2 GPUs each GPU processes four lanes' batches via the tail round-robin.
The samples are the same; only the physical distribution changes.

This also holds when the per-device micro-batch size changes: a different
micro-batch size changes how many samples each lane packs into one local
batch, but the round-robin still cycles through the same lanes producing
the same data.  Unlike other elastic-determinism systems, micro-batch size
is therefore free to change across runs.  Our integration tests verify
these guarantees end-to-end across varying GPU counts, mapping strategies,
and micro-batch sizes.

### What users need to do

1. **Choose `canonical_replicas`.**  This is the hard ceiling on
   `dp_degree` for the lifetime of the training run --- you cannot use more
   GPUs than lanes.  Set it to the largest data-parallel degree you expect
   to scale to, and ideally make it divisible by every `dp_degree` you plan
   to use.

2. **Keep `canonical_replicas` and the seed constant.**  The per-lane data
   order is a function of both; changing either produces a different global
   schedule.  Changing `canonical_replicas` also invalidates existing
   checkpoints. See [Checkpointing](checkpointing.md) for details.

3. **Micro-batch size and execution parameters are free to change.**  As
   described [above](#execution-parameters-do-not-change-data-order),
   parallelism, runner type, prefetch settings, and even the per-device
   micro-batch size can change freely between runs without affecting the
   global data order.

```python
# Initial run: 8 GPUs, microbatch 4
pipeline = (
    Pipeline(ws)
    .tokenize(tokenizer_id="meta-llama/Llama-3-8B", field="text", max_length=4096)
    .batch(microbatch_size=4)
    .options(
        dp_degree=8,
        canonical_replicas=32,
        deterministic=True,
    )
)

# Resume on 4 GPUs, microbatch 2 --- same global batches if you use gradient_accumulation_steps = 4
pipeline = (
    Pipeline(ws)
    .tokenize(tokenizer_id="meta-llama/Llama-3-8B", field="text", max_length=4096)
    .batch(microbatch_size=2)
    .options(
        dp_degree=4,
        canonical_replicas=32,         # unchanged
        deterministic=True,
    )
)
```

### Note: Mapping strategies

When a GPU owns multiple lanes, the engine needs to decide *which* lanes
it gets.  Two strategies are available via
`Pipeline.options(mapping_strategy=...)`:

- **`"contiguous"`** (default) --- each GPU gets a contiguous block of lane
  IDs.  With `canonical_replicas=8` and `dp_degree=2`, GPU 0 gets
  lanes [0, 1, 2, 3] and GPU 1 gets [4, 5, 6, 7].  This maximizes shard
  locality: adjacent lanes tend to read adjacent data.

- **`"interleaved"`** --- lanes are dealt round-robin across GPUs.  GPU 0
  gets [0, 2, 4, 6], GPU 1 gets [1, 3, 5, 7]. 