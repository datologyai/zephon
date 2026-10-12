# Elastic Determinism

[Why Zephon?](../why_zephon.md) claims that a run resumed on a different number of GPUs
sees the same global batches, and [Distributed
Training](../pipelines/distributed_training.md) tells you which options to set in order to
get that. This page explains the mechanism that makes the claim true. It also explains the
boundaries of it, which matter just as much.

**Determinism by induction.** Zephon's determinism guarantee is compositional, which is a
grand way of saying that it is proven the boring way. The `WorkSource` produces pointers
in a fixed order for a given seed and lane. Each operator is deterministic given its
inputs. By induction over the operator chain, the whole pipeline therefore produces the
same output for the same configuration and the same starting state. There is no global
coordination step that fixes up the order after the fact. There is just a chain of
individually boring components.

This is why the guarantee survives things that would break a less structured design. You
can checkpoint in the middle of training, restart on different hardware, and get the same
global sample order. The one condition is that the pipeline definition stays the same: the
seed, the operators, the mixture weights, and the number of lanes. The induction does not
care which machine runs which step.

The determinism is on by default and can be turned off with
`Pipeline.options(deterministic=False)`, which lets each stage forward results in whatever
order they finish instead of in input order. That can shave latency off a serving-style
workload. It also means elastic determinism no longer holds, because the per-lane ordering
that the rest of this page depends on is exactly what gets relaxed.

(execution-parameters-are-not-data-parameters)=

**Execution parameters are not data parameters.** The single most useful property Zephon
has is a negative one: **changing how a pipeline runs never changes what it produces.**
None of the execution settings appear anywhere in the data path, including the runner
type, the parallelism of each operator, the queue depths, the prefetch settings, and the
per-device microbatch size. You can tune a pipeline for a week, and the experiment you ran
on the first day is still comparable to the one you run on the last.

Two mechanisms buy this, and [Operators and Runners](operators_and_runners.md) covers both
in detail. Micro-batches get a sequence number before they are dispatched and are put back
in order before they are forwarded, so parallelism cannot reorder anything. And every
decision that depends on more than one record lives in a single-threaded accumulator that
sees records in stream order, so parallelism cannot change the decisions either. That
covers which sequences get packed together, which records a shuffle buffer holds back, and
everything else of that kind.

It is worth dwelling on how unusual this is, because the alternative is the normal case.
[Megatron Energon](https://nvidia.github.io/Megatron-Energon/advanced/repro_scaling.html)
documents that reproducibility requires the global worker count
(`world_size * num_workers`) to stay fixed, which couples a performance setting to the
data order: tune the loader, lose the comparison. [Mosaic
Streaming](https://docs.mosaicml.com/projects/streaming/en/latest/distributed_training/elastic_determinism.html)
avoids that with a careful partitioning algorithm, but requires a constant global batch
size across runs. In both cases a knob you would like to treat as free is not free.

**The problem with partitioning by GPU.** The naive way to distribute data is to partition
it across the physical GPU count: with 8 GPUs, GPU *k* takes every eighth sample. Resume
that run on 4 GPUs and every GPU's stream changes completely, because the partition
function changed. The run is not a continuation of the old one in any meaningful sense. It
is a new run that happens to start from an old set of weights.

The standard fix is a layer of indirection between the logical data order and the physical
hardware. Mosaic Streaming calls its version **canonical nodes**, a fixed number of
logical partitions that are mapped onto physical nodes at runtime. Keep the number of
canonical nodes constant and the global sample order survives a change in node count.

**Zephon's version: lanes.** A **lane** is a logical, deterministic sub-stream of the
global sample order, and it is the natural equivalent of a canonical node for an iterable
pipeline. The number of lanes is fixed for the lifetime of a run by the
`canonical_replicas` option, and the lanes are partitioned among whatever data-parallel
groups the current run happens to have.

`canonical_replicas` is a hard upper bound on `dp_degree`, because every data-parallel
group must own at least one lane. Set it to 32 and you can run on anything from 1 to 32
groups, but you cannot run on 64. Per-lane sample streams are identical in every one of
those cases. The only thing that changes is which group owns which lane:

```
canonical_replicas = 8, dp_degree = 2

  group 0 owns lanes [0, 1, 2, 3]      group 1 owns lanes [4, 5, 6, 7]

Resume with dp_degree = 4:

  group 0: [0, 1]   group 1: [2, 3]   group 2: [4, 5]   group 3: [6, 7]

Resume with dp_degree = 8:

  group 0: [0]  group 1: [1]  group 2: [2]  group 3: [3]  group 4: [4]  ...
```

Two things cannot change across a resume without changing the curriculum: the seed and
`canonical_replicas`. The per-lane order is a function of both, so changing either
produces a different global schedule, and Zephon refuses to restore a checkpoint whose
`canonical_replicas` does not match. Everything else is free, including the parallelism,
the runner type, the prefetch depth, and the per-device microbatch size. The one
qualification is the batches-per-step rule in
{ref}`planning for elasticity <planning-for-elasticity>`.

**Samples do not travel in separate lane streams.** A design choice that surprises people:
Zephon does **not** physically separate the data into one stream per lane. Every record
carries its `lane_id` in its metadata, and operators process records from every owned lane
together, through the same queues and the same worker pools.

The reason is that the alternative is wasteful. Separate per-lane streams means separate
per-lane operator instances, each with its own buffers and its own slice of the worker
pool. A group that owns eight lanes would run eight tokenizer pools with an eighth of the
work each. Keeping the lanes mixed means one pool, sized once, shared across all of them.

The engine feeds the pipeline in fair round-robin across the lanes the group owns: one
sample from lane 0, one from lane 1, and so on, skipping lanes that have finished. That
keeps the lanes progressing at roughly the same rate, which matters because the output end
has to pull one batch from each lane in turn. How a lane works out which chunks belong to
it in the first place, without asking any other rank, is covered in
{ref}`Chunk-to-lane routing <chunk-to-lane-routing>`.

The cost of mixing is that any operator holding per-record state has to be **lane-aware**,
or it will mix data across lanes and destroy the per-lane determinism that the whole
scheme rests on. Packing keeps separate bins per lane, so a sequence from lane 0 is never
packed together with one from lane 3. Batching fills separate per-lane buffers. The
shuffle buffer keeps separate reservoirs. Stateless operators never have to think about
lanes at all, which covers maps, tokenization, filters, and most of the rest.

**The tail multiplexer.** Because lanes are interleaved on the way in and operators like
packing and filtering change the rate at which each lane produces output, the raw output
of the pipeline does not emit lanes in any predictable order. A group that owns lanes 0
through 3 might see batches finish as lane 2, lane 0, lane 2, lane 1, and so on.

The training loop needs better than that. Each optimizer step has to contain a whole
number of round-robin cycles across the lanes, or the step composition depends on the
topology. So the engine ends every pipeline with a **tail multiplexer** that buffers
output per lane and emits in strict round-robin:

```
Raw pipeline output (arrival order varies with processing time):

  ... lane2  lane0  lane2  lane1  lane3  lane0  lane1  lane3 ...
         │      │      │      │      │      │      │      │
         v      v      v      v      v      v      v      v
  ┌──────────────────────────────────────────────────────────────┐
  │                     Per-lane buffers                         │
  │   lane 0: [batch, batch, ...]    lane 2: [batch, ...]        │
  │   lane 1: [batch, batch, ...]    lane 3: [batch, ...]        │
  └───────────────────────┬──────────────────────────────────────┘
                          │  strict round-robin emission
                          v
  Training loop sees:  lane0  lane1  lane2  lane3  lane0  lane1 ...
```

A group that owns exactly one lane skips the multiplexer entirely and passes its upstream
through unchanged, which is the common case for a run at full width. Flush sentinels pass
through without consuming a turn, since they are bookkeeping rather than data.

(what-actually-survives-a-topology-change)=

**What actually survives a topology change.** The guarantee rests on two pieces of state
that behave very differently, and the difference is the thing most worth understanding on
this page:

- **Per-lane progress is the source of truth.** Each lane independently records how far it
  has advanced, as a chunk ID and an offset within that chunk. This state is durable,
  topology-independent, and checkpointed. It is what guarantees that no sample is ever
  skipped or delivered twice, no matter how the lanes are redistributed. This guarantee is
  unconditional.
- **The round-robin pointer is a physical detail.** It records which lane the tail
  multiplexer should emit from next. It is keyed by the DataLoader worker, the number of
  active workers, and the exact list of owned lanes, deliberately not by rank, so that
  peers inside a data-parallel group own identical lanes and share one entry. When the
  topology is unchanged, the key still matches on restore, the pointer is restored
  exactly, and the per-group output is identical sample for sample to an uninterrupted
  run. When the topology changes, no saved key matches the new ownership, so the engine
  recomputes a starting point from per-lane progress instead, picking the least-advanced
  owned lane ordered by `(chunk_id, offset, lane_id)`.

That recomputation is why checkpoints belong at global step boundaries. At the end of a
step every lane has contributed exactly once per cycle, so every lane is at the same
frontier. The tiebreaker then resolves to the same place regardless of topology, and the
recomputed pointer starts where the saved one would have. Checkpoint in the middle of a
step and the lanes are at different frontiers. The no-skip and no-duplicate guarantees
still hold, but the exact interleaving across the remap is not promised.

What is preserved, then, is the **global training step**: the union of the per-lane
batches across all groups contains the same samples whatever the topology. What is not
preserved is the assignment of those samples to physical GPUs, which is the part nobody
should care about. With eight groups and eight lanes, each group handles one lane's batch.
With two groups, each handles four lanes' batches through the tail round-robin. Same
samples, different machines. The same holds when the per-device microbatch size changes. A
different microbatch size changes how many samples a lane puts in one local batch, but the
round-robin still cycles the same lanes producing the same data.

**Mapping strategies.** When a group owns several lanes, something has to decide which
ones. Two strategies are available through `Pipeline.options(mapping_strategy=...)`. Both
are deterministic and produce identical global batches, so they only decide which group
does which work.

`"contiguous"` is the default and gives each group a contiguous block of lane IDs. With
`canonical_replicas=8` and `dp_degree=3` the groups get `[0,1,2]`, `[3,4,5]`, and `[6,7]`.
Adjacent lanes tend to read adjacent data, so this is the choice that maximises shard and
block-cache locality.

`"interleaved"` deals lanes round-robin, so lane *i* goes to group *i % num_groups*. With
the same numbers the groups get `[0,3,6]`, `[1,4,7]`, and `[2,5]`. This spreads
heterogeneous shards more evenly and tends to reduce tail variance between groups.

Note the uneven split in both examples. Eight lanes across three groups has to leave
somebody short. That is the reason for the divisibility advice in
{ref}`planning for elasticity <planning-for-elasticity>`: a lane count that is not
divisible by the group count means some groups own more lanes than others, and therefore
more work per step.
