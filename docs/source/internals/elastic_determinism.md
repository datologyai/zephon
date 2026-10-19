# Elastic determinism

[Why Zephon?](../why_zephon.md) claims that a run resumed on a different number of GPUs
sees the same global batches, and [Distributed
Training](../pipelines/distributed_training.md) tells you which options to set in order to
get it. This page explains the mechanism behind that claim, and just as importantly, where
its boundaries are.

**Determinism by induction.** Zephon's determinism guarantee is compositional, which means
we get it by making each piece deterministic rather than by coordinating the pieces. The
`WorkSource` produces pointers in a fixed order for a given seed and lane, and each
operator is deterministic given its inputs, so by induction over the operator chain the
whole `Pipeline` produces the same output for the same configuration and the same starting
state. There is no global coordination step that fixes up the order after the fact.

This is why the guarantee survives things that would break a less structured design. You
can checkpoint in the middle of training, restart on different hardware, and still get the
same global sample order, as long as the `Pipeline` definition stays the same: the seed,
the operators, the mixture weights, and the number of lanes. The induction does not care
which machine runs which step.

Turning determinism off with `Pipeline.options(deterministic=False)` therefore costs you
more than the ordering guarantee it names. Elastic determinism goes with it, because the
per-lane ordering that the rest of this page depends on is exactly what gets relaxed.

(execution-parameters-are-not-data-parameters)=

**Execution parameters are not data parameters.** The most useful property Zephon has is a
negative one: **changing how a `Pipeline` runs never changes what it produces.** Execution
settings (the runner type, the parallelism of each operator, the queue depths, the prefetch
settings, and the worker counts) never appear anywhere in the data path. You can tune them
for a week and the experiment you ran on the first day is still comparable to the one you
run on the last.

Batch shape (the microbatch size and the number of batches per step) and
`flush_every_k_chunks` are a different matter: they are data settings. Batch shape decides
which samples land together, and `flush_every_k_chunks` decides where packing and mixture
operators cut their buffers, so changing either changes the sample stream of any pipeline
that packs. Fix them at the start of a run, like the seed and `canonical_replicas`.

The two mechanisms that buy you this, sequence-number reordering and the single-threaded
accumulator, are the ones [How pipelines run](../pipelines/how_pipelines_run.md)
introduces and [Operators and runners](operators_and_runners.md) takes apart.

Not every loader separates the two. [Megatron
Energon](https://nvidia.github.io/Megatron-Energon/advanced/repro_scaling.html), for
example, documents that reproducibility requires the global worker count
(`world_size * num_workers`) to stay fixed, so tuning the loader costs you the comparison.

**The problem with partitioning by GPU.** The naive way to distribute data is to partition
it across the physical GPU count, so that with 8 GPUs, GPU *k* takes every eighth sample.
Resume that run on 4 GPUs and every GPU's stream changes completely, because the partition
function changed. The run is no longer a continuation of the old one in any meaningful
sense; it is a new run that happens to start from an old set of weights.

The standard fix is a layer of indirection between the logical data order and the physical
hardware. Mosaic Streaming calls its version **canonical nodes**, a fixed number of
logical partitions that are mapped onto physical nodes at runtime. Keep the number of
canonical nodes constant and the global sample order survives a change in node count.

**Zephon's version: lanes.** {ref}`Planning for elasticity <planning-for-elasticity>`
introduces the **lane** as the fixed logical data stream you size with
`canonical_replicas`, which is the natural equivalent of a canonical node for an iterable
pipeline. What matters here is the consequence: the per-lane sample streams are identical
at every `dp_degree`, and the only thing a topology change moves is which group owns which
lane. The owner is one GPU in a job without tensor or pipeline parallelism, and otherwise
the set of GPUs that share a `dp_group_id`.

```
canonical_replicas = 8, dp_degree = 2

  group 0 owns lanes [0, 1, 2, 3]      group 1 owns lanes [4, 5, 6, 7]

Resume with dp_degree = 4:

  group 0: [0, 1]   group 1: [2, 3]   group 2: [4, 5]   group 3: [6, 7]

Resume with dp_degree = 8:

  group 0: [0]  group 1: [1]  group 2: [2]  group 3: [3]  group 4: [4]  ...
```

The per-lane order is a function of the seed and `canonical_replicas`, so both are fixed
for the lifetime of a run. The seed is stored in the checkpoint and restored from it, so a
resume always continues with the original seed whatever the new `Pipeline` was configured
with. `canonical_replicas` is checked instead: Zephon refuses to restore a checkpoint whose
`canonical_replicas` does not match.

**Samples do not travel in separate lane streams.** 
Zephon does **not** physically separate the data into one stream per lane. Every record
carries its `lane_id` in its metadata, and the operators process records from every owned
lane together, through the same queues and the same worker pools.

We do it this way because the alternative is wasteful. Separate per-lane streams would
mean separate per-lane operator instances, each with its own buffers and its own slice of
the worker pool, so a group that owns eight lanes would end up running eight tokenizer
pools with an eighth of the work each. Keeping the lanes mixed means one pool, sized once, shared across all of them.

Zephon feeds the `Pipeline` in fair round-robin across the lanes the group owns, taking
one sample from lane 0, one from lane 1, and so on, skipping lanes that have finished.
That keeps the lanes progressing at roughly the same rate, which matters because the
output end has to pull one batch from each lane in turn. How a lane works out which chunks
belong to it in the first place, without asking any other rank, is covered in
{ref}`Chunk-to-lane routing <chunk-to-lane-routing>`.

What mixing costs us is that any operator holding per-record state has to be
**lane-aware**, or it will mix data across lanes and destroy the per-lane determinism that
the whole scheme rests on. Packing keeps separate bins per lane, so a sequence from lane 0
is never packed together with one from lane 3, batching fills separate per-lane buffers,
and the shuffle buffer keeps separate reservoirs. Stateless operators such as maps,
tokenization and filters never have to think about lanes at all, and that covers most of
them.

**The output multiplexer.** Because the lanes are interleaved on the way in, and operators
like packing and filtering change the rate at which each lane produces output, the raw
output of a `Pipeline` does not emit lanes in any predictable order. A group that owns
lanes 0 through 3 might see batches finish as lane 2, lane 0, lane 2, lane 1, and so on.

The training loop needs better than that, because each optimizer step has to contain a
whole number of round-robin cycles across the lanes or its composition depends on the
topology. So Zephon ends every `Pipeline` with a **tail multiplexer** that buffers output
per lane and emits in strict round-robin:

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

**What is preserved across topology changes.** The guarantee rests on two layers:

- **Per-lane progress is the source of truth.** Each lane independently records how far it
  has advanced, as a chunk ID and an offset within that chunk. This state is durable,
  topology-independent, and checkpointed, and it is what guarantees that no sample is ever
  skipped or delivered twice, no matter how the lanes are redistributed. That guarantee is
  unconditional.
- **The round-robin pointer is a physical detail.** It records which lane the tail
  multiplexer should emit from next, and it is keyed by the DataLoader worker, the number
  of active workers, and the exact list of owned lanes, deliberately not by rank, so that
  peers inside a data-parallel group own identical lanes and share one entry. When the
  topology is unchanged, the key still matches on restore, the pointer comes back exactly,
  and the per-group output is identical sample for sample to an uninterrupted run. When the topology changes, the old key no longer matches and the engine recomputes a starting point from per-lane progress.

That recomputation is why you want to checkpoint at global step boundaries. At the end of
a step every lane has contributed exactly once per cycle, so every lane sits at the same
frontier, the tiebreaker resolves to the same place regardless of topology, and the
recomputed pointer starts where the saved one would have. Checkpoint in the middle of a
step and the lanes are at different frontiers; the no-skip and no-duplicate guarantees
still hold, but we cannot promise you the exact interleaving across the remap.

What is preserved is the **global training step**: the union of the per-lane
batches across all groups contains the same samples whatever the topology. What can differ is the assignment of lanes to GPUs. With eight groups and eight lanes, each group handles one
lane's batch; with two groups, each handles four lanes' batches through the tail
round-robin. The samples are the same; only the physical distribution changes. The same holds when you change the
per-device microbatch size, because that changes how many samples a lane puts in one local
batch while the round-robin still cycles the same lanes producing the same data.

<figure class="zephon-post zephon-embed">
  <iframe src="../_static/embeds/lane-multiplexing.html" height="414" loading="lazy"
          title="Multiplex lanes, accumulate gradients"></iframe>
  <figcaption>
    <strong>The same global batch on two GPUs and on four.</strong> A global batch of 16
    samples drawn from four fixed lanes, at a per-device microbatch of 4. On two groups
    each one accumulates gradients over a microbatch from each of its two lanes; on four
    groups each takes one microbatch from its single lane. Either way the step contains
    the next four samples from every lane.
  </figcaption>
</figure>

**Mapping strategies.** When a GPU owns multiple lanes, the engine needs to decide which lanes it gets. Two strategies are available via `Pipeline.options(mapping_strategy=...)`. Both
are deterministic and produce identical global batches, so all they decide is which group
does which work.

`"contiguous"` is the default and gives each group a contiguous block of lane IDs, so with
`canonical_replicas=8` and `dp_degree=3` the groups get `[0,1,2]`, `[3,4,5]`, and `[6,7]`.
Adjacent lanes tend to read adjacent data, which makes this the choice that maximises
shard and block-cache locality.

`"interleaved"` deals lanes round-robin, so lane *i* goes to group *i % num_groups* and
the same numbers give the groups `[0,3,6]`, `[1,4,7]`, and `[2,5]`. This spreads
heterogeneous shards more evenly and tends to reduce tail variance between groups.

Note the uneven split in both examples, since eight lanes across three groups has to leave
somebody short. That is the reason for the divisibility advice in
{ref}`planning for elasticity <planning-for-elasticity>`: a lane count that is not
divisible by the group count means some groups own more lanes than others, and therefore
do more work per step.
