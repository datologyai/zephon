# Distributed Training

Modern large-model training typically uses multiple forms of parallelism at
once — data parallelism (DP), tensor parallelism (TP), pipeline parallelism
(PP), context parallelism (CP), and combinations thereof.  This page explains
how Zephon fits into that picture and where its approach differs from other data
loaders.

## The data loader's job in nD parallelism

Not every parallelism dimension cares about data.  In a job with DP × TP × PP,
the data loader's core responsibility is:

1. **Partition** the sample stream across the **data-parallel** dimension so
   that each DP group sees a unique, non-overlapping slice.
2. **Replicate** the same samples to every rank inside a DP group (the TP and
   PP ranks) so that they all operate on the same input.
3. Stay out of the way of model-parallel communication — the data loader should
   not introduce synchronisation barriers across TP/PP ranks.

Zephon achieves (1) through its
{py:class}`~zephon.work.WorkSource` and lane system,
(2) by having every rank in a DP group run an identical, deterministic pipeline,
and (3) by being fully rank-local — no inter-rank communication happens during
data loading.

## Telling Zephon about your topology

Zephon needs four pieces of information from your training framework:

| Parameter | Meaning |
|-----------|---------|
| `world_size` | Total number of ranks in the job |
| `global_rank` | This rank's unique ID (0 … world_size − 1) |
| `dp_degree` | Number of independent data-parallel groups |
| `dp_group_id` | Which group this rank belongs to (0 … dp_degree − 1) |

`world_size` and `global_rank` identify each rank globally — they are needed
for checkpoint aggregation (rank 0 acts as the leader) and for the Engine to
reason about the full topology.  `dp_degree` and `dp_group_id` control the
actual data partitioning: which lanes a rank owns and therefore which samples
it reads.

```python
from torch.distributed.device_mesh import init_device_mesh

# Example: 8 GPUs with TP=2
mesh = init_device_mesh("cuda", (4, 2), mesh_dim_names=("dp", "tp"))
dp_mesh = mesh["dp"]

pipeline = pipeline.options(
    world_size=8,
    global_rank=dist.get_rank(),
    dp_degree=dp_mesh.size(),         # 4
    dp_group_id=dp_mesh.get_local_rank(),  # 0–3
)
```

Notice that `dp_degree` and `dp_group_id` come from the **data-parallel
sub-mesh**, not from the global rank.  The training framework (e.g.
torchtitan, Composer) typically derives these for you from a
`DeviceMesh`.

### What about TP / PP / CP ranks?

Zephon is not aware of model-parallel dimensions.  All ranks that share the
same `dp_group_id` independently run the same deterministic pipeline and
therefore produce the same batches in the same order.  Because the work
schedule is deterministic for a given seed and lane assignment, no
communication between these ranks is needed — they arrive at the same data by
construction.

This means that in a DP=4, TP=2 setup (8 GPUs total), there are four
independent data streams, each consumed by two GPUs.  The two TP ranks within
a group each run their own Zephon pipeline, both producing identical batches:

```
GPU 0 ─┐ dp_group_id=0 ─── same data stream
GPU 1 ─┘ (TP peers)

GPU 2 ─┐ dp_group_id=1 ─── same data stream
GPU 3 ─┘ (TP peers)

GPU 4 ─┐ dp_group_id=2 ─── same data stream
GPU 5 ─┘ (TP peers)

GPU 6 ─┐ dp_group_id=3 ─── same data stream
GPU 7 ─┘ (TP peers)
```

This is a deliberate trade-off: each TP peer does redundant I/O and
preprocessing, but the design stays simple and fully decentralised.  In
practice, data loading is rarely the bottleneck in model-parallel training
because forward/backward computation and all-reduce dominate wall-clock time.

### FSDP, DDP, and HSDP

From the data loader's perspective, **all data-parallel dimensions need
different data** — regardless of whether model weights are replicated (DDP) or
sharded (FSDP).  The distinction between DDP and FSDP is about how parameters
and gradients are managed, not about what data each rank reads.

Frameworks like torchtitan expose this as two separate mesh dimensions:

- `dp_replicate` — DDP-style full-model replicas
- `dp_shard` — FSDP-style parameter sharding

Both dimensions are data-parallel: every rank across both dimensions needs a
**unique** slice of the global batch.  For data loading, they are flattened
into a single DP mesh:

```python
# torchtitan flattens replicate × shard into one "dp" dimension
dp_mesh = world_mesh["dp"]   # size = dp_replicate × dp_shard
dp_degree = dp_mesh.size()
dp_rank = dp_mesh.get_local_rank()
```

Zephon sees only the flattened result.  A job with `dp_replicate=2,
dp_shard=4` produces `dp_degree=8` — eight independent data streams, one per
lane.  Zephon does not need to know whether a particular rank is a DDP replica
or an FSDP shard; the only thing that matters is its `dp_group_id`.

```{note}
Hybrid Sharded Data Parallelism (HSDP) is the case where both dimensions are
greater than 1.  For Zephon, HSDP is no different from any other DP
configuration — `dp_degree` is still the product of both dimensions.
```

### Comparison with Mosaic Streaming's approach

MosaicML Streaming handles data replication through an explicit `replication`
parameter on the dataset: you set `replication=N` to tell Streaming that every
N consecutive ranks should receive the same samples.  The data loader
internally groups ranks into replica sets and partitions data across groups.

Zephon takes a different approach: there is no replication parameter.  Instead,
the training framework computes `dp_degree` and `dp_group_id` from its device
mesh and passes them to Zephon.  Ranks that share a `dp_group_id` (e.g. TP
peers) independently produce the same data by construction — determinism
guarantees it without any explicit coordination.  This is more flexible than
consecutive-rank grouping, since the mapping from ranks to DP groups can follow
any topology the mesh defines.

### Multiple pipelines on the same node

Because every rank runs its own Zephon pipeline, a node with 8 GPUs will have
8 independent pipeline instances, each with its own fetch threads, operator
workers, and buffers.  This can add up quickly in CPU and memory usage —
especially when operators like tokenization or custom transforms use multiple
worker threads per pipeline.

Keep this in mind when configuring parallelism and prefetch depths.  If you
set `fetch_parallelism=8` and run 8 pipelines on one node, that is 64 fetch
threads competing for CPU cores and network bandwidth.  In practice, you will
want to scale per-pipeline parallelism down as the number of pipelines per
node goes up.  Use {py:meth}`~zephon.api.Pipeline.explain` to inspect the
actual thread/process counts before launching at scale.

## How data distribution works

The {py:class}`~zephon.work.WorkSource` is responsible for partitioning the
sample stream (see [WorkSources](worksources.md) for more detail).  The
mechanism works in two steps:

### 1. Lanes

The Engine assigns each `dp_group_id` one or more **lanes** — logical,
deterministic sub-streams of the global sample order.  In the common case
(`canonical_replicas == dp_degree`), there is exactly one lane per DP group.
When `canonical_replicas > dp_degree`, a single DP group may own multiple
lanes; this is the basis of
[elastic determinism](determinism.md).

(compute-everywhere-discard-locally)=
### 2. Compute-everywhere, discard-locally

The current
{py:class}`~zephon.work.StaticMixtureWorkSource`
uses a *compute-everywhere-then-discard* strategy: every rank enumerates the
**global** chunk schedule deterministically, then keeps only the chunks that
belong to its lane(s) and skips the rest.

```python
# Simplified from StaticMixtureWorkSource.next_chunk()
while True:
    chunk = self._next_chunk()       # deterministic global schedule
    g = self._global_chunk_index
    chunk_lane = g % canonical_replicas
    if chunk_lane == self._lane:
        return chunk                 # keep — this chunk is ours
    # discard — belongs to another lane
```

This means **no rank needs to talk to any other rank** to figure out what data
to read.  Every rank can derive the correct answer locally, because the
schedule is a pure function of the seed and the lane assignment.

```{note}
The compute-everywhere approach is one possible strategy.  Because the
{py:class}`~zephon.work.WorkSource` is an abstraction, alternative
implementations could use a central coordinator that is globally aware and
distributes work explicitly, or employ hash-based assignment, or any other
scheme — as long as the resulting per-lane streams are deterministic and
non-overlapping.
```

## How this differs from other data loaders

### The sequence is not pre-computed

Most distributed data loaders (e.g. PyTorch's `DistributedSampler`,
MosaicML Streaming) **materialise the full sample order up front** — they
build a permutation of all sample indices, shard it across ranks, and then
iterate.  This works well when the dataset size and the number of ranks are
both known before training starts.

Zephon takes a different approach: the
{py:class}`~zephon.work.WorkSource` is a **streaming generator**.  It produces
work chunks on demand, one at a time, without ever materialising the full
sequence.  This has several consequences:

- **Infinite and dynamically growing datasets** work naturally — there is no
  requirement that the total number of samples be known at construction time.
- **Changing the number of ranks** mid-training does not require recomputing a
  global permutation.  The lane abstraction handles re-mapping (see
  [Elastic Determinism](determinism.md)).
- **Memory overhead stays constant** regardless of dataset size — no rank ever
  holds a full index in memory.

The trade-off is that certain operations that assume a known length (like
"skip to 73 % of the epoch") are not directly expressible.  Instead, Zephon
tracks progress through opaque cursors inside the WorkSource, which can be
checkpointed and restored.

### No rank-to-rank communication for data loading

In some data loaders, ranks coordinate to avoid redundant downloads or to
implement collective shuffling.  Zephon is fully rank-local: each rank
independently computes which data it needs and fetches it directly.
Coordination only happens at checkpoint time (see
[Checkpointing](checkpointing.md)).

## Putting it together: a 3D training example

Here is a sketch of how Zephon is wired up in a DP=2, TP=2 job (4 GPUs):

```python
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from zephon.api import Pipeline
from zephon.io import Dataset
from zephon.work import MixtureSpec, StaticMixtureWorkSource

# 1. Set up the device mesh
mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=("dp", "tp"))
dp_mesh = mesh["dp"]

# 2. Build the Zephon pipeline (same code on every rank)
ds = Dataset.from_path("train", "/data/train/")
ws = StaticMixtureWorkSource(
    [ds],
    mixture=MixtureSpec({"train": 1.0}),
    chunk_size=1024,
    seed=42,
)

pipeline = (
    Pipeline(ws)
    .tokenize(tokenizer_id="meta-llama/Llama-3-8B", max_length=4096)
    .batch(microbatch_size=4)
    .options(
        world_size=dist.get_world_size(),     # 4
        global_rank=dist.get_rank(),          # 0–3
        dp_degree=dp_mesh.size(),             # 2
        dp_group_id=dp_mesh.get_local_rank(), # 0 or 1
        aggregate_dir="/shared/zephon_agg/",
    )
)

# 3. Train — TP peers (e.g. rank 0 & 1) get identical batches
for batch in pipeline:
    training_step(batch.to_training())
```

The key point: Zephon only needs to know the **data-parallel** topology.  It
does not need to know about TP degree, PP stages, or any other model-parallel
dimension.  The training framework handles those concerns at the model level.
