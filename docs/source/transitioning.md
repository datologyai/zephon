# Transitioning from Streaming

[Mosaic Streaming](https://docs.mosaicml.com/projects/streaming/en/stable/index.html) sits at the heart of many large-scale training setups and
has proven the value of several important ideas: shard-based streaming with
caching, elastic determinism, and reproducible shuffling.  This page is for
users who know Streaming and want to understand how Zephon builds on and
diverges from that foundation.

---

## Philosophy: Static Mapping vs. Stream Processing

Streaming was designed for a world where each sample on disk maps
one-to-one to a training sample: a pre-tokenized text sequence, an image,
a pre-processed record.  You write your data, shard it, upload it, and
Streaming delivers it to your training loop.  The dataset *is* the
curriculum.

This works beautifully when every sample is ready to train on as-is.  But multi-modal workloads frequently break the one-to-one assumption (see also
[Why an Iterable Pipeline?](basic_concepts.md#why-an-iterable-pipeline)):

- **Sequence packing** merges multiple short samples into one training
  sequence (many-to-one).
- **Tokenization with splitting** turns one long document into multiple
  training sequences (one-to-many).
- **Multimodal preprocessing** may decode images, resize, tile, or
  composite --- transformations that benefit from different parallelism
  strategies than text tokenization. Some operations might even get elastically scaled across standalone CPU preprocessing nodes.
- **Dynamic curricula** adjust dataset mixture weights during training
  based on model performance. The curriculum is no longer static.

Zephon takes a **stream-processing** approach to data loading.  Data flows
through an operator graph where each step can independently transform,
filter, split, or merge samples.  The "what to train on" question (the
{py:class}`~zephon.work.WorkSource`) is decoupled from the "how to
process it" question (the {py:class}`~zephon.Pipeline`'s operator
chain).  This separation makes it natural to support dynamic, N:M
processing patterns, while preserving the ideas that Streaming got right:
high throughput reads from cloud storage, deterministic shuffling, and elastic resumption
across different hardware.  Zephon also reads the MDS shard format
natively, so existing Streaming datasets can be used without conversion.

---

## Data Loading Workers vs. Operator-Based Parallelism

The deepest architectural difference between Streaming and Zephon is how
they parallelize the data pipeline.  Understanding this matters because it
affects terminology, debugging intuitions, and how you reason about data
ordering.

### The Traditional Model: Every Worker Does Everything

Streaming builds on PyTorch's `DataLoader` with `IterableDataset`.  When
you set `num_workers=N`, PyTorch spawns N worker processes per rank.  Each
worker independently runs the *entire* data pipeline for its assigned
samples: fetch the shard from the cache (or download it), decode the
sample, apply any transforms defined in `__getitem__`, and return the
result.  The DataLoader collects finished samples from workers in
round-robin order and collates them into batches.

```
┌──── Worker 1 ──────────────────────┐
│ Fetch ──> Decode ──> Transform     │──┐
└────────────────────────────────────┘  │
┌──── Worker 2 ──────────────────────┐  │    ┌───────────┐
│ Fetch ──> Decode ──> Transform     │──┼──> │  Output   │──> Training
└────────────────────────────────────┘  │    │  Queue    │    Loop
              ...                       │    └───────────┘
┌──── Worker N ──────────────────────┐  │
│ Fetch ──> Decode ──> Transform     │──┘
└────────────────────────────────────┘
```

This design is simple and battle-tested.  IPC only happens at the output
boundary (finished samples crossing from worker processes to the main
process), and there are essentially two tuning knobs: the number of
workers and the prefetch factor.

But the model has trade-offs:

- **One parallelism setting for all operations.**  If fetching is
  I/O-bound and benefits from 8-way concurrency but tokenization is
  CPU-bound and would benefit from 16-way parallelism, you cannot tune
  them independently --- `num_workers` scales everything together.
- **Limited to processes.**  Each worker is a separate OS process (due to
  Python's GIL before 3.13).  For workloads where much of the pipeline is
  I/O-bound, threads would be more memory-efficient, but the DataLoader
  model does not offer that choice.
- **Determinism can couple to execution settings.** For example [Megatron Energon](https://nvidia.github.io/Megatron-Energon/advanced/repro_scaling.html) explicitly mentions that the number of data loading workers impacts the data ordering, i.e., changing an execution parameter that should only impact performance also changes the input data ordering. Mosaic Streaming works around
  this with a clever partition algorithm,
  but it requires careful engineering to make `num_workers` a pure
  performance knob rather than something that changes what the model sees. 

### Zephon's Operator Graph

Zephon replaces the "clone the whole pipeline per worker" model with an
**operator graph** organized into **stages**.  Each stage is assigned a
**runner** --- an execution backend (threads, processes, or inline) ---
and each operator within a stage has its own **parallelism** level.

```
WorkSource
    │
    v  pointers
┌─── Stage 0 (Thread Runner) ────────────────┐
│                                             │
│  Fetch @p4  ──>  Tokenize @p4              │
│                                             │
└─────────────────┬───────────────────────────┘
                  │  samples
                  v
┌─── Stage 1 (Inline Runner) ────────────────┐
│                                             │
│  EnsureMixture @p1  ──>  Batch @p1         │
│                                             │
└─────────────────┬──────────────────────────┘
                  │  batches
                  v
            Training Loop
```

In this example, fetch runs with 4 threads and tokenization runs with 4
threads, but these are independently configurable.  You could set fetch to
2 and tokenize to 8 if tokenization were your bottleneck.  The batch stage
runs inline (single-threaded) because batching is lightweight and benefits
from avoiding concurrency overhead.

Stages run **concurrently**: while the batch stage assembles the current
batch, the fetch-tokenize stage is already loading the next samples.
Bounded queues between stages provide backpressure. If the downstream
stage is full, upstream producers block automatically, preventing
unbounded memory growth.

Crucially, the execution topology does not affect sample ordering.
Operators process deterministic micro-batches identified by sequence
numbers.  Whether fetch runs with 1 thread or 8, the output order is the
same.  This means parallelism settings in Zephon are always pure
performance knobs. They never change what the model sees.

### A Note on Terminology

Because of this architectural difference, several terms mean different
things across the two systems:

| Streaming concept | Zephon equivalent | Notes |
|---|---|---|
| DataLoader worker | *(no direct equivalent)* | Zephon has per-operator parallelism, not monolithic "workers" |
| `num_workers` | `parallelism` per operator | Each operator may be tuned independently, e.g. `.tokenize(field="text", parallelism=8)` |
| Canonical node | Lane | Both are logical data-parallel partitions for elastic determinism |
| `StreamingDataset` | WorkSource + Pipeline | Zephon separates data declaration from processing |
| `__getitem__` / transforms | Operators in the Pipeline | Transforms are explicit, composable steps in the operator graph |

When Streaming says "4 workers per rank," it means 4 complete pipeline
clones running in parallel.  When Zephon's
{py:meth}`~zephon.Pipeline.explain` output shows `fetch@p4`, it means
the fetch operator specifically has 4 concurrent executors inside its
stage. Other operators in the same or different stages may have
different parallelism.  Zephon does not use the term "worker" in its
user-facing API at all.

---

## Shuffle Algorithms

Both systems support deterministic shuffling.  The approaches differ in
how the controls are organized and how tightly shuffling is coupled to
sample distribution.

### Zephon: Three Orthogonal Knobs

Zephon's {py:class}`~zephon.work.StaticMixtureWorkSource` controls
shuffling through three independent settings, applied in order when the
sample sequence is constructed:

1. **Shard shuffling** (`shuffle_shards`, on by default) --- permutes the
   order in which shards are visited.
2. **Within-shard shuffling** (`shuffle_within_shard`, off by default) ---
   permutes sample offsets inside each shard, at the cost of sequential
   I/O locality.
3. **Block shuffling** (`shuffle_block_size`, off by default) --- after
   the per-dataset sample sequence is assembled, partitions it into
   non-overlapping windows of `shuffle_block_size` samples and shuffles
   each window independently.  This creates controlled cross-shard mixing
   within a bounded window. Accepts a positive integer, or the sentinels
   `"auto"` (= `8 × max(shard_size)` across the mix) and `"global"` (= that
   dataset's total sample count, i.e. one block per dataset; note that
   `"global"` buffers on the order of the dataset's total samples in memory
   while the block is materialized). Resolved per-dataset values are locked
   into the checkpoint, so resuming after adding shards keeps the original
   block size.

These are purely WorkSource-level controls that determine the pointer
ordering.  The Engine's lane system, which handles distribution across
GPUs, operates independently and does not introduce additional
interleaving (unlike Streaming's canonical-node partition ---
see the [example below](#example-hidden-cross-shard-interleaving)).

Additionally, Zephon supports **in-pipeline shuffling** via
{py:meth}`Pipeline.shuffle() <zephon.Pipeline.shuffle>`, a
deterministic buffer-based shuffle that can be placed anywhere in the
operator chain.  This matters after operators that change sample
boundaries: for instance, after tokenization splits long documents into
multiple sequences, the effective ordering may benefit from re-shuffling
to avoid clusters of sequences that all originate from the same document.

### Streaming: Pre-Composed Strategies

Streaming exposes shuffling through named algorithms (`py1s`, `py1br`,
`py2s`, `py1e`).  Each algorithm is a **pre-composed strategy** that
jointly decides two things:

1. **How to shuffle** --- within shards, across blocks, or both.
2. **How to group samples for canonical nodes** --- which determines
   download locality and cache pressure.

This coupling is the key difference from Zephon.  The shuffle algorithm
also influences which shards each node needs to download.  For example,
`py1s` keeps samples within shard boundaries during shuffling, which
minimizes the number of shards each node must cache at once.  `py1br`
shuffles in blocks that can cross shard boundaries, improving randomness
but increasing cache pressure because a single node needs shards from
more distant parts of the dataset.

---

## Example: Hidden Cross-Shard Interleaving

This section ties together the two topics above --- the worker-based
execution model and the shuffle algorithms --- with a real debugging story.
It illustrates how Streaming's shuffle configuration can promise one data
ordering while the actual training stream delivers another, and why this
mismatch cost significant debugging time when trying to reproduce a run.

**The setup.**  We configured Streaming with `py1s` shuffling --- the
algorithm that shuffles *within* each shard but not *across* shard
boundaries.  The shards were small.  We expected each worker to read
shards mostly sequentially, moving on to the next shard only after
finishing the current one.

**What happened.**  Workers were reading from many shards simultaneously
from early in training.  The access pattern looked like a cross-shard
shuffle, even though we had explicitly disabled it.

**Why.**  As described [above](#streaming-pre-composed-strategies),
Streaming's shuffle algorithms are coupled to canonical-node assignment.
But the coupling goes deeper than shuffling: the **partition algorithm**
that assigns samples to the hardware hierarchy introduces its own
structural interleaving, entirely independent of the shuffle algorithm.

Streaming's partition produces a 5D assignment array:

```
(physical_nodes, ranks_per_node, workers_per_rank, batches_per_worker, batch_size)
```

Before worker assignment even happens, the **canonical node mapping**
interleaves samples from different parts of the dataset across physical
nodes.  This interleaving is structural --- it is how Streaming achieves
elastic determinism, not a "shuffle" --- but it has the same practical
effect.

On top of that, the **worker assignment** deals batch-sized chunks to
workers in round-robin order.  The combined effect is that each worker's
sample stream jumps across distant shards:

```
Worker 0: chunk from Shard 0, then Shard 4, then Shard 1, ...
Worker 1: chunk from Shard 4, then Shard 1, then Shard 5, ...
Worker 2: chunk from Shard 1, then Shard 5, then Shard 2, ...
```

The result is that each rank's actual **training stream** --- the sequence
of samples the model sees --- jumps across distant shards, even though
`py1s` promised no cross-shard shuffling.  This is not just an I/O
artifact (workers pre-downloading many shards in the background); the
sample ordering itself is cross-shard.  `py1s` only controls the explicit
shuffle step (within-shard sample permutation), but the partition's
canonical-node interleaving has already scattered samples from different
shards into each rank's stream before the shuffle even runs.

The shuffle configuration said "no cross-shard shuffling", but the model
was trained on a cross-shard interleaved stream.  This only surfaced when
we tried to reproduce the run: the actual data ordering did not match what
the config implied, and tracking down the discrepancy required diving deep
into Streaming's partition internals.

```{note}
The partition's round-robin dealing to workers and the DataLoader's
round-robin collection from workers being inverses is also why changing
`num_workers` does not break elastic determinism in Streaming.  The global
sequence stays the same; only the distribution across workers changes.
This relies on PyTorch's round-robin dispatch order for `IterableDataset`
workers --- stable in practice, though technically an implementation
detail rather than a documented API guarantee.
```

**The takeaway.**  In Streaming, the shuffle algorithm (`py1s`, `py1br`,
etc.) controls one layer of sample ordering, but the partition algorithm
and the
[worker-based execution model](#the-traditional-model-every-worker-does-everything)
introduce their own structural interleaving.  The shuffle configuration
alone does not describe what the model actually trains on --- and this gap
between config and reality becomes a concrete bug the moment you try to
reproduce or reason about a training run.

In Zephon, these concerns are separated.  The WorkSource controls the
sample ordering via its [shuffling knobs](#zephon-three-orthogonal-knobs),
and the Engine distributes samples to lanes without introducing additional
interleaving.  Operators process samples in the order they receive them,
regardless of how many threads or processes execute them.  If you disable
cross-shard shuffling in the WorkSource, samples genuinely arrive in
shard-sequential order.
