# Coming from Mosaic Streaming

[Mosaic Streaming](https://docs.mosaicml.com/projects/streaming/en/stable/) sits at the
heart of many large-scale training setups, and it has proven the value of several ideas
that Zephon builds on directly: shard-based streaming from object storage through a local
cache, elastic determinism, and reproducible shuffling without a full global permutation.
If you know Streaming well, most of Zephon will feel familiar, but there are a few places
where the two loaders diverge in ways that matter for how you reason about your data. This
page walks through those differences first, and then covers what you need to change to
move a training job from one to the other.

**Static mapping versus stream processing.** Streaming was designed for a world in which
each sample on disk maps one-to-one to a training sample, like a pre-tokenized text
sequence, an image, or a pre-processed record. You write your data, shard it, upload it,
and Streaming delivers it to your training loop, so the dataset effectively *is* the
curriculum. This works beautifully when every sample is ready to train on as it is, but
many modern workloads break the one-to-one assumption in one way or another: sequence
packing merges several short samples into one training sequence, tokenization with
splitting turns one long document into several training sequences, and multimodal
preprocessing may decode, resize, tile, or composite images, all of which benefit from
different parallelism strategies than text tokenization does.

Zephon takes a stream-processing approach to data loading instead. Data flows through an
operator graph in which each step can independently transform, filter, split, or merge
samples, and the question of *what* to train on (answered by the `WorkSource`) is
decoupled from the question of *how* to process it (answered by the operators of the
`Pipeline`). This separation makes many-to-many processing patterns natural, while
preserving the ideas that Streaming got right: high-throughput reads from cloud storage,
deterministic shuffling, and elastic resumption across different hardware. Zephon also
reads the MDS shard format natively, so you can keep using your existing Streaming
datasets without converting them.

**Every worker does everything.** The deepest architectural difference between Streaming
and Zephon is how they parallelize the data pipeline, and it is worth understanding
because it affects terminology, debugging intuitions, and how you reason about the order
in which data arrives. Streaming builds on PyTorch's `DataLoader` with an
`IterableDataset`, so when you set `num_workers=N`, PyTorch spawns `N` worker processes
per rank. Each of these workers independently runs the *entire* data pipeline for the
samples that are assigned to it: it fetches the shard from the cache (or downloads it),
decodes the sample, applies whatever transforms are defined in `__getitem__`, and returns
the result. The `DataLoader` then collects the finished samples from the workers in
round-robin order and collates them into batches:

```
┌─── Worker 1 ─────────────────────┐
│ Fetch ──> Decode ──> Transform   │──┐
└──────────────────────────────────┘  │
┌─── Worker 2 ─────────────────────┐  │    ┌──────────┐
│ Fetch ──> Decode ──> Transform   │──┼──> │  Output  │──> Training
└──────────────────────────────────┘  │    │  queue   │    loop
              ...                     │    └──────────┘
┌─── Worker N ─────────────────────┐  │
│ Fetch ──> Decode ──> Transform   │──┘
└──────────────────────────────────┘
```

This design is simple and battle-tested. Inter-process communication only happens at the
output boundary, where finished samples cross from the worker processes to the main
process, and there are essentially two knobs to tune: the number of workers and the
prefetch factor. But the model also comes with a few tradeoffs:

- **One parallelism setting for every step.** If fetching is I/O-bound and benefits from
  8-way concurrency while tokenization is CPU-bound and would benefit from 16-way
  parallelism, you cannot tune the two independently, since `num_workers` scales
  everything together.
- **Processes only.** Each worker is a separate operating system process, which is the
  only way around Python's global interpreter lock in this model. For workloads in which
  much of the pipeline is I/O-bound, threads would be far more memory-efficient, but the
  `DataLoader` does not offer that choice.
- **Data order can depend on execution settings.** In many loaders, changing a parameter
  that should only affect performance also changes the order in which the model sees its
  data; [Megatron Energon](https://nvidia.github.io/Megatron-Energon/advanced/repro_scaling.html),
  for example, documents that the number of data loading workers affects the data order.
  Streaming works around this with a clever partition algorithm, but it takes careful
  engineering to keep `num_workers` a pure performance knob, and that engineering only
  covers the samples that Streaming reads itself, not any splitting, filtering, or packing
  that you add around it.

**Zephon's operator graph.** Zephon replaces the "clone the whole pipeline into every
worker" model with an operator graph that is organized into *stages*. Each stage is
assigned a *runner*, which is the execution backend for that stage (threads, processes, or
inline), and each operator within a stage has its own *parallelism* setting:

```
WorkSource
    │
    v  pointers
┌─── Stage 0 (thread runner) ───────────┐
│                                       │
│   fetch@p4  ──>  tokenize@p4          │
│                                       │
└──────────────────┬────────────────────┘
                   │  samples
                   v
┌─── Stage 1 (inline runner) ───────────┐
│                                       │
│   ensure_mixture@p1  ──>  batch@p1    │
│                                       │
└──────────────────┬────────────────────┘
                   │  batches
                   v
             Training loop
```

In this example, fetching and tokenization both run on four threads, but the two settings
are independent of each other, so you could just as well give fetching two threads and
tokenization eight if tokenization turned out to be your bottleneck. The batching stage
runs inline because batching is lightweight enough that handing it off to another thread
would only add overhead. The stages themselves run concurrently, so while the batching
stage assembles the current batch, the first stage is already loading and tokenizing the
next samples, and the bounded queues between the stages provide backpressure: if a
downstream stage falls behind, the upstream stages block automatically instead of growing
their memory usage without limit.

Crucially, none of this affects the order of the samples. Each operator puts its results
back into their original order before passing them on, so whether fetching runs on one
thread or eight, the output is exactly the same. This means that the parallelism settings
in Zephon are always pure performance knobs, and never change what the model sees.
[How pipelines run](../pipelines/how_pipelines_run.md) explains this execution model in
more detail.

**A note on terminology.** Because of this architectural difference, several familiar
terms mean different things in the two loaders:

| Streaming | Zephon | Notes |
| --- | --- | --- |
| DataLoader worker | — | Zephon has per-operator parallelism, not whole-pipeline workers. |
| `num_workers` | `parallelism` | Set per operator, e.g., `.tokenize(field="text", parallelism=8)`. |
| Canonical node | Lane | Both are logical partitions of the data for elastic determinism. |
| `StreamingDataset` | `WorkSource` + `Pipeline` | Zephon separates what to read from how to process it. |
| `__getitem__` | Operators | Transforms are explicit, composable steps in the pipeline. |

When Streaming talks about "4 workers per rank", it means four complete copies of the
pipeline running side by side. When the output of `Pipeline.explain()` shows `fetch@p4`,
on the other hand, it means that the fetch operator specifically has four concurrent
executors inside its stage, while other operators in the same or in other stages may have
completely different parallelism settings. Zephon does not use the term "worker" in its
user-facing API at all.

(streaming-shuffling)=

**Shuffling with three independent settings.** Both loaders shuffle deterministically,
but they differ in how the controls are organized and in how tightly shuffling is coupled
to the way samples are distributed across the hardware. In Zephon, the
`StaticMixtureWorkSource` controls shuffling through three settings that are applied in
order when the sequence of samples is constructed. First, `shuffle_shards` (on by default)
permutes the order in which the shards are visited. Second, `shuffle_within_shard` (off
by default) permutes the samples inside each shard, at the cost of sequential I/O
locality. Third, `shuffle_block_size` (off by default) splits the sequence of samples of
each dataset into non-overlapping windows and shuffles each window independently, which
gives you controlled cross-shard mixing within a bounded window. You can set it to a
number of samples, to `"auto"` for a window of eight times the largest shard in the
mixture, or to `"global"` to shuffle the whole dataset, which requires holding all of its
sample identifiers in memory.
[Shuffling and repeating samples](../worksources/shuffling_and_repeating_samples.md)
covers these settings in more detail.

These are purely `WorkSource`-level controls that determine the order in which samples are
read. The lanes that distribute the data across GPUs operate independently of them and do
not introduce any additional interleaving of their own, unlike Streaming's canonical-node
partition (more on that below). In addition to these settings, `Pipeline.shuffle()`
provides a deterministic buffer shuffle that you can place anywhere in the operator chain,
which matters after operators that change sample boundaries; for example, after
tokenization has split long documents into several sequences, a buffer shuffle avoids
training on clusters of sequences that all come from the same document.

**Shuffling with pre-composed strategies.** Streaming exposes shuffling through a set of
named algorithms (`py1s`, `py1b`, `py1br`, `py1e`, and `py2s`), each of which is a
pre-composed strategy that decides two things at once: how to shuffle the samples (within
shards, across blocks, or both), and how to group the samples for the canonical nodes,
which determines download locality and cache pressure. This coupling is the key difference
from Zephon, since the shuffle algorithm also influences which shards each node needs to
download. For example, `py1s` keeps samples within shard boundaries while shuffling, which
minimizes the number of shards that each node needs to cache at once, whereas `py1br`
shuffles in blocks that can cross shard boundaries, which improves randomness but
increases cache pressure, because a single node needs shards from more distant parts of
the dataset. As a rough guide, `shuffle_within_shard=True` is the closest equivalent of
`py1s`, and adding `shuffle_block_size="auto"` plays the role of `py1b`, `py1br`, and
`py1e`.

**Hidden cross-shard interleaving.** To show why this coupling matters in practice, here
is a real debugging story that ties together the worker-based execution model and the
shuffle algorithms. We had configured Streaming with `py1s` shuffling, which shuffles
samples *within* each shard but not *across* shard boundaries, and our shards were small,
so we expected each worker to read its shards mostly sequentially, moving on to the next
shard only after finishing the current one. Instead, we found that workers were reading
from many shards at once from early on in training, in an access pattern that looked
exactly like a cross-shard shuffle, even though we had explicitly not asked for one.

The reason is that Streaming's coupling goes deeper than shuffling. The partition
algorithm that assigns samples to the hardware hierarchy introduces its own structural
interleaving, entirely independently of the shuffle algorithm, and it produces a
five-dimensional assignment array:

```
(physical_nodes, ranks_per_node, workers_per_rank, batches_per_worker, batch_size)
```

Before any samples are assigned to workers, the canonical-node mapping already interleaves
samples from different parts of the dataset across the physical nodes. This interleaving
is structural, since it is how Streaming achieves elastic determinism, but it has the same
practical effect as a shuffle. On top of that, the worker assignment deals out batch-sized
chunks to the workers in round-robin order, and the combined effect is that the samples
that each worker reads jump across distant shards:

```
Worker 0: chunk from shard 0, then shard 4, then shard 1, ...
Worker 1: chunk from shard 4, then shard 1, then shard 5, ...
Worker 2: chunk from shard 1, then shard 5, then shard 2, ...
```

As a result, the sequence of samples that each rank's model actually trained on jumped
across distant shards, even though `py1s` promised no cross-shard shuffling. Note that
this was not just an I/O artifact of workers downloading many shards in the background;
the order of the samples itself was cross-shard. `py1s` only controls the explicit shuffle
step, which permutes samples within each shard, but by the time that step runs, the
canonical-node interleaving has already scattered samples from different shards into each
rank's stream. The configuration said "no cross-shard shuffling", but the model was trained
on a cross-shard interleaved stream, and this only surfaced when we tried to reproduce the
run and found that the actual data order did not match what the configuration implied.
Tracking down the discrepancy required a deep dive into Streaming's partition internals.

```{note}
The fact that the partition deals chunks to workers in round-robin order, and that the
`DataLoader` collects them from the workers in round-robin order as well, is also why
changing `num_workers` does not break elastic determinism in Streaming: the global
sequence stays the same, and only its distribution across the workers changes. This
relies on PyTorch's round-robin dispatch order for `IterableDataset` workers, which is
stable in practice, but is technically an implementation detail rather than a documented
guarantee.
```

The takeaway is that in Streaming, the shuffle algorithm controls only one layer of the
sample order, while the partition algorithm and the worker-based execution model introduce
their own structural interleaving, so the shuffle configuration alone does not describe
what the model actually trains on. In Zephon, these concerns are separated: the
`WorkSource` controls the order of the samples through the shuffle settings described
above, the lanes distribute those samples without introducing any additional
interleaving, and the operators process samples in the order in which they receive them,
regardless of how many threads or processes execute them. If you disable cross-shard
shuffling in the `WorkSource`, the samples really do arrive in shard order. Keep this in
mind when you compare a migrated run against its Streaming baseline, since matching the
data order of a Streaming run that used `py1s` may require *more* shuffling in Zephon than
the Streaming configuration suggests, not less.

**Moving a job over.** With these differences in mind, the practical side of migrating a
Streaming job is fairly mechanical. Start by installing Zephon with the `mds` extra (which
requires Python 3.11 or newer), plus the `cloud` extra if your shards live in object
storage:

```bash
pip install "zephon[mds,cloud]"
```

Zephon reads MDS shards with the reader from the `streaming` library itself, so each
`Stream` simply becomes a `Dataset` that points at the same directory, as shown on the
[MDS page](../datasets/binary_shard_formats/mosaic.md). The rest of the `StreamingDataset`
arguments move to the object that owns them. The `proportion` of each stream becomes a
weight in the `MixtureSpec` of a `StaticMixtureWorkSource`, along with the seed and the
shuffle settings, although note that Zephon repeats datasets that run out early instead of
fixing a number of samples per stream, as described in
[Mixing datasets](../worksources/mixing_datasets.md). The body of `__getitem__` becomes a
chain of operators like `tokenize`, `map_transform`, and `pack_flat`, as described in
[Preparing samples](../pipelines/preparing_samples.md), and the cache settings (`local`,
`cache_limit`, `download_retry`, and so on) move to `CacheOptions`, as described in
[Storage backends and shard cache](../datasets/storage_backends_and_shard_cache.md).

**Canonical nodes become lanes.** `num_canonical_nodes` and `canonical_replicas` solve the
same problem in the same way, so the value that you used for one is a reasonable first
choice for the other. Unlike Streaming, though, Zephon does not read `RANK` and
`WORLD_SIZE` from the environment, so you need to pass the topology to
`Pipeline.options()` yourself:

```{literalinclude} ../../../examples/guide/migrating/streaming_distributed.py
:language: python
:caption: examples/guide/migrating/streaming_distributed.py
```

:::{warning}
If you leave out `world_size` and `global_rank` (or `dp_degree` and `dp_group_id`), every
`Pipeline` assumes that it is the only one, and **every rank trains on the same data**.
Nothing fails loudly when this happens; your loss curve will just look like you trained on
a fraction of your data.
:::

There is also no per-device `batch_size` argument like the one that Streaming needs for its
partition. Instead, every optimizer step needs to consume a multiple of
`canonical_replicas` microbatches across all of the data-parallel groups, so that every
lane contributes equally to every step;
{ref}`Planning for elasticity <planning-for-elasticity>` explains this rule and how to
choose a good value for `canonical_replicas`.

**Checkpointing.** A `StreamingDataLoader` checkpoints the number of samples that have been
seen so far, and resumes by skipping ahead by that number, which works as long as there is
exactly one sample per training example. Zephon's `Pipeline.checkpoint()` instead records
which work chunks have been fully delivered to the training loop, and replays the most
recent ones on restore to rebuild the state of operators like shuffle buffers and packing
bins, which is what allows it to resume a pipeline that packs across document boundaries,
even on a different topology. The one new rule to keep in mind is that **every rank must
call `checkpoint()` at the same step**, because the ranks exchange their progress while the
checkpoint is being assembled.
[Checkpointing and resuming](../pipelines/checkpointing_and_resuming.md) covers the
details. Streaming checkpoints cannot be converted into Zephon checkpoints, so you should
make the switch either at the start of a run or at a point where you are willing to restart
the curriculum.

**A migration checklist.** To summarize, migrating a Streaming job to Zephon usually comes
down to the following steps:

1. Install `zephon[mds]`, point one `Dataset` at each stream's directory, and check that
   `dataset.total()` matches the sample count that you expect.
2. Move the proportions, the seed, and the shuffle settings to a `StaticMixtureWorkSource`,
   and decide explicitly how much shuffling you want instead of relying on the defaults.
3. Rewrite `__getitem__` as `Pipeline` operators, and move the cache settings to
   `CacheOptions`.
4. Pass the topology and `canonical_replicas` to `Pipeline.options()`, and check that every
   optimizer step consumes a multiple of `canonical_replicas` microbatches.
5. Call `checkpoint()` on every rank at the same step, and save the result together with
   the model.
6. Compare a short run against your Streaming baseline, keeping in mind that the order of
   the samples is not expected to match exactly.
