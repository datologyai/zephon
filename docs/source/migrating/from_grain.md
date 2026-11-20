# Coming from Grain

[Grain](https://google-grain.readthedocs.io/) is the data loader of choice for many JAX
training stacks, and its design overlaps with Zephon's in several important ways: both
loaders are deterministic by construction, both read from random-access shard formats, and
both checkpoint the position of the input pipeline itself instead of relying on the
training loop to skip ahead. If you know Grain well, most of Zephon will feel familiar, but
the two loaders start from different ideas about what a dataset is, and those ideas lead to
different behavior once a pipeline filters, packs, or changes its number of workers. This
page walks through those differences first, and then covers what you need to change to
move a training job from one to the other.

**Index-based datasets versus a curriculum plan.** A Grain `MapDataset` is a lazy view over
record indices: `ds[i]` gives you record `i`, and every transformation you chain onto it,
like `shuffle`, `map`, `repeat`, or slicing, produces a new view that maps indices to
records in a different way. This makes Grain very flexible, since you can put most
transformations in nearly any order, and it is a natural fit for ArrayRecord, a format that
was designed for fast random access to individual records. A global shuffle, for example,
is simply a permutation of the indices of all records.

Zephon instead separates the job of a data loader into two halves. The `WorkSource`
decides *which* samples are read and in what order, so shuffling, mixing, and repetition
all happen there, as a plan that is fixed before any record is fetched. The `Pipeline`
then decides *what happens* to each sample after it is fetched, through a chain of
operators that decode, filter, tokenize, pack, and batch. Because Zephon is designed to
stream large datasets from object storage through a local cache, where opening an entire
shard just to read a single record is expensive, the plan is organized around shards
rather than around individual indices. The practical consequence is that the
transformations that you used to chain onto a `MapDataset` are split across two objects:
anything that decides the order of the records moves to the `WorkSource`, and anything that
changes the records themselves becomes an operator on the `Pipeline`.

**Where Grain's determinism stops.** Grain's determinism holds for a fixed configuration,
and its own documentation is quite candid about where that ends. The documentation for
`mp_prefetch` notes that if a pipeline contains filters or stateful transformations such
as packing, its output changes when the number of worker processes changes. The reason
lies in how the work is divided: each worker process runs the rest of the pipeline on its
own slice of the records, so a packer in one worker can only combine the records that
ended up in that worker's slice, and changing the number of workers changes which records
end up together. `grain.experimental.ElasticIterator` makes resumption elastic across
changes in sharding, but it explicitly disallows filters and other many-to-one
transformations, and it requires the pipeline to be unbatched and unsharded.

For many language-model pretraining pipelines, which filter, split documents, and pack
sequences as a matter of course, this means that the global batches depend on the number
of workers and hosts that the run happens to use. Zephon's elastic determinism covers
exactly these pipelines: you can filter, split, shuffle, pack, change the number of
workers, and resume on a different number of hosts, and every global batch stays the
same. [Why Zephon?](../why_zephon.md) explains why this matters for the results of your
experiments, and
[Elastic determinism](../internals/elastic_determinism.md) explains how it works.

**Threads, processes, and operators.** Grain parallelizes a pipeline in two places.
`ReadOptions.num_threads` sets the number of threads that read and transform records
before `to_iter_dataset`, and `mp_prefetch` starts worker processes that each run the rest
of the pipeline on their own slice of the data, much like the workers of a PyTorch
`DataLoader`. Zephon instead compiles the pipeline into stages that are connected by
bounded queues, where each operator gets its own `parallelism` setting and each stage runs
on its own runner (threads, processes, or inline). For example, fetching can run on four
threads while tokenization runs on eight, and batching can run inline because it is too
cheap to be worth handing off. Each operator puts its results back into their original
order before passing them on, so the parallelism settings in Zephon are always pure
performance knobs: whether tokenization runs on one thread or sixteen, the output is
exactly the same. For CPU-heavy Python code that is limited by the global interpreter
lock, you can switch to the `process` runner, and to keep the whole pipeline out of the
training process, which is what we recommend for production runs, you can set
`Pipeline.options(mtp_mode=True)`.
[How pipelines run](../pipelines/how_pipelines_run.md) explains this execution model in
more detail.

**A note on terminology.** Because of these differences, several familiar concepts have a
slightly different shape in Zephon:

| Grain | Zephon | Notes |
| --- | --- | --- |
| `MapDataset`, `IterDataset` | `WorkSource` + `Pipeline` | Zephon separates what to read from how to process it. |
| `num_threads`, `num_workers` | `parallelism` | Set per operator, e.g., `.tokenize(field="text", parallelism=8)`. |
| Shard of the data | Data-parallel group | Set with `dp_degree` and `dp_group_id`. |
| `ElasticIterator` | Lanes | A fixed number of logical streams, set with `canonical_replicas`. |
| Segment IDs | Document IDs | Both start at 1 in each packed row, with 0 for padding. |

(grain-shuffling)=

**Shuffling with three independent settings.** Since Grain shuffles a permutation of
record indices, its shuffle is global by default. Zephon instead offers three settings on
the `StaticMixtureWorkSource`, all of which are driven by its `seed` and applied in order
when the sequence of samples is constructed. First, `shuffle_shards` (on by default)
permutes the order in which the shards are visited. Second, `shuffle_within_shard` (off by
default) permutes the records inside each shard. Third, `shuffle_block_size` (off by
default) shuffles records within windows that cross shard boundaries, either of a given
number of samples, of eight times the largest shard with `"auto"`, or of the whole dataset
with `"global"`, which shuffles just like Grain does, at the cost of holding every record
identifier in memory.

A global shuffle is not as important as you might think, though. Mixing across eight
shards is close to a full shuffle in practice, as discussed in
[Shuffling and repeating samples](../worksources/shuffling_and_repeating_samples.md), so
`shuffle_within_shard=True` together with `shuffle_block_size="auto"` is a good default for
data that came from Grain, and `"global"` is only worth its memory cost if you need to
reproduce the statistics of a Grain run exactly. Note that random reads within a shard are
only cheap if each record can be decoded independently, so it is worth checking the
`group_size` that your ArrayRecord shards were written with; the
[ArrayRecord page](../datasets/binary_shard_formats/array_record.md) explains this
tradeoff. A `WindowShuffleIterDataset` later in the pipeline becomes `Pipeline.shuffle`,
which is a deterministic buffer shuffle that you can place anywhere in the operator chain.

Since there is no global record index in Zephon, a function that used `map_with_index` to
identify a record can read `record.meta.sample_id` in a custom operator instead, which
returns a `(dataset, shard, index within shard)` tuple that is stable across runs.

(grain-mixing)=

**What happens when a dataset runs out.** Grain's `mix` and Zephon's `MixtureSpec` take
the same relative weights, but they behave differently when one of the datasets runs out
of samples. Grain stops the mixture as soon as any of its inputs is exhausted, unless you
`repeat` the inputs first, whereas Zephon's default is to keep the requested mixture by
repeating any dataset that runs out early, until every dataset has completed at least one
pass. To get Grain's behavior, set `exhausted_policy="stop"`, and to repeat indefinitely,
as you would by calling `.repeat()` before mixing, set `exhausted_policy="repeat"`. A
fixed number of epochs becomes `stop_after_passes`, and Zephon reshuffles on every pass by
default, just like Grain does with `reseed_each_epoch=True`. Grain also recommends
shuffling each input before mixing to keep the proportions right, but since Zephon applies
each dataset's shuffle settings before it interleaves them, you don't need to do anything
special for this. [Mixing datasets](../worksources/mixing_datasets.md) covers the rest,
including how to mix by tokens instead of by samples.

(grain-random-transformations)=

**Randomness that survives a restore.** Every function that you hand to Zephon must be
deterministic, which means that the same input must always give the same output, including
the decision to drop a record, since Zephon re-runs these functions when it restores a
checkpoint and relies on getting the same result each time. Grain's `random_map` handles
randomness by giving each record its own random number generator, derived from the
pipeline seed and the position of the record. Zephon has no built-in equivalent yet, but
you can build one with a custom operator that derives the seed from the metadata of each
record:

```{literalinclude} ../../../examples/guide/migrating/grain_random_map.py
:language: python
:caption: examples/guide/migrating/grain_random_map.py
```

In this example, the lane, chunk, and offset of a record are fixed by the curriculum and
not by the topology, so a record gets the same draws when it is replayed after a restore or
read on a different number of hosts. Because the chunk changes on every pass, a record that
is repeated in a later epoch gets new draws; if you want the same draws on every pass
instead, seed the generator from `meta.sample_id` and `meta.lineage` alone. The operator
sets `calibration_replayable=False` because Zephon's token-estimation calibration replays
records with placeholder metadata, which would change the draws, and Zephon's operator
validation reports an error if you set it to `True`.

**Moving a job over.** With these differences in mind, the practical side of migrating a
Grain job is fairly mechanical. Zephon reads ArrayRecord files in place, so you only need
to install the `array_record` extra, plus the `cloud` extra if your shards live in object
storage:

```bash
pip install "zephon[array_record,cloud]"
```

There is one structural difference to be aware of: a Grain `ArrayRecordDataSource` takes a
list of file paths, while a Zephon `Dataset` takes a *directory* and treats every
`.array_record` file directly inside it as a shard. If your file lists were assembled from
several directories, you can either give each directory its own `Dataset` and mix them, or
move the files so that each dataset lives in a single directory. For large or remote
datasets, we also recommend building an index with
`python -m zephon.build_index array_record <path>`, so that Zephon does not have to read
the footer of every shard at startup, as described on the
[ArrayRecord page](../datasets/binary_shard_formats/array_record.md).

Zephon returns each ArrayRecord record as bytes, just like Grain, so you can keep using
the same decode function. `map` and `filter` both become `map_transform`, whose function
returns either the new payload or `None` to drop the record, so be careful to invert your
filters, since a Grain predicate returns `True` for the records to **keep**; a decode step
and a filter usually combine naturally into a single function. Grain's packing datasets
map onto the algorithms of `pack_flat`, with `ConcatThenSplitIterDataset` becoming
`algorithm="wrap"`, `FirstFitPackIterDataset` becoming `algorithm="first_fit"`, and
`BestFitPackIterDataset` becoming `algorithm="best_fit"`, although `"wrap"` drops the
tokens that cannot fill a complete sequence whenever Zephon flushes its operator state,
instead of carrying them over. Finally, note that `Pipeline.batch` drops an incomplete
final batch by default, while Grain's `batch` keeps it, and that
`SampleBatch.to_training()` returns NumPy arrays that can go straight into `jax.device_put`
when you pass it a NumPy `dtype`. [Packing](../pipelines/packing.md) and
[Building training batches](../pipelines/building_training_batches.md) cover these
operators in detail.

(grain-sharding)=

**Sharding across hosts.** In Grain, each JAX process reads its own slice of the data,
which you set with `ShardByJaxProcess`, `ShardOptions`, or slicing. In Zephon, each
process instead builds the same `Pipeline` and tells it which data-parallel group it
belongs to through `dp_degree` and `dp_group_id`, while `canonical_replicas` sets the
number of lanes that keep the run elastic. Let's look at a pipeline in which each process
loads its own part of the global batch:

```{literalinclude} ../../../examples/guide/migrating/grain_sharding.py
:language: python
:caption: examples/guide/migrating/grain_sharding.py
```

In this example, `dp_degree` and `dp_group_id` are simply `jax.process_count()` and
`jax.process_index()`, and the 16 lanes mean that each host should pull
`16 / process_count` microbatches (or a multiple of that) per optimizer step, so that every
lane contributes equally to every step;
{ref}`Planning for elasticity <planning-for-elasticity>` explains this rule in more
detail. If several processes need to receive identical data, for example because they hold
different shards of the same model replica, give them the same `dp_group_id`, as described
in [Distributed training](../pipelines/distributed_training.md).

:::{warning}
If you leave out `dp_degree` and `dp_group_id` (and `world_size` and `global_rank`),
every process assumes that it is the only one, and **every host trains on the same data**.
Nothing fails loudly when this happens; your loss curve will just look like you trained on
a fraction of your data.
:::

(grain-checkpointing)=

**Checkpointing.** In Grain, each process saves and restores its own iterator state. In
Zephon, a single checkpoint describes the whole job instead: every process calls
`Pipeline.checkpoint()` at the same training step, the process with `global_rank=0`
combines everyone's progress, and when the call returns, every process holds the combined
checkpoint, which it passes to `Pipeline.restore()` on resume, even if the number of
processes has changed in the meantime. The processes exchange their progress through a
shared location, and since there is usually no `torch.distributed` TCPStore in a JAX job,
you should set `aggregate_dir` to a bucket prefix or shared filesystem that every host can
reach. [Checkpointing and resuming](../pipelines/checkpointing_and_resuming.md) covers the
details. Grain iterator states cannot be converted into Zephon checkpoints, so you should
make the switch either at the start of a run or at a point where you are willing to restart
the curriculum.

**A migration checklist.** To summarize, migrating a Grain job to Zephon usually comes
down to the following steps:

1. Install `zephon[array_record]`, arrange the shards of each dataset in a single
   directory, and build an index for remote data.
2. Move the seed, shuffle, mixture, and repeat settings to a `StaticMixtureWorkSource`, and
   choose `exhausted_policy` deliberately, since the defaults differ from Grain's.
3. Rewrite `map` and `filter` as `map_transform`, inverting the filter predicates, and
   rewrite `random_map` as a seeded custom operator.
4. Replace Grain's packing dataset with `pack_flat`, check `drop_last` on `batch`, and pass
   a NumPy `dtype` to `to_training`.
5. Pass `dp_degree`, `dp_group_id`, `world_size`, `global_rank`, `canonical_replicas`,
   `aggregate_dir`, and `run_id` to `Pipeline.options()`.
6. Call `checkpoint()` on every process at the same step, and save the result together
   with the model.
