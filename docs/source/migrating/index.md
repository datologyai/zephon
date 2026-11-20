# Migrating to Zephon

If you are used to training with
[Mosaic Streaming](https://docs.mosaicml.com/projects/streaming/en/stable/) or
[Grain](https://google-grain.readthedocs.io/) and want to move over to Zephon, the pages in
this section are for you. Each one translates the concepts and arguments of one of these
loaders into their Zephon equivalents, and points out the places where a straight
one-to-one translation would quietly change the data that your model sees.

We have guides for migrating from:

```{toctree}
:maxdepth: 1

from_mosaic_streaming
from_grain
```

But in general, three ideas account for most of the differences:

1. **The curriculum is separate from the processing.** Zephon divides the job of a data
   loader between three objects: a `Dataset` describes where the shards are, a `WorkSource`
   decides which samples to read, how to mix and shuffle them, and when to stop, and a
   `Pipeline` decides what happens to each sample after it is fetched. Arguments that used
   to sit together on a single dataset or loader constructor therefore end up on whichever
   of these three objects owns them. [Basic concepts](../basic_concepts.md) introduces all
   three.
2. **Parallelism is set per operator, not per worker.** Zephon does not copy your whole
   pipeline into each worker process; instead, each operator gets its own `parallelism`
   setting, and the engine restores the original order after every parallel step. This
   means that changing the parallelism only makes a pipeline faster or slower, and never
   changes which records arrive or in what order.
   [How pipelines run](../pipelines/how_pipelines_run.md) explains how this works.
3. **Elasticity covers the whole pipeline.** Zephon splits the curriculum into a fixed number
   of *lanes*, set by `canonical_replicas`, and checkpoints progress through the curriculum
   instead of a sample index. This is how a run can resume on a different number of GPUs
   and still produce the same global batches, even when the pipeline filters, splits,
   shuffles, or packs records along the way.
   [Distributed training](../pipelines/distributed_training.md) covers the settings involved.

**Keeping your data where it is.** You should not need to rewrite your dataset in order
to migrate, since Zephon reads
[Mosaic Data Shards (MDS)](../datasets/binary_shard_formats/mosaic.md) and
[ArrayRecord](../datasets/binary_shard_formats/array_record.md), the native formats of
Streaming and Grain, in place, alongside [JSONL](../datasets/jsonl.md),
[Parquet](../datasets/parquet.md), [litData](../datasets/binary_shard_formats/litdata.md),
[Vortex](../datasets/binary_shard_formats/vortex.md), and
[Lance](../datasets/binary_shard_formats/lance.md).
[Working with datasets](../datasets/index.md) lists the formats and what each needs.
