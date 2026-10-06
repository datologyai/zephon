# Shuffling and Repeating Samples

**Sample Order Is Part of the Curriculum.** While the mixture specification determines
how many samples each `Dataset` contributes to each training epoch and how samples from
different `Dataset`s are interleaved, the shuffle specification determines the order in
which the `StaticMixtureWorkSource` draws samples from each individual `Dataset` before
that interleaving occurs. Shuffling is critical because the samples that are adjacent to
one another within the shards of a `Dataset` often share common attributes (like source,
format, language, or quality) that likely do not reflect the overall representation of
those attributes in the overall data that we want to train on. Ideally, each local batch
of training data is made up of a mixture of samples that reflect the overall
distribution of the training data to ensure that the optimization process does not
over-index on one specific type of data, which can make the training process unstable or
lead to misleading loss curves. A random shuffle of the training data is one of the best
tools we have to help local batches better reflect the overall distribution of data from
the training set as a whole.

Of course, there is a tradeoff between the strength of the random shuffle we perform and
the throughput of our data loading pipeline: while a purely random global shuffle reduces
local correlation between samples caused by the original ordering during the training
epoch to a maximum, it comes at a cost for the I/O locality and efficiency of reading the
data, especially for data formats that are not designed for efficient random access to
individual records within a file or cloud setups. Zephon provides a number of settings on
the `StaticMixtureWorkSource` that allows you to control how samples are shuffled within
each training epoch so that you can maximize the performance of your data loading pipeline
while minimizing the impact of spurious correlations between samples on your training
loop.

**Choosing A Shuffle Strategy.** Our shuffling configuration is inspired by [Mosaic
Streaming](https://docs.mosaicml.com/projects/streaming/en/latest/dataset_configuration/shuffling.html).
By default, the `StaticMixtureWorkSource` will shuffle the order of the individual
shards within a `Dataset`. You can turn this default behavior off by setting
`shuffle_shards=False` on the constructor. Shuffling the shards within each `Dataset`
every time it is processed during an epoch is a mechanism for getting a minimal amount
of randomness in your training loop without any meaningful cost in I/O efficiency,
because the `WorkSource` will be able to draw the samples from each shard in the
`Dataset` sequentially.

:::{note}
TODO: add the shuffle strategies visualization.
:::

The next level of shuffling is `shuffle_within_shard`, which is disabled by default; it
randomizes the order in which samples from a single shard are read on each epoch. This is
a good option to enable when the adjacent samples within a shard have strongly correlated
properties that we would like to avoid training on, but it does impose an additional
compute cost when the format of our `Dataset` is not designed for efficient random access
of individual records (most notably [Parquet](../datasets/parquet.md)).

The third level is the `shuffle_block_size` setting, which allows us to optionally shuffle
samples within a single `Dataset` across a window of samples that may cross shards (it
defaults to `None`, which disables this additional shuffling). If you don't have a strong
sense of what a good shuffle window size would be for your dataset, setting
`shuffle_block_size="auto"` will have Zephon choose a shuffle block size that is equal to
eight times the number of samples in the largest shard in any `Dataset` included in the
mixture (or simply the size of the `Dataset` itself if it happens to be smaller than that
window size). This setting provides cross-shard mixing over several shards and is a solid
default until you have a compelling reason to set a different window size yourself.

Finally, setting the `shuffle_block_size` to the string `"global"` will apply a truly
global shuffle across the entire `Dataset`. Note that this requires us to materialize all
sample IDs of the entire `Dataset` upfront and shuffle them. At trillion-token scale
training, this might not be possible as just the sample identifiers can take hundreds of
gigabytes of DRAM, and this is slow. Our other shuffle algorithms avoid full
materialization of the identifiers up front.

A global shuffle is not as important as you might think it is: our friends at Mosaic
Streaming have observed that the entropy of a shuffle across the shards is
[extremely close to an actual global shuffle](https://docs.mosaicml.com/projects/streaming/en/latest/distributed_training/performance_tuning.html#shuffle-quality).

(repeating-samples)=

**Repeating Samples.** In addition to determining the order in which the records from
each `Dataset` are processed, we also need to decide when to stop drawing samples from
each `Dataset`, i.e., when an epoch ends. By default, the `StaticMixtureWorkSource`
repeats datasets that finish a pass sooner under the requested mixture in order to
maintain the requested mixture until every dataset has reached the end of its first
pass. To continue drawing samples until every dataset has reached the end of at least
`n` passes, set the `stop_after_passes` argument to `n` instead of the default value of
`1`:

```{literalinclude} ../../../examples/guide/worksources/repeat_passes.py
:language: python
:caption: examples/guide/worksources/repeat_passes.py
```

Note that each time a `Dataset` is processed, its shuffle settings will be applied to it
with a deterministic per-epoch shuffle seed in order to generate a different ordering of
the samples for that pass; you can disable this behavior by setting the
`reshuffle_on_repeat` argument to `False`.

The `stop_after_passes` argument controls how many passes **every dataset** must complete
before the work source stops. Datasets that reach this target sooner continue repeating to
maintain the requested mixture while the remaining datasets catch up.

You can instead control what happens when an individual dataset runs out of samples using
`exhausted_policy`:

- `exhausted_policy="stop"` stops the entire work source as soon as any dataset can no
  longer supply its requested allocation. No dataset repeats, but other datasets may still
  have unread samples. This means that setting both `exhausted_policy="stop"` and
  `stop_after_passes` is an invalid configuration.
- `exhausted_policy="repeat"` restarts each dataset independently whenever it runs out.
  Without an additional stopping limit, the work source continues indefinitely. This is
  useful when your training loop controls termination through a fixed number of steps or
  tokens.

With `exhausted_policy="repeat"`, you can also set `max_repeats=N`. This allows each
dataset its initial pass plus at most `N` additional passes. The entire WorkSource stops
when any dataset can no longer supply its requested allocation without exceeding that
limit. Other datasets may have completed fewer passes, or may still be on their first
pass.

`stop_after_passes` waits for every dataset to reach a minimum number of passes, while
`max_repeats` stops the source when any dataset would exceed its maximum number of
repeats. Just like with `exhausted_policy="stop"`, `stop_after_passes` can also not be
combined with `max_repeats`.

When none of these settings is supplied, Zephon defaults to `stop_after_passes=1` and
automatically repeats datasets as needed. Specifying `exhausted_policy` or `max_repeats`
replaces this default stopping behavior. Assuming you don't set `max_repeats`:

| What you supply | Effective `exhausted_policy` | Effective `stop_after_passes` | When the source stops |
| --- | --- | --- | --- |
| Nothing | `"repeat"` | `1` | Once every dataset has completed at least one pass |
| Only `exhausted_policy="repeat"` | `"repeat"` | `None` | Never automatically; datasets repeat indefinitely |
| Only `exhausted_policy="stop"` | `"stop"` | `None` | When any dataset cannot supply its next allocation, no repeats |
| Only `stop_after_passes=N` | `"repeat"` | `N` | Once every dataset has completed at least `N` passes |

The `exhausted_policy`, `max_repeats`, and `reshuffle_on_repeat` settings can each be
configured per-`Dataset` to allow for more complex training plans:

```{literalinclude} ../../../examples/guide/worksources/per_dataset_repeat_settings.py
:language: python
:caption: examples/guide/worksources/per_dataset_repeat_settings.py
```
