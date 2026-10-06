# WorkSources: The Training Curriculum

**The center of Zephon.** The `WorkSource` defines the training curriculum: which samples
Zephon should use, and the order they reach the pipeline in. The WorkSource uses the
metadata provided by the `Dataset` objects and the configuration options you specify to
construct a deterministic plan for the run as a sequence of **work chunks**, each of which
is a fixed-size list of pointers to samples. Work chunks are the basic unit that the
`Pipeline` uses to organize its work across GPUs, track progress, and record state in
checkpoints.

All of the examples and code here reference the `StaticMixtureWorkSource` class, which is
the primary implementation of the `WorkSource` abstraction in Zephon today and is the one
that we use for our regular model training pipelines. By default, a
`StaticMixtureWorkSource` puts 16,384 pointers in each work chunk, but this can be
adjusted with the `chunk_size` parameter on its constructor. We find this to be a good
balance between per-chunk overhead and state size. Please note that some examples in the
documentation use a deliberately smaller chunk size (e.g., 4) to demonstrate certain
effects and that should not be carried into production.

The simplest `StaticMixtureWorkSource` is one that processes a single `Dataset` instance:

```{literalinclude} ../../../examples/guide/worksources/single_dataset_worksource.py
:language: python
:caption: examples/guide/worksources/single_dataset_worksource.py
```

Here, the `datasets` argument provides a list of `Dataset` objects whose metadata is used
to construct the work chunks, and the `mixture` is a mapping from `Dataset` names to their
relative weights in the mixture (here, a weight of `1.0` indicates that every sample
selected comes from the example `Dataset`). The `seed` parameter is used to make the
resulting curriculum reproducible across training runs.

The remaining subpages explain how to use `StaticMixtureWorkSource` to combine samples
from multiple `Dataset`s, how to size the work chunks appropriately, and how to control
how samples are ordered and repeated during training.

:::{note}
**Future WorkSources.** We're working on support for more advanced WorkSources, e.g. for
dynamic curricula or a SQL-like interface inspired by data loaders like
[Mixtera](https://github.com/eth-easl/mixtera). Stay tuned!
:::

```{toctree}
:maxdepth: 2

mixing_datasets
shuffling_and_repeating_samples
```
