# Pipelines: From a Curriculum to Training Batches

A `Pipeline` turns the curriculum defined by a `WorkSource` into the batches of tensors
that are consumed by your training loop. It fetches the underlying records defined in the
`Dataset`s that comprise the `WorkSource`, applies any transformations that are defined on
the `Pipeline` to those records, and then optionally groups the resulting records into a
batch for training. For some workloads, your `Pipeline` definition may be very simple — it
may only need to fetch and batch records. For other workloads, you can define rich
transformations on your `Pipeline` in order to decode, transform, tokenize, shuffle, and
pack the samples prior to sending them to the training loop.

<figure class="zephon-post zephon-embed">
  <iframe src="../_static/embeds/operator-model.html" height="240" loading="lazy"
          title="Zephon's operator model"></iframe>
  <figcaption>
    <strong>Zephon's operator model.</strong> The stages a sample passes through: the
    <code>WorkSource</code> turns a training
    curriculum into mixture-aware per-lane chunks, the operator graph fetches and
    transforms them, and the training loop — outside Zephon — consumes the result.
  </figcaption>
</figure>

Building off of the web/code `StaticMixtureWorkSource` introduced previously, your first
pipeline might look something like this:

```{literalinclude} ../../../examples/guide/pipelines/first_pipeline.py
:language: python
:caption: examples/guide/pipelines/first_pipeline.py
```

This kind of simple pipeline definition is appropriate when the data you are training on
has mostly been prepared offline. The `batch` operator groups records into `SampleBatch`
objects, which [Building Training Batches](building_training_batches.md) shows how to
convert into tensors for your model.

Most training frameworks are happy to consume data from any kind of Python iterable, like
a Zephon `Pipeline`, directly. However, if you are using a PyTorch-based training
framework that really requires a torch `IterableDataset` or a `DataLoader`, Zephon
provides an adapter method that allows you to treat a `Pipeline` object as if it were an
`IterableDataset` that can be passed in to a `DataLoader` like this:

```{literalinclude} ../../../examples/guide/pipelines/torch_dataloader.py
:language: python
:caption: examples/guide/pipelines/torch_dataloader.py
```

In general, we do not recommend adding this extra level of conversion if you don't
absolutely need it; it's generally simpler to act on the abstractions that Zephon provides
for processing records and checkpointing state directly instead of trying to divide these
responsibilities between the Zephon and PyTorch libraries.

The rest of this section takes you through the life of a `Pipeline`, from definition to
production. We start with an overview of how Zephon executes `Pipeline`s, then cover the
built-in operators for turning raw samples into training data (preparing, shuffling,
mixing, packing, and batching records), and then discuss what you need to know to run a
`Pipeline` in a real training job, from performance tuning to distributed training and
checkpointing. Finally, we'll close out this section with how you can write your own
operators for pipelines that need more than the built-in ones that we provide.

```{toctree}
:maxdepth: 2

how_pipelines_run
preparing_samples
shuffling_and_maintaining_mixtures
packing
building_training_batches
inspecting_and_tuning_pipelines
distributed_training
checkpointing_and_resuming
user_defined_operators
```
