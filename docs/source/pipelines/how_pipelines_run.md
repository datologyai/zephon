# How Pipelines Run

A `Pipeline` definition reads like a sequence of steps: fetch the records, tokenize them,
pack them, shuffle them, and batch them. But underneath the hood, Zephon is executing
these individual steps in parallel, such that at any given moment the engine may be
downloading shards for future processing, tokenizing records from several different data
sources, and packing and shuffling the results.

You don't need to understand how Zephon does this in order to use it, but knowing the
basics about the execution model will help us explain a couple of key features that will
come up in the rest of this guide. First, why changing the settings that control how a
`Pipeline` runs never changes the data that it produces, and why certain stateful
operations like shuffling and packing samples behave the way that they do. This page
introduces Zephon's runtime execution model, while the subsequent
[Inspecting and Tuning Pipelines](inspecting_and_tuning_pipelines.md) page covers how to
measure and adjust it.

**From Operators To Stages.** When your code begins iterating over a `Pipeline`, Zephon
first compiles the chain of operators into an execution plan. Consecutive operators are
grouped together into **stages**, and each stage is executed by a **runner** that
determines how the work in that stage is run. Right now, the default runner is `threads`,
which uses a pool of worker threads to run each operator inside the same process. The
`process` runner executes each operator's work in separate Python processes, which is
useful for CPU-intensive Python code that is subject to Python's infamous global
interpreter lock. Finally, the `inline` runner does the work directly on the thread that
is actually driving the stage, and Zephon prefers this runner for lightweight operators
like `batch`, where the overhead of handing off the work to a thread or a separate process
would cost more than it saves. You can control which runner the `Pipeline` defaults to by
calling `Pipeline.options(runner="<runner>")`.

Records flow from one operator to the next through bounded queues, so every operator in
the plan can be working at the same time on its own part of the stream. You can see the
plan that Zephon will execute by calling `Pipeline.explain()`:

```{literalinclude} ../../../examples/guide/pipelines/pipeline_explain.py
:language: python
:caption: examples/guide/pipelines/pipeline_explain.py
```

Each operator is listed with its parallelism level (`tokenize@p4` means that the tokenize
operator runs with four workers), along with the sizes of the queues between the operators
and the stages and the number of finished batches of training data that the `Pipeline`
prepares ahead of the training loop (`final_prefetch`). You don't need to configure any of
this yourself, but it's a useful reference when you want to understand where the resources
that a `Pipeline` uses are going.

**Parallelism Without Changing The Data.** When an operator runs with multiple workers,
Zephon hands its input records out to those workers in small groups that we call
**micro-batches**. Micro-batches are an internal grouping for processing efficiently, and
are not the same thing as the batches you pass to training (apologies for this; naming
things is hard). The workers may finish their micro-batches in any order, but in its
default deterministic mode, Zephon tags each micro-batch with a sequence number before
passing it to the worker and puts the results back in order before it passes them down to
the next operator. The output of the operator is therefore identical to what it would have
produced if a single worker had processed every record in order.

Operators that need to track state across their records, like shuffling and packing, keep
that state in a single-threaded component (the *accumulator*) that sees every record in
order, while only the stateless part of their work is handed out to the parallel workers.
This means that the decisions that the operators make, like which records get packed
together, never depend on how the work was divided up among the workers.

Together, these two properties mean that the execution settings of a `Pipeline` are purely
performance settings. Changing them can make a `Pipeline` run faster or slower, but it
never changes the sequence of batches that the `Pipeline` produces, and it never prevents
a checkpoint from being restored, so you can tune them freely without invalidating an
experiment.

If you don't need determinism for a particular run, setting
`Pipeline.options(deterministic=False)` lets each stage pass along results in whatever
order they finish, which may reduce latency at the cost of the ordering guarantees
described above.

**Tracking Progress Through Chunks.** Recall from the
[Basic Concepts](../basic_concepts.md) that a `WorkSource` hands out the training
curriculum as a sequence of work chunks, which are groups of samples that are fetched and
processed together. Chunks are also how Zephon tracks the progress of a `Pipeline`: as
records move through the operators, Zephon keeps track of which chunk each of them came
from, including the records that are split, filtered, packed, or held in a buffer. A chunk
is only considered complete once all records that were derived from it have been delivered
to the training loop.

The queues between the pipeline's operators are bounded, so if the training loop starts to
fall behind, the operators upstream of it will pause instead of fetching more and more
data. This backpressure mechanism means that only a limited number of chunks are ever in
progress at the same time, which keeps the resource usage of a `Pipeline` bounded. It is
also the thing that keeps checkpoints small, because instead of needing to save the
internal state of each operator, a checkpoint only needs to record the chunks that were
still in progress, which the `Pipeline` can re-process when it is restored.

Re-processing chunks is sufficient for operators that aren't stateful, like
`map_transform` and `tokenize`. But operators that buffer and reorder records, like
`shuffle` and the packing ones, may have state that depends on every record they have seen
since the run began. To ensure that the amount of information needed for checkpointing
stays bounded, Zephon will periodically send a **flush sentinel** through the `Pipeline`.
When the flush sentinel arrives, any stateful operators emit whatever information they are
holding and return to a clean state, so that a restored `Pipeline` only needs to
re-process the chunks since the most recent flush sentinel in order to rebuild the
operators' state exactly.

Zephon only sends flush sentinels when a `Pipeline` includes an operator that reorders or
regroups records, like `shuffle`, `ensure_mixture`, or one of the packing operators. In
that case, a flush will happen every eight work chunks by default, which can be modified
via `Pipeline.options(flush_every_k_chunks=...)`. Flushing less often gives the stateful
operators more records to work with between resets and reduces the partial outputs that
each flush produces (which are detailed on the individual operator pages in the rest of
this document), but comes at the cost of more memory usage and more work for Zephon to do
when a `Pipeline` is restored from a checkpoint.

**Buffering Ahead of the Training Loop.** The last stage of a `Pipeline` maintains a
buffer of finished batches that are ready for the training loop, which smooths out short
variations in how long each batch takes to prepare. Zephon derives the size of this
buffer, along with the sizes of the queues between operators and stages, from the number
of workers that are available on each machine. The
[Inspecting and Tuning Pipelines](inspecting_and_tuning_pipelines.md) page gives more
detail on when and how you should change them.

By default, the `Pipeline` machinery for running stages runs inside of the training
process, where they can compete with the training loop for Python's global interpreter
lock. Using `Pipeline.options(mtp_mode=True)` runs the `Pipeline` in a separate process
instead, such that only finished training-ready batches are handed back to the main
training process. Both of our [Training Integrations](../training_integrations.md) enable
this mode for their pipelines, and we generally recommend it for production runs.

Finally, keep in mind that in a distributed training run, every rank that loads data will
run its own `Pipeline` instance. The [Distributed Training](distributed_training.md) page
covers how these `Pipeline` instances can be configured to divide up the training
curriculum and share the compute resources on each node.
