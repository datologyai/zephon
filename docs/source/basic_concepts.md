# Basic Concepts

Zephon separates *what* you train on from *how* data is loaded and transformed, also known as last-mile data processing.
This page introduces each of Zephon's core concepts step by step.
While some of these concepts may seem like implementation details, for proper usage of Zephon and understanding features like elastic determinism,
it is key to build a rough mental model on how the system works.

---

## Data on Disk: Datasets, Shards, and Samples

Before diving into how Zephon processes data, let us establish how we
think about data stored on disk.

A **dataset** is a named collection of data that lives at a path --- a local
directory, an S3 bucket, a GCS prefix, etc. Physically, a dataset is split
into one or more **shards**: individual files that each contain a slice of the
data. A **sample** is the smallest addressable unit within a shard --- one
JSON object in a JSONL file, one row in a Parquet file, one record in an MDS
shard, and so on.

```
my_dataset/
  ├── shard_000.jsonl    # shard 0:  samples 0–999
  ├── shard_001.jsonl    # shard 1:  samples 0–1249
  └── shard_002.jsonl    # shard 2:  samples 0–873
```

Zephon is **not tied to a single file format**. It ships with built-in
support for JSONL, Parquet, MosaicML Streaming / MDS, LitData, and Vortex,
and auto-detects the format when you point it at a directory. For testing and
prototyping, Zephon also supports in-memory datasets where you pass sample
data directly as Python dicts. The abstraction Zephon works with is always
the same: a dataset has shards, and each shard has a known number of samples
that can be addressed by index. Everything above this layer is format-agnostic.

---

## Declaring Your Data

Everything starts with a {py:class}`~zephon.work.WorkSource`. A WorkSource defines the training
curriculum: which datasets, in what proportions, in what order. It produces
**work chunks**, i.e., fixed-size sets of **pointers** to samples, each a
`(dataset_id, shard_id, sample_idx)` triple. The WorkSource only decides *which* samples to train on. No I/O happens here.
Right now, Zephon only implements the {py:class}`~zephon.work.StaticMixtureWorkSource`.

```python
from zephon.io import Dataset
from zephon.work import MixtureSpec, StaticMixtureWorkSource

fineweb = Dataset.from_path("fineweb", "/data/fineweb")
dclm = Dataset.from_path("dclm", "/data/dclm")

ws = StaticMixtureWorkSource(
    datasets=[fineweb, dclm],
    mixture=MixtureSpec({"fineweb": 0.7, "dclm": 0.3}),
    chunk_size=16384,
    seed=42,
)
```

A {py:class}`~zephon.io.Dataset` describes where the data lives (e.g., local path, distributed file system, S3, GCS) and its
format (JSONL, Parquet, MDS, etc.). A {py:class}`~zephon.work.MixtureSpec` sets the target
proportions (**mixture**). In the example above, based upon this information, the implementation of the {py:class}`~zephon.work.StaticMixtureWorkSource` generates the chunks, respecting the mixture weights via {py:class}`Smooth Weighted Round Robin <zephon.utils.swrr.SmoothWeightedRoundRobin>`. For example, a chunk of size 100 would contain 70 pointers to samples in FineWeb, and 30 pointers to samples in DCLM.

This separation of *what to train on* from *how to process it* is one of
Zephon's central design principles. In the future, we can introduce, for example, more declarative work sources with custom DSLs, or work sources that yield data based on active curricula.

### Shuffling

Because the WorkSource owns the curriculum (the global ordering of what the
model sees) it is also the natural place to control shuffling.
{py:class}`~zephon.work.StaticMixtureWorkSource` offers three
orthogonal knobs, applied in order when the sample sequence is constructed:

1. **Shard shuffling** (`shuffle_shards=True`, on by default) --- permutes the
   order in which shards are visited. This is the coarsest level: it avoids
   always reading shards in the same sequence, but samples within each shard
   still appear in their natural order.

2. **Within-shard shuffling** (`shuffle_within_shard=False`, off by default)
   --- independently permutes the sample offsets inside each shard. This adds
   finer-grained randomness but means shards are no longer read sequentially,
   which can reduce I/O locality.

3. **Block shuffling** (`shuffle_block_size=None`, off by default) --- after
   the full per-dataset sample sequence has been assembled, it is partitioned
   into non-overlapping windows of `shuffle_block_size` samples, and each
   window is shuffled independently. This creates controlled cross-shard
   mixing: samples from different shards can appear adjacent, but only within
   a local window. Smaller blocks preserve I/O locality; larger blocks
   make for a better shuffle.

   In addition to a positive integer, `shuffle_block_size` accepts two
   sentinels: `"auto"` resolves to `8 × max(shard_size)` across all datasets
   in the mix, giving a sensible default that grows with the largest shard.
   `"global"` resolves to *that dataset's* total sample count, i.e. one block
   per dataset --- the strongest in-dataset shuffle, at the cost of buffering
   on the order of the dataset's total samples in memory while the block
   is materialized. The resolved per-dataset value is locked into the
   checkpoint, so resuming after adding shards keeps the original block
   size rather than silently re-resolving the sentinel.

If you are coming from Mosaic Streaming and wondering how Zephon's shuffling
knobs relate to algorithms like `py1s` or `py1br`, see
[Transitioning from Streaming](transitioning.md) after understanding the basic concepts.
Shuffling and the per-dataset repetition controls (exhaustion policy and `stop_after_passes`) are the curriculum-level controls that this {py:class}`~zephon.work.StaticMixtureWorkSource` implementation currently has.
If you need control on the data curriculum beyond this, a new WorkSource is needed.

It is important to note that shuffling and mixing are not *exclusively*
WorkSource concerns. As we will see below, steps in the preprocessing
pipeline, such as tokenization splitting a long document into multiple
sequences, or sequence packing merging short samples, can alter the
effective ordering and mixture ratios of what eventually reaches the training
loop. Zephon provides pipeline-level tools like
{py:meth}`Pipeline.shuffle() <zephon.api.Pipeline.shuffle>` (a buffer-based
shuffle step) and
{py:meth}`Pipeline.ensure_mixture() <zephon.api.Pipeline.ensure_mixture>` to
address exactly this interplay. Users should be aware that WorkSource-level
shuffling sets the *initial* ordering, but downstream processing may require
additional shuffling or mixture correction to maintain the desired guarantees
end-to-end.

---

## Building the Pipeline

Once you have declared *what* to train on via a WorkSource, you need to
describe *how* to prepare the data for your model. For example, if your input are strings, this preparation might include tokenizing the text and grouping samples into batches, but it
can also involve image decoding, sequence packing, filtering, and other
transformations, depending on the workload.

```{note}
For LLM pipelines that use fully pre-tokenized and pre-packed data, the only
processing step may be batching. In this case the online processing
aspect is minimal and the pipeline is essentially just a data feeder.
```

In Zephon, you describe this sequence of processing steps using a
{py:class}`~zephon.api.Pipeline`. The Pipeline provides a builder-style API:
each method call adds another step to the chain.

```python
from zephon.api import Pipeline
from zephon.io import Dataset
from zephon.work import MixtureSpec, StaticMixtureWorkSource

# Using the same WorkSource from above (or any other)
fineweb = Dataset.from_path("fineweb", "/data/fineweb")
ws = StaticMixtureWorkSource(
    datasets=[fineweb],
    mixture=MixtureSpec({"fineweb": 1.0}),
    chunk_size=16384,
    seed=42,
)

pipeline = (
    Pipeline(ws)
    .tokenize(tokenizer_id="gpt2",
              field="text",
              max_length=2048,
              split_long_samples=True)    # text --> token IDs
    .ensure_mixture()                     # fix token-level mixture drift
    .batch(microbatch_size=8)             # group into training batches
)
```

Reading this top to bottom: take samples from the WorkSource, tokenize them,
correct the mixture ratios, and collect them into batches of 8. The result is
a standard Python iterable --- you consume it with a regular `for` loop:

```python
for batch in pipeline:
    train_step(batch.to_training())
```

Note that the pipeline does not do anything until you start iterating over it.
Each processing step in the chain is called an **operator**. You typically
instantiate operator classes yourself; the Pipeline builder creates and wires
them for you. Some of the key built-in operators are:

- {py:meth}`~zephon.api.Pipeline.tokenize` --- tokenize text using any
  HuggingFace-compatible tokenizer
- {py:meth}`~zephon.api.Pipeline.batch` --- collect samples into
  fixed-size training batches
- {py:meth}`~zephon.api.Pipeline.ensure_mixture` --- correct mixture ratios
  after processing steps that change sample sizes (explained
  [below](#keeping-mixtures-on-track))
- {py:meth}`~zephon.api.Pipeline.shuffle` --- deterministic buffer-based
  shuffle within the pipeline
- {py:meth}`~zephon.api.Pipeline.map_transform` --- apply an arbitrary
  Python function to each sample
- {py:meth}`~zephon.api.Pipeline.pack_sequences` --- bin-pack short
  tokenized sequences into full-length samples

```{note}
The design about custom operators is currently still a bit in flux. We might add
an API/interface to allow fully custom operators beyond map transformations down
the line.
```

Operators can have different input-output relationships: most transform one
sample into one output (**1:1**), but tokenization can split a long document
into multiple sequences (**1:N**), and packing or batching merge multiple
samples into one (**N:1**). For a deeper look at how samples flow through
these operators, see [Sample Lifecycle](understanding/sample_lifecycle.md).

### From pointers to data

The WorkSource produces lightweight pointers, but at some point Zephon needs
to read the actual data. This happens automatically: every Pipeline starts
with a built-in {py:class}`~zephon.ops.FetchOp` that you never add yourself.
When the pipeline runs, FetchOp takes each pointer, opens the corresponding
shard file, and loads the sample's raw payload into a
{py:class}`~zephon.core.SampleRecord`. All subsequent operators work on this
loaded data.

When loading from remote storage (S3, GCS, maybe even on a slow DFS), you can optionally insert a
{py:class}`~zephon.ops.PrefetchOp` that looks ahead in the pointer stream and
downloads shards to a local cache before they are needed. This reduces fetch
latency without changing the data in any way. It is purely a performance
optimization.

```python
pipeline = (
    Pipeline(ws)
    .prefetch(buffer_size=2048)    # look ahead and warm the cache
    .tokenize(tokenizer_id="gpt2", field="text")
    .batch(microbatch_size=8)
)
```

For local storage, prefetch is unnecessary and can be omitted. See the
[Prefetch Operator Guide](guides/prefetch_op.md) for tuning advice.

### What a sample looks like inside the pipeline

Once FetchOp has loaded a sample, it becomes a
{py:class}`~zephon.core.SampleRecord`. This is a container that pairs a **payload**
with internal tracking metadata. The payload is typically a Python dict whose
keys are called **fields**:

```python
# A text sample after fetch might look like:
{"text": "Zephon is a modular data loader for ..."}

# After tokenization, the tokenizer adds fields:
{"text": "Zephon is a ...", "input_ids": [464, 89, ...], "attention_mask": [1, 1, ...]}
```

Operators read and write these fields as they transform samples. For example,
the tokenizer reads the `text` field and writes `input_ids` (it may also drop the text afterwards). This is why
operators like `ensure_mixture` can reference fields by name. As we will
see in the next section, `weight_by="auto"` looks for common token-count
fields like `input_ids` to measure how many tokens each sample contributes.

After the `batch` operator, individual records are collected into a
`SampleBatch`. This object offers certain convenience functions. For example, when you call `batch.to_training()` in your training loop, the
batch's payloads are collated into the dict-of-lists format many models in training frameworks such as torchtitan typically
expect. For a deeper look at how samples are created, split, and packed, see
[Sample Lifecycle](understanding/sample_lifecycle.md).

---

## Keeping Mixtures on Track

The WorkSource mixes datasets at the **sample level**, e.g., 70% of pointers
come from FineWeb, 30% from DCLM. But tokenization changes the game. One
FineWeb sample might produce 500 integer tokens while one DCLM document might produce
2,000. After tokenization, the *token-level* ratio (which eventually ends up at the model) drifts away from 70/30.

{py:meth}`Pipeline.ensure_mixture() <zephon.api.Pipeline.ensure_mixture>` fixes this. It observes
the actual token counts flowing through and reorders samples using {py:class}`Smooth
Weighted Round Robin (SWRR) <zephon.utils.swrr.SmoothWeightedRoundRobin>` to bring the token-level mixture back to target.

```python
pipeline = (
    Pipeline(ws)
    .tokenize(tokenizer_id="gpt2", field="text", max_length=2048)
    .ensure_mixture(weight_by="auto")   # auto-detects token field
    .batch(microbatch_size=8)
)
```

As described [above](#what-a-sample-looks-like-inside-the-pipeline), each
sample carries named fields. By default, `weight_by="auto"` looks for common
token-count fields (`input_ids`, `tokens`, etc.) and weights by the number of
tokens in each sample. You can also pass `weight_by="samples"` for
sample-level enforcement, or provide a custom callable. This even works correctly
when you pack samples and one sample might contain samples that originally stem from different mixing domains.

---

## Execution: Stages and Runners

So far we have talked about *what* the pipeline does. But how does it
actually run? Behind the scenes, Zephon does not simply execute operators
one after another in a single nested loop. Instead, it groups operators into
**stages** and assigns each stage to a **runner**, an execution backend
that determines how the work is parallelized.

Before the pipeline runs, Zephon **compiles** it: the Pipeline's operator
graph is optimized and translated into a sequence of stages, each annotated with a runner
type and worker counts. This compilation step applies deterministic rules
based on operator properties and any overrides you provide via
{py:meth}`Pipeline.options() <zephon.api.Pipeline.options>`. You can inspect the compiled plan with
{py:meth}`~zephon.api.Pipeline.explain` (shown
[below](#inspecting-your-pipeline)).

Three runner types are currently available:

| Runner | How it works | Best for |
|---|---|---|
| **Inline** | Synchronous, single-threaded | Batching, lightweight post-processing, debugging |
| **Thread** | Shared thread pool within the main process | I/O-bound work (fetch, decode), free-threaded Python |
| **Process** | Separate subprocesses with IPC | CPU-bound work that needs true parallelism |

By default, most operators run in the **thread runner**. The main exception
is `batch`: when it is the last operator in a stage (the common case), it is
placed in its own **inline** stage, since batching is lightweight and benefits
from avoiding threading overhead. If you add post-batch operators (e.g.,
{py:meth}`~zephon.api.Pipeline.map_batch`), the batch stage switches to
threads to avoid any potential serialization of tensors. You can
override the runner for any stage via `Pipeline.options(per_stage_runner=...)`.

```{note}
The default of threads is to ensure quick iteration and prototyping. 
Unless you are using GIL-free Python, you probably want to use the process-based runner. We will elaborate on the runners and their tradeoffs in a future version of this documentation.
The API on how to assign runners to stage and set stage breaks is also still a bit in flux.
```

Each operator within a stage has a configurable **parallelism**, i.e., the
number of concurrent workers that execute it. For example, `fetch` and
`tokenize` might each run with 4 workers inside the same stage,
processing different samples simultaneously. You can tune this per operator
(e.g., `.tokenize(..., parallelism=8)`) or let Zephon pick defaults. The
`explain()` output displays this as `@pN` next to each operator name.

The key idea behind this model is that operators do not need to know how they are executed. The
same `tokenize()` step runs identically whether in a thread pool or across
four subprocesses.

Crucially, all stages run **concurrently**. While a downstream batch stage
is assembling the current batch, the upstream fetch/tokenize stage is already
processing the next samples. This overlap (or pipelining) is where much of Zephon's
throughput comes from: the GPU training step, data fetching, and
tokenization all happen in parallel rather than sequentially.

Stages connect through bounded queues. When a downstream queue fills up,
upstream producers block automatically. This **backpressure** prevents
the pipeline from accumulating unbounded work in memory. At the pipeline
output, you can set `prefetch_batches` in
{py:meth}`Pipeline.options() <zephon.api.Pipeline.options>` to pre-fill a
buffer of fully-processed batches ahead of the training loop. As long as
this buffer stays non-empty, the GPU never stalls waiting for data.

---

## The Engine

The {py:class}`~zephon.core.Engine` is the runtime that ties everything
together. When you iterate over a Pipeline, the Engine:

1. Requests work chunks from the WorkSource
2. Expands pointers into the sample stream
3. Wires each stage to its assigned runner
4. Manages **lanes** --- independent data streams so each GPU sees a
   deterministic, non-overlapping slice of data. In the common case there
   is one lane per GPU, but a single GPU can serve multiple lanes (see
   [Elastic Determinism](understanding/determinism.md) for details)
5. Tracks sample progress for deterministic checkpointing

You rarely interact with the Engine directly. It is created automatically
when you begin iterating and cleaned up when iteration ends.

**Determinism is compositional.** The WorkSource produces pointers in a
fixed order for a given seed. Each operator is individually deterministic
given its inputs. By induction, the entire pipeline produces the same output
for the same configuration and checkpoint state. You can checkpoint
mid-training, restart on different hardware, and get exactly the same global
sample order. For a deeper treatment, see
[Determinism](understanding/determinism.md).

---

## Putting It All Together

Now that we know all the pieces --- WorkSource, Pipeline, operators,
runners, and the Engine --- here is how a sample flows through the full
system, from a pointer to a training batch:

```
WorkSource
    |
    |  work chunk = [(0,3,17), (1,0,42), (0,3,18), ...]
    |                 ^^^^^^^
    v                 one pointer: dataset 0, shard 3, sample 17
+-----------------------------------------------------------+
|                        Engine                              |
|  Expands chunk into a stream of pointers, one per sample.  |
|  Assigns pointers to lanes (typically one per GPU).         |
+----+-------------------------------------------------+----+
     |                                                 |
     v                                                 v
  Lane 0                                  Lane 1
     |
     v  pointers
┌─────────────── Stage 0 (Thread Runner) ──────────────┐
│                                                       │
│  Prefetch ──> FetchOp ──> Tokenize                    │
│  (optional)   pointer     text ──> tokens             │
│  warms cache  ──> data    (1:N)                       │
│                   ^^^^                                │
│          pointers become data here                    │
└──────────────────────┬───────────────────────────────┘
                       |
                       v  samples
┌─── Stage 1 (Inline) ─────────────────────────────────┐
│                                                       │
│  EnsureMixture ──> Batch                              │
│  fix ratios        N ──> 1                            │
└──────────────────────┬───────────────────────────────┘
                       |
                       v  batches
                 Training Loop
```

The key transition is at **FetchOp**: everything before it deals in pointers
(tiny tuples of three integers), everything after it deals in actual data.
The WorkSource and pointer logic stay lightweight regardless of sample size.

Notice how the stages map to runners: the I/O-heavy and compute-heavy
operators (fetch, tokenize) run in a thread pool, while the lightweight
final steps (mixture correction, batching) run inline. Note this is an example pipeline, it would be just as valid to place EnsureMixture in the thread runner. The Engine
orchestrates all of this, feeding pointers in and delivering batches out.

---

## Inspecting Your Pipeline

Call {py:meth}`~zephon.api.Pipeline.explain` to see how Zephon compiled your
pipeline without running it. This shows stages, runner assignments, operator
parallelism, and buffer depths:

```python
print(pipeline.explain())
```

```
Stage[0] place=local break='start' ops=['fetch@p4', 'tokenize@p4']
Stage[1] place=auto break='placement-hint' ops=['ensure_mixture@p1', 'batch@p1']

Execution Graph:
Allocation=fit_to_ops (cap=sum(node.parallelism), min 1/op)
Bookkeeping=contributor-aware (packing/shuffle-safe)
Stage[0] place=local runner=threads cap=8 mode=microbatches
  fetch@p4 -[in_q=16]-> tokenize@p4
  --[stage_out=16]-->
Stage[1] place=local runner=inline cap=1 mode=stream_items
  ensure_mixture@p1 -> batch_replay_filter@p1 -> batch@p1
  ==[final_prefetch=3]==> pipeline_end
```

Reading this output:

- **Stage[0]** uses the thread runner with a total capacity of 8 workers
  (4 + 4). It contains `fetch` and `tokenize`, each with their own
  parallelism (`@pN`). Items flow between operators through in-process
  queues (`in_q=16`).
- **Stage[1]** runs inline (single-threaded) and contains `ensure_mixture`,
  a `batch_replay_filter` (auto-inserted for checkpoint correctness), and
  `batch`. These are lightweight and benefit from running without
  concurrency overhead.
- `stage_out` shows the buffer size between stages.
- `final_prefetch=3` means the pipeline pre-fills 3 batches ahead of the
  training loop, so the GPU never stalls waiting for data.

The `explain()` output is invaluable for diagnosing performance. If your
pipeline is slow, this is the first thing to check.

---

## Why an Iterable Pipeline?

If you have used PyTorch's `MapDataset` or Mosaic Streaming, you may be
used to an **indexable** approach: every training step can be mapped back
to a specific sample index, and the entire sample ordering is
pre-computed before training begins.  This works well when each sample on
disk corresponds one-to-one to a training sample: you shard your
pre-tokenized data, and the loader simply looks up sample *i* at step *i*.

Zephon is an **iterable** pipeline instead.  The reason is that the
one-to-one assumption breaks down as soon as the pipeline does non-trivial
online processing:

- **Tokenization with splitting** turns one document into a variable
  number of training sequences, depending on document length.
- **Sequence packing** merges multiple short samples into one, depending
  on what the packer has buffered.
- **Dynamic or learned mixtures** adjust dataset proportions on the fly. The curriculum at step *N* may depend on the model's state at step
  *N−1*, so it cannot be pre-computed.

In all of these cases, you cannot know which raw sample ends up at
training step *X* without actually running the pipeline up to that point.
An indexable dataset would require  pre-computing and storing the
fully processed tensors (imagine storing raw decoded video frames, the
storage cost alone makes this impractical), and giving up on online
processing entirely.

Zephon opts for online processing and accepts the trade-off.  The indexable
approach has a real advantage: checkpointing and mid-epoch resumption are
cheap, since the loader only needs to store a sample index to know exactly
where to pick up.  An iterable pipeline must capture more state, making checkpoints larger and resumption
more involved if we want to avoid full replay. It is also harder to trace a training step back to the
exact raw sample that produced it.  In exchange, the pipeline can handle the
dynamic, N:M processing patterns that multi-modal and online-processing
workloads demand without requiring a separate offline preprocessing stage.

If you are coming from Mosaic Streaming and want to understand the
broader architectural differences, see
[Transitioning from Streaming](transitioning.md).

---

## Next Steps

This page gave you the conceptual map. To go deeper:

- [WorkSources](understanding/worksources.md) --- data selection, mixture
  strategies, and shuffling modes
- [Sample Lifecycle](understanding/sample_lifecycle.md) --- how samples are
  created, split, packed, and batched in full detail
- [Accumulators and Operators](understanding/accumulators_operators.md) ---
  the operator execution model and how to write custom ops
- [Determinism](understanding/determinism.md) --- reproducibility
  guarantees and elastic determinism across GPU counts
- [Checkpointing](understanding/checkpointing.md) --- saving and restoring
  pipeline state for fault-tolerant training
- [Prefetch Operator Guide](guides/prefetch_op.md) --- optimizing remote
  storage performance
- [API Reference](api/zephon_api) --- full reference for the Pipeline
  class and all operators
