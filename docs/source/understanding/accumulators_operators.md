# Accumulators and Operators

The [Basic Concepts](../basic_concepts.md) page introduced operators as the
processing steps in a Pipeline and briefly mentioned runners as the execution
backends that run them.  This page goes deeper: what operators actually are,
how they can run in arbitrary parallelism without breaking determinism, and
what role accumulators play in making all of this work.

---

## Operators

An **operator** is a unit of data transformation in a Zephon pipeline.
Tokenizing text, packing sequences, batching samples, shuffling: each of
these is an operator.  Operators are the building blocks you compose via the
{py:class}`~zephon.api.Pipeline` builder to describe *how* your data is
processed:

```python
pipeline = (
    Pipeline(ws)
    .tokenize(tokenizer_id="gpt2", max_length=2048)
    .shuffle(buffer_size=4096, seed=42)
    .batch(microbatch_size=8)
)
```

An operator does not know or care how it is executed.  The same `tokenize`
step produces the same output whether it runs in a thread pool with 8
workers or inline with a single worker.  This separation of *what* an
operator does from *how* it is parallelized is central to Zephon's design.

### The operator contract

Every operator satisfies a small contract.  The three most important methods
are:

- **`accumulator(deterministic, ctx)`**: returns an
  [accumulator](#accumulators) that decides how incoming elements are grouped
  into [micro-batches](#micro-batches) for processing.  The `deterministic`
  flag is passed by the runner and controls whether time-based flushing is
  allowed (see [CountingAccumulator](#built-in-accumulators) below).
- **`process_many(elems)`**: the workhorse.  Takes a list of elements
  (a micro-batch), transforms them, and returns a list of results.  This is
  the function that runs in parallel across workers.
- **`traits()`**: declares properties that guide compilation and
  scheduling, such as the suggested parallelism and whether the operator
  preserves sample ordering.

Two additional methods complete the contract:

- **`setup(ctx, ...)`**: called once before the operator processes any
  data.  The runner passes an `OpContext` that provides runtime services
  such as access to dataset stores, metrics callbacks, and replay state.
- **`process_one(elem)`**: transforms a single element.  The default
  implementation of `process_many` calls `process_one` per element, so
  operators can override either method depending on whether they benefit
  from batch-level optimizations.

The critical invariant is that **`process_many` must be stateless across
calls**.  It must not rely on anything from a previous invocation.  Any
cross-invocation state (buffers, counters, open bins) belongs in the
accumulator, not in `process_many`.  This invariant is what makes it safe to
dispatch `process_many` calls to an arbitrary number of workers in parallel:
each call is independent, and results can be reordered back into the
original input order without loss of correctness.

Not every operator splits its work evenly between accumulator and workers.
For operators like {py:class}`~zephon.ops.TokenizeText` or
{py:class}`~zephon.ops.FetchOp`, the accumulator merely groups elements
into fixed-size micro-batches and the real work (tokenization, disk I/O)
happens in `process_many` across parallel workers.  But for operators whose
core logic is inherently stateful and serial, such as
{py:class}`~zephon.ops.PackSequences`,
{py:class}`~zephon.ops.EnsureMixture`, and
{py:class}`~zephon.ops.ShuffleBuffer`, the accumulator does the heavy
lifting.  In the current implementation their `process_many` is a trivial
passthrough (`return list(elems)`), though in principle some of their work
(e.g., payload assembly for packing) is stateless and could be moved to
`process_many` to run on parallel workers.  The key architectural
requirement is that any logic that needs to see elements in stream order
*must* live in the accumulator.

```{note}
We should investigate this a bit more. The shuffling and packing _should_
actually happen within the operator to optimize performance. This is most likely
a (performance) bug.
```


### Input-output cardinalities

Operators can have different relationships between input and output:

| Cardinality | Meaning | Examples |
|---|---|---|
| **1:1** | One input produces exactly one output | Tokenize (without splitting), map transforms |
| **1:N** | One input produces multiple outputs | Tokenize with `split_long_samples=True` (one document becomes multiple sequences) |
| **N:1** | Multiple inputs are merged into one output | Batch, pack sequences |
| **1:0** | An input is dropped (filtering) | Map transform returning `None` |

Filtering (1:0) is deterministic as long as the filter function itself is
deterministic, meaning it depends only on the sample content, not on
external state or wall-clock timing.

Operators that change the number of samples (1:N, N:1) or reorder them
interact with [Checkpointing](checkpointing.md) through contributor
tracking.  See [Sample Lifecycle](sample_lifecycle.md) for the full
details.

---

## Micro-batches

```{note}
Micro-batches in this section refer to the internal batches of elements
dispatched to operator workers, not the training batches (the output
of the {py:meth}`~zephon.api.Pipeline.batch` operator) that the model trains
on.
```

When an operator runs with multiple workers, the runner does not dispatch
samples one at a time.  Instead, it groups samples into **micro-batches**,
small lists of elements that are processed together in a single
`process_many` call.

Micro-batches exist for efficiency: they amortize the overhead of worker
dispatch and enable operators to exploit batch-level optimizations (e.g., a
tokenizer can batch-encode multiple texts in one call).  The size of a
micro-batch is not the same as the training batch size; it is a Zephon-internal
grouping decided by the operator's accumulator.

For example, a tokenizer might process micro-batches of 64 samples at a
time.  When the runner has 4 workers, four such micro-batches can be in
flight simultaneously, giving a throughput of 256 samples being tokenized
concurrently while each `process_many` call sees a manageable batch of 64.

The question is: who decides the micro-batch boundaries?  That is the
accumulator's job.

---

## Accumulators

An **accumulator** is a lightweight, single-threaded object that sits between
an operator's input stream and its worker pool.  It receives elements from
upstream in order, decides when a micro-batch of elements is ready for
processing, and hands that micro-batch off to workers. 

The accumulator is guaranteed to run **serially**: the runner never calls
into it from multiple threads at the same time, and elements always arrive
in upstream order.  In the current runner implementations this is achieved
by a dedicated pump thread per operator, but the architectural contract is
simply that the accumulator is never invoked in parallel with itself.
Workers execute the stateless `process_many`; the accumulator decides
*what* each worker sees.

```
Upstream elements
       │
       v
┌──────────────────────────────────────────────────────────┐
│  Accumulator (runs serially)                              │
│                                                           │
│  accumulator.push_many(elems)                             │
│      │                                                    │
│      ├── not enough yet → buffer internally               │
│      └── ready → emit micro-batch                         │
│                         │                                 │
│  Runner tags micro-batch with seq# (monotonic counter     │
│  used for deterministic reordering, see below)            │
└─────────────────────────┼─────────────────────────────────┘
                          │
                          v
          ┌───────────────────────────┐
          │     Worker pool           │
          │                           │
          │  worker 0: process_many() │
          │  worker 1: process_many() │
          │  worker 2: process_many() │
          │  ...                      │
          └─────────────┬─────────────┘
                        │  results (may arrive out of order)
                        v
              Reorder by seq# (deterministic mode)
                        │
                        v
                   Downstream
```

### Why accumulators matter for determinism

As explained in [Determinism](determinism.md), Zephon guarantees that a
pipeline produces the exact same output regardless of how many workers
execute an operator.  Accumulators are the mechanism that makes this
possible.

Consider sequence packing.  The packer maintains open bins, partially
filled sequences waiting for more short samples to complete them.  If each
worker maintained its own bins, bin assignment would depend on which worker
receives which sample, i.e., on scheduling timing.  Running with 1
worker versus 4 would produce different packed sequences.  Instead, all bin
state lives in the packing accumulator, which runs serially: it receives
elements in stream order and assigns them to bins on a single thread, so the
stateful decision of which sample goes where is always deterministic.
(For a concrete walkthrough, see
[Example: deterministic packing](#example-deterministic-packing) below.)

The same principle applies to every operator that needs cross-invocation
state: batching, mixture correction, shuffle buffering.  By confining all
mutable state to the single-threaded accumulator, micro-batch boundaries are
identical whether the operator runs with 1 worker or 16.

### The accumulator interface

Different operators need different batching strategies.  A tokenizer wants
fixed-size groups of samples for efficient batch encoding.  A packer needs
to wait until bins are full.  A batch operator needs to collect exactly
`microbatch_size` samples per lane.  The accumulator abstraction captures
this diversity through a simple interface:

- **`push_many(elems)`**: consume new elements from upstream.  Buffer
  them internally and return zero or more ready micro-batches.
- **`flush()`**: called when upstream closes.  Return any remaining
  partial micro-batches.
- **`has_pending_data()`**: returns whether the accumulator has buffered
  data that would be emitted on `flush()`.  The runner uses this to avoid
  premature termination.

`flush()` is also deterministic: it is called exactly once when the stream
ends, and because the accumulator's internal state is fully determined by
the elements it has seen, the flushed micro-batch is reproducible.

This interface is all a runner needs.  The runner calls `push_many` each
time new elements arrive, dispatches any returned micro-batches to workers,
and calls `flush` when the stream ends.

### Built-in accumulators

Zephon provides several accumulator implementations.  Most operators use one
of the first two; specialized operators bring their own.

#### PassthroughAccumulator

Emits whatever comes in, immediately.  Every `push_many` call produces
exactly one micro-batch containing the input elements.  No buffering, no
state.

Used by operators that are pure pass-throughs or need no grouping, such as
the internal {py:class}`~zephon.ops.ReplayFilter`.

#### CountingAccumulator

Buffers incoming elements and emits a micro-batch each time the buffer
reaches a configured `max_batch` size.  On `flush`, it emits any remaining
elements.

This is the most common accumulator.  It is used by
{py:class}`~zephon.ops.FetchOp`,
{py:class}`~zephon.ops.TokenizeText`,
{py:class}`~zephon.ops.MapTransform`,
{py:class}`~zephon.ops.ShuffleBuffer`, and several other operators.  The
batch size is tuned per operator: `FetchOp` uses 64, `DecodeText` uses 128,
and `ShuffleBuffer` uses the full `buffer_size` so the entire buffer is
shuffled as one micro-batch.

In non-deterministic mode, the `CountingAccumulator` can optionally flush
on a **time limit** (`max_latency_ms`): if the buffer has been non-empty
for longer than the limit, it emits a partial micro-batch rather than
waiting.  This reduces latency for bursty workloads.  In deterministic
mode, the time-based flush is disabled so that micro-batch boundaries depend
only on element counts, never on wall-clock timing.

#### BatchAccumulator

Used by the {py:class}`~zephon.ops.Batch` operator.  Maintains per-lane
buffers and emits a micro-batch when a lane's buffer reaches
`microbatch_size` records.

#### PackingAccumulator

Used by {py:class}`~zephon.ops.PackSequences`.  Maintains per-lane bins for
variable-length sequence packing.  A micro-batch is emitted when a bin's
remaining capacity falls below `min_sequence_length` (the bin is "full
enough") or when the number of open bins exceeds the limit.  Supports
first-fit and best-fit algorithms.

#### EnsureMixtureAccumulator

Used by {py:class}`~zephon.ops.EnsureMixture`.  Maintains per-lane buffers
grouped by mixture component and uses Smooth Weighted Round Robin
({py:class}`~zephon.utils.swrr.SmoothWeightedRoundRobin`) to emit
samples in the order that best tracks the target mixture ratios.
Buffering is adaptive: samples are emitted immediately when the desired
component is available, and buffered otherwise.

---

## How runners enforce ordering

As described in [Basic Concepts: execution](../basic_concepts.md#execution-stages-and-runners),
a **runner** is the execution backend that manages worker threads or
processes for an operator stage.  Zephon ships with three runner types
(inline, thread, process), but from the accumulator's perspective they all
work the same way.

When an operator runs with parallelism > 1, multiple workers execute
micro-batches concurrently.  Workers may finish in any order: a slow
micro-batch might complete after a fast one that was dispatched later.  But
the downstream operator expects results in the original input order.

In **deterministic mode** (the default), the runner tags each micro-batch
with a monotonically increasing **sequence number** before dispatching it
to a worker.  When results come back, the runner holds out-of-order
completions in a buffer and only forwards them downstream once all
preceding sequence numbers have been emitted.  The effect is that the
output stream is identical to a single-threaded execution, regardless of
how many workers run in parallel or how their runtimes vary.

In **non-deterministic mode**, the runner skips this reordering and forwards
results in completion order instead, trading reproducibility for lower
latency.

The combination of accumulator-defined micro-batch boundaries and
sequence-number reordering is what makes execution parameters
(parallelism, runner type, queue depths) pure performance knobs
that never change the data order.  See
[Determinism: execution parameters do not change data order](determinism.md#execution-parameters-do-not-change-data-order)
for more on this guarantee.

---

## Example: deterministic packing

Sequence packing illustrates the accumulator pattern well.  The
{py:class}`~zephon.ops.PackSequences` operator packs variable-length
tokenized sequences into fixed-length bins to maximize GPU utilization.

The packer must decide which incoming sequence goes into which bin.  This
decision depends on what is already in the bins, making it inherently
stateful.  If multiple workers each maintained their own bins, bin
assignment would depend on which worker sees which element first, that
is, on scheduling timing.

Instead, all bin state lives in the `PackingAccumulator`, which runs on the
accumulator, which runs serially.  It receives elements in stream order and
runs first-fit or best-fit bin assignment without any parallelism.  In the
current implementation, the accumulator also assembles the packed payloads
and emits finished records, making `process_many` a passthrough.  The key
requirement is that the bin assignment, the stateful decision of which
sample goes into which bin, must run serially in the accumulator to remain
deterministic.

```
PackingAccumulator (runs serially):

  incoming seq (len=300) → assign to bin 2
  incoming seq (len=500) → assign to bin 0
  incoming seq (len=700) → bin 0 is now full → emit bin 0 as micro-batch
  incoming seq (len=200) → assign to bin 2
  incoming seq (len=500) → bin 2 is now full → emit bin 2 as micro-batch
  ...
```

The bins are also per-lane, so samples from different lanes are never packed
together.  This is essential for
[elastic determinism](determinism.md#lanes-zephons-approach): each lane's
packing decisions are independent of how many GPUs are in use.

---

## Example: data-derived seeds in the shuffle buffer

The {py:class}`~zephon.ops.ShuffleBuffer` needs randomness, but using a
plain incrementing RNG would tie the permutation to micro-batch boundaries.
Since those boundaries can shift across restarts (because of checkpoint
replay boundaries), this would break determinism.

Instead, the shuffle buffer derives its seed from the **data itself**: it
hashes the cursor keys of all elements in the micro-batch, specifically
`(chunk_id, chunk_offset, lineage, sample_id)` where
`sample_id = (dataset_id, shard_id, local_sample_id)`, together with a
user-provided base seed.  The same set of elements always produces the same
permutation, regardless of how they were grouped or on which restart they
appear.

The shuffle buffer uses a `CountingAccumulator` with `max_batch` set to the
full `buffer_size`, ensuring deterministic micro-batch boundaries.  The
data-derived seed then makes the permutation within each micro-batch
reproducible.

## User-defined operators

Zephon supports user-defined transformations through three Pipeline methods:

| Pipeline method | Operator | What it does |
|---|---|---|
| `.map_transform(fn)` | {py:class}`~zephon.ops.MapTransform` | Apply a function to each sample's payload |
| `.map_batch(fn)` | {py:class}`~zephon.ops.MapBatchTransform` | Apply a function to each training batch (after `.batch()`) |
| `.stateful_transform(...)` | {py:class}`~zephon.ops.StatefulTransformOp` | User-managed state in the accumulator |

{py:meth}`~zephon.api.Pipeline.map_transform` is the primary UDF
mechanism.  Your function receives a sample's payload dict and returns a
(possibly modified) dict, or `None` to filter the sample out:

```python
def add_length_field(payload: dict) -> dict:
    payload["text_length"] = len(payload["text"])
    return payload

pipeline = (
    Pipeline(ws)
    .tokenize(tokenizer_id="gpt2", max_length=2048)
    .map_transform(add_length_field, parallelism=4)
    .batch(microbatch_size=8)
)
```

The function runs inside `process_many` on worker threads (or processes),
so it should be stateless and thread-safe.  The `MapTransform` operator
wraps it with a `CountingAccumulator` for micro-batching.

{py:meth}`~zephon.api.Pipeline.map_batch` is similar but operates on
`SampleBatch` objects after the batch operator.  This is useful for
post-batch processing like tensor collation or padding.

{py:meth}`~zephon.api.Pipeline.stateful_transform` is the most flexible
option.  It lets you provide custom state management functions that run
serially in the accumulator, with an optional parallel
`transform_fn` for the workers.  The key callbacks are:

- **`init_state()`**: returns the initial state object (called lazily on
  first data).
- **`push(state, items)`**: called serially for each incoming
  micro-batch.  Returns `(new_state, outputs)`.
- **`flush(state)`**: called on stream end to emit any buffered items.
- **`should_flush(state)`**: optional early-flush predicate.

```python
pipeline = (
    Pipeline(ws)
    .tokenize(tokenizer_id="gpt2", max_length=2048)
    .stateful_transform(
        "collect_pairs",
        init_state=lambda: [],
        push=lambda state, items: (
            ([], state + items) if len(state) + len(items) >= 2
            else (state + items, [])
        ),
        flush=lambda state: state,  # emit any leftover
    )
    .batch(microbatch_size=8)
)
```

See the {py:meth}`~zephon.api.Pipeline.stateful_transform` API reference
for the full set of parameters.

```{note}
There is currently no public API for defining fully custom operators with
custom accumulators.  The `Op` protocol and accumulator interfaces exist
internally and are used by all built-in operators, but they are not yet
stabilized for external use.  If the built-in operators and UDF mechanisms
do not cover your use case, please open an issue; we are evaluating how
to best expose this extension point.
```