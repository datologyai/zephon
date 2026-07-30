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
{py:class}`~zephon.Pipeline` builder to describe *how* your data is
processed:

```python
pipeline = (
    Pipeline(ws)
    .tokenize(tokenizer_id="gpt2", field="text", max_length=2048)
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
- **`process_many(elems)`**: the workhorse.  Takes a list of stream
  elements (a micro-batch) — {py:class}`~zephon.types.SampleRecord`
  instances, or {py:class}`~zephon.types.SampleBatch` objects
  downstream of `.batch()` — transforms them, and returns a list of the **same element
  types**.  To transform a sample, rebuild its `.payload` while preserving
  its `.meta` (which carries the sample's identity, lineage, and ordering
  cursor); never return a bare `dict`, list, or raw payload.  The engine
  rejects any other output element type with
  `TypeError: Unsupported element type`.  This is the function that runs in
  parallel across workers.
- **`traits()`**: declares properties that guide compilation and
  scheduling, such as the suggested parallelism and whether the operator
  preserves sample ordering.

Two additional methods complete the contract:

- **`setup(ctx)`**: called once before the operator processes any data. The
  runner passes an `OpContext` with runtime services and the operator's plan
  placement in `ctx.stage_info`.
- **`process_one(elem)`**: transforms a single element. The default
  implementation calls `process_many` with a single-element list, so operators
  should override `process_one` only when they need a distinct scalar fast
  path.

The critical invariant is that **`process_many` must be stateless across
calls**.  It must not rely on anything from a previous invocation.  Any
cross-invocation state (buffers, counters, open bins) belongs in the
accumulator, not in `process_many`.  This invariant is what makes it safe to
dispatch `process_many` calls to an arbitrary number of workers in parallel:
each call is independent, and results can be reordered back into the
original input order without loss of correctness.

Not every operator splits its work evenly between accumulator and workers.
For operators like ``TokenizeText`` or
``FetchOp``, the accumulator merely groups elements
into fixed-size micro-batches and the real work (tokenization, disk I/O)
happens in `process_many` across parallel workers.  But for operators whose
core logic is inherently stateful and serial, such as
``PackSequences``,
``EnsureMixture``, and
``ShuffleBuffer``, the accumulator does the heavy
lifting.  In the current implementation their `process_many` is a trivial
passthrough (`return list(elems)`), though in principle some of their work
(e.g., payload assembly for packing) is stateless and could be moved to
`process_many` to run on parallel workers.  The key architectural
requirement is that any logic that needs to see elements in stream order
*must* live in the accumulator.

```{note}
Known limitation: shuffling and packing currently run entirely in the
accumulator (pump thread).  Moving the heavy work into parallel workers
would improve throughput but requires careful state splitting.
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
of the {py:meth}`~zephon.Pipeline.batch` operator) that the model trains
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
- **`flush(*, reset=False, lane_id=None)`**: emit remaining buffered data.
  Called in two contexts:
  - **End of stream** (`reset=False`, `lane_id=None`, the default): every
    lane is done.  Emit any remaining partial micro-batches.
  - **Mid-stream** (`reset=True`, `lane_id` set): a flush sentinel has
    arrived (see [Flush sentinels](checkpointing.md#flush-sentinels-and-epoch-boundaries)).
    Flush sentinels are injected **per lane**, so the flush is scoped to
    `lane_id`: emit and **fully reset** only that lane's state, leaving the
    other lanes untouched.  This is what makes epoch boundaries safe for
    replay — each lane's next epoch starts from clean state; flushing every
    lane at one lane's boundary corrupts the others and breaks replay.
    The only built-in operator that intentionally stalls instead of
    flushing is ``Batch`` with `drop_last=True`.
    Other operators must flush to a fresh state at the sentinel.  See
    [Flush sentinels and accumulator stalling](#flush-sentinels-and-accumulator-stalling)
    below.
- **`has_pending_data(lane_id=None)`**: returns whether the accumulator has
  buffered data that would be emitted on `flush()`, scoped to `lane_id` when
  set.  The runner uses this to avoid premature termination and to detect
  [stalling](#flush-sentinels-and-accumulator-stalling).

`flush()` is deterministic: its output depends only on the elements the
accumulator has seen, so the flushed micro-batch is reproducible.

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
the internal ``ReplayFilter``.

#### CountingAccumulator

Buffers incoming elements and emits a micro-batch each time the buffer
reaches a configured `max_batch` size.  On `flush`, it emits any remaining
elements.

This is the most common accumulator.  It is used by
``FetchOp``,
``TokenizeText``,
``MapTransform``,
``ShuffleBuffer``, and several other operators.  The
batch size is tuned per operator: `FetchOp` uses 64, `DecodeText` uses 128,
and `ShuffleBuffer` uses the full `buffer_size` so the entire buffer is
shuffled as one micro-batch.

In non-deterministic mode, the `CountingAccumulator` can optionally flush
on a **time limit** (`max_latency_ms`): if the buffer has been non-empty
for longer than the limit, it emits a partial micro-batch rather than
waiting.  This reduces latency for bursty workloads.  In deterministic
mode, the time-based flush is disabled so that micro-batch boundaries depend
only on element counts, never on wall-clock timing.

#### Batch (via CountingAccumulator)

The ``Batch`` operator uses a
{py:class}`~zephon.ops.CountingAccumulator` with
``key_fn=lane_of`` and ``drop_last`` support.  Maintains per-lane
buffers and emits a micro-batch when a lane's buffer reaches
``microbatch_size`` records.

#### PackingAccumulator

Used by ``PackSequences``.  Maintains per-lane bins for
variable-length sequence packing.  A micro-batch is emitted when a bin's
remaining capacity falls below `min_sequence_length` (the bin is "full
enough") or when the number of open bins exceeds the limit.  Supports
first-fit and best-fit algorithms.

#### EnsureMixtureAccumulator

Used by ``EnsureMixture``.  Maintains per-lane buffers
grouped by mixture component and uses Smooth Weighted Round Robin
(``SmoothWeightedRoundRobin``) to emit
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
``PackSequences`` operator packs variable-length
tokenized sequences into bins.

The packer must decide which incoming sequence goes into which bin.  This
decision depends on what is already in the bins, making it inherently
stateful.  If multiple workers each maintained their own bins, bin
assignment would depend on which worker sees which element first, that
is, on scheduling timing.

Instead, all bin state lives in the `PackingAccumulator`, which runs
serially on the pump thread.  It receives elements in stream order and
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

Note that bin state is **history-dependent**: which bins exist and their
remaining capacity depends on the full sequence of inputs the packer has
seen, not just the samples currently in the bins.  This has important
implications for checkpointing — evicting a chunk whose samples shaped
the current bin state would break replay.  See
[Epoch-Based Eviction](checkpointing.md#epoch-based-eviction) for how
flush sentinels solve this by periodically resetting the accumulator.

---

## Example: data-derived seeds in the shuffle buffer

The ``ShuffleBuffer`` needs randomness, but using a
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

## Flush sentinels and accumulator stalling

Non-monotonic pipelines (those with packing, shuffling, or mixture
correction) use **flush sentinels** to create epoch boundaries for safe
checkpoint eviction.  Every `flush_every_k_chunks` chunks per lane, the
engine injects a sentinel that triggers `flush(reset=True, lane_id=lane)` on
each history-dependent accumulator, scoped to that sentinel's lane.  For the
full motivation and eviction model, see
[Epoch-Based Eviction](checkpointing.md#epoch-based-eviction).

This section covers the accumulator side: which accumulators are flushed,
what "fully reset" means, and the stalling mechanism.

### Which accumulators are flushed

The runner calls `flush(reset=True, lane_id=lane)` on **every** accumulator
when a flush sentinel arrives — scoped to the sentinel's lane — unless the
operator is `Batch(drop_last=True)`
(see [below](#stalling-at-epoch-boundaries)).
For history-dependent accumulators (`preserves_cursor_order=False`) this
flush is critical — it resets state so the next epoch can replay
independently.  For order-preserving accumulators, the flush is harmless
(a no-op or an empty drain) and only the epoch-floor bookkeeping differs.

| Accumulator | Behavior at sentinel | Why |
|---|---|---|
| `PackingAccumulator` | Flushed (emits padded partial bins, resets all bin state) | Bin state is history-dependent |
| `EnsureMixtureAccumulator` | Flushed | SWRR deficit state is history-dependent and must reset at the epoch boundary for replay after eviction |
| `ShuffleBuffer` (via `CountingAccumulator`) | Flushed (drains buffer) | Buffer contents span chunk boundaries |
| `CountingAccumulator` | Flushed (no-op: stateless grouping) | Flush is harmless; returns whatever is buffered |
| `BatchAccumulator` (`drop_last=False`) | Flushed (emits partial batch) | Flush is harmless; partial batch is acceptable |
| `BatchAccumulator` (`drop_last=True`) | Stalled | Cannot emit partial batches; this is the only built-in stalling special case |
| `PassthroughAccumulator` | Flushed (no-op: no state) | `flush()` returns `[]` |

### What "fully reset" means

After `flush(reset=True, lane_id=lane)`, the flushed lane must be
indistinguishable from a freshly constructed instance.  That lane's internal
buffers, counters, and derived state must be cleared, while other lanes are
left untouched.  For the packing accumulator, this means the lane's bins are
emptied and its partially filled bins are emitted with padding.  For the
mixture corrector, the lane's SWRR deficit weights are reset.  This guarantee
is what allows each lane's next epoch to replay independently from a clean
state.

### Stalling at epoch boundaries

Stalling is intentionally narrow in the current design.  Zephon only
supports it for ``Batch`` with `drop_last=True`.

Why the restriction exists:

- For a general stalled operator, the effective reset point is later than
  the stored epoch boundary.
- After eviction, replay would need the cross-boundary consumption state
  that determined that delayed reset point.
- Zephon checkpoints inflight chunks, epoch boundaries, and replay
  cursors, but it does **not** checkpoint a general operator replay
  capsule.

So a non-Batch operator that wants to stall would need additional
replay-specific state to make restore exact.  Stalling is therefore not
an operator trait: the runner derives it from the op instance, for
`Batch(drop_last=True)` specifically.

`Batch(drop_last=True)` is the special case that still works:

- it preserves input order
- it only emits complete batches
- the planner inserts `ReplayFilter` immediately before `Batch`
- the consumer-visible checkpoint cut is therefore between complete
  batches, so the live Batch buffer is empty at the replay cursor

Live execution can still produce a cross-epoch batch such as
`[old, old, new]` before the sentinel is released.  That is expected.
What matters is that on resume the pre-cursor prefix is filtered *before*
Batch, and Batch rebuilds the suffix from an empty buffer at the same
batch boundary.

When the runner sees a flush sentinel for a supported stalling accumulator
with pending data on the sentinel's lane, it skips the mid-stream flush and
holds the sentinel behind the buffered data.  The sentinel is released once
that lane's carry has drained enough that `try_epoch_reset(boundary, lane)`
succeeds; lanes stall independently, so a blocked lane does not hold back
another lane's ready sentinel.

Operators that do **not** opt into the supported Batch stalling path must
fully flush at the sentinel.  If `flush(reset=True, lane_id=lane)` returns
while `has_pending_data(lane)` is still `True`, the flush contract has been
violated and the runner raises.  Leaving residual cross-boundary state
would break the checkpoint/replay contract.

---

## User-defined operators

Zephon supports user-defined transformations through three Pipeline methods:

| Pipeline method | Operator | What it does |
|---|---|---|
| `.map_transform(fn)` | ``MapTransform`` | Apply a function to each sample's payload |
| `.map_batch(fn)` | ``MapBatchTransform`` | Apply a function to each training batch (after `.batch()`) |
| `.stateful_transform(...)` | ``StatefulTransformOp`` | User-managed state in the accumulator |

{py:meth}`~zephon.Pipeline.map_transform` is the primary UDF
mechanism.  Your function receives a sample's payload dict and returns a
(possibly modified) dict, or `None` to filter the sample out:

```python
def add_length_field(payload: dict) -> dict:
    payload["text_length"] = len(payload["text"])
    return payload

pipeline = (
    Pipeline(ws)
    .tokenize(tokenizer_id="gpt2", field="text", max_length=2048)
    .map_transform(add_length_field, parallelism=4)
    .batch(microbatch_size=8)
)
```

The function runs inside `process_many` on worker threads (or processes),
so it should be stateless and thread-safe.  The `MapTransform` operator
wraps it with a `CountingAccumulator` for micro-batching.

{py:meth}`~zephon.Pipeline.map_batch` is similar but operates on
`SampleBatch` objects after the batch operator.  This is useful for
post-batch processing like tensor collation or padding.

{py:meth}`~zephon.Pipeline.stateful_transform` is the most flexible
option.  It lets you provide custom state management functions that run
serially in the accumulator, with an optional parallel
`transform_fn` for the workers.  State is partitioned **per lane** — each
lane gets its own state instance and your callbacks only ever see one lane at
a time — so a mid-stream flush resets just that lane's epoch, keeping replay
deterministic when one engine owns several lanes.  The key callbacks are:

- **`init_state()`**: returns the initial state object (called lazily on
  each lane's first record).
- **`push(state, items)`**: called serially with one lane's records at a
  time.  Returns `(new_state, outputs)`.
- **`flush(state)`**: emit a lane's remaining buffered items.  Called at
  end-of-stream and, in non-monotonic pipelines, mid-stream at each lane's
  epoch boundary; that lane's state is re-initialized afterward via
  `init_state()`.
- **`should_flush(state)`**: optional per-lane early-flush predicate.

```python
pipeline = (
    Pipeline(ws)
    .tokenize(tokenizer_id="gpt2", field="text", max_length=2048)
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

See the {py:meth}`~zephon.Pipeline.stateful_transform` API reference
for the full set of parameters.

## Fully custom operators

When `map_transform` / `map_batch` / `stateful_transform` aren't enough —
for instance, you need a custom accumulator paired with a custom
worker-side `process_many`, parallel workers over a non-default batching
discipline, or full control over operator traits — use
{py:meth}`~zephon.Pipeline.add_op`.  It accepts two forms.

### Instance form: `add_op(op)` with a `BaseOp` subclass

Subclass {py:class}`~zephon.ops.BaseOp` when your op needs the full
lifecycle: configuration in `__init__`, per-worker resource construction
in `setup` (with access to the runner's
{py:class}`~zephon.ops.OpContext`), per-op traits via `traits`, and a
custom accumulator.  The framework deep-copies the instance per parallel
worker and runs `setup` on each copy, so `self.*` attributes are
isolated per worker on every runner — the same lifecycle built-in
operators use.

```python
from zephon.ops import BaseOp, CountingAccumulator, OpContext, OpTraits

class Tokenize(BaseOp):
    def __init__(self, tokenizer_name: str, max_batch: int = 64):
        super().__init__()
        self._tokenizer_name = tokenizer_name  # picklable config
        self._max_batch = max_batch
        self._tokenizer = None                 # built per-worker in setup()

    def traits(self) -> OpTraits:
        return OpTraits(preserves_cursor_order=True, parallelism=4)

    def setup(self, ctx: OpContext):
        super().setup(ctx)
        # Per-worker init: build heavy / non-picklable resources here (the repo
        # convention — keeps __init__ picklable for the process and Ray
        # runners), and read services from ctx.
        self._tokenizer = load_tokenizer(self._tokenizer_name)
        self._metrics_cb = ctx.get("emit_my_metrics")

    def accumulator(self, *, deterministic, ctx):
        return CountingAccumulator(max_batch=self._max_batch)

    def process_many(self, elems):
        # Records in, records out: rebuild each payload, preserve each .meta.
        for e in elems:
            e.payload = {**e.payload, "ids": self._tokenizer.encode(e.payload["text"])}
        return elems

pipeline.add_op(Tokenize("gpt2", max_batch=64))
```

Traits come from `op.traits()`; pass them via your
{py:class}`~zephon.ops.OpTraits` rather than as kwargs on `add_op`.
The optional `name` kwarg controls the node name in plan graphs and
metrics (defaults to the class name); `placement` works the same way
as in the kwargs form.

### Kwargs form: `add_op(name, *, process_many=…, …)`

Convenience for stateless transforms whose only per-worker dependency
is picklable config captured in a closure.  The framework builds an
internal `BaseOp` subclass from the kwargs.

```python
from zephon.ops import CountingAccumulator

eos = " <eos>"  # picklable config captured in the closure

def append_eos(elems):
    # Records in, records out: rebuild each payload, preserve each .meta.
    for e in elems:
        e.payload = {**e.payload, "text": e.payload["text"] + eos}
    return elems

pipeline.add_op(
    "append_eos",
    process_many=append_eos,
    accumulator=lambda: CountingAccumulator(max_batch=64),
    parallelism=4,
    preserves_cursor_order=True,
)
```

`name`, `process_many`, and `preserves_cursor_order` are required.
Pass `preserves_cursor_order=True` for 1:1 maps, payload transforms,
and non-reordering filters; pass `False` when the op reorders,
shuffles, or packs — the planner uses this to pick an eviction
strategy, and getting it wrong corrupts checkpoint semantics silently.

The accumulator factory defaults to a `PassthroughAccumulator`, in
which case each upstream micro-batch is forwarded as one ready batch.
The remaining kwargs (`parallelism`, `placement`, `indexable`,
`batch_shape_sensitive`, `requires_serial_state`) map 1:1 to
{py:class}`~zephon.ops.OpTraits` and follow the semantics described in
[The operator contract](#the-operator-contract) above.  See the
{py:meth}`~zephon.Pipeline.add_op` API reference for the full kwarg
list and signature variants of the `accumulator` factory.

Two invariants the kwargs callables must honor — the same ones the
built-in operators honor:

- **`process_many` must stay stateless.**  All cross-invocation state
  belongs in the accumulator returned by your `accumulator` factory,
  never captured in a closure that mutates between calls.  This is what
  lets the runtime fan `process_many` out across parallel workers
  deterministically (see [The operator contract](#the-operator-contract)).
- **Read-only resources go in closures.**  A tokenizer object, a codec
  table, a threshold — capture them in the closure around your
  `process_many` callable.  They will be deep-copied per parallel worker
  (and cloudpickled across the process/Ray boundary), so they must be
  picklable on those runners.  Heavy or non-picklable per-worker
  construction (model handles, open file descriptors) is what the
  instance form's `setup` hook is for — reach for the class-based path
  when closures aren't enough.

{py:class}`~zephon.ops.BaseOp`, {py:class}`~zephon.ops.StageInfo`,
{py:class}`~zephon.ops.OpContext`, {py:class}`~zephon.ops.OpTraits`,
{py:class}`~zephon.ops.CountingAccumulator`, and
{py:class}`~zephon.ops.PassthroughAccumulator` are all exported from
{py:mod}`zephon.ops`.

### Helping the validator: `validation_samples`

When `Pipeline.__iter__` runs the validation harness (controlled by
`RuntimeOptions.auto_validation`, default `"strict"`), it probes each
user op with synthetic {py:class}`~zephon.types.SampleRecord` instances
whose payload is a generic `{'text': str, 'value': int}` dict.  The
harness never invokes `setup()`, so runtime probes (determinism,
cross-call state, statelessness, sample identity) run against an
un-setup'd op.  Two degradation paths fall out of that constraint, one
per attachment shape.

**Kwargs form.**  The callable is wrapped in a stateless internal op,
so the runtime probes always run.  When the callable needs payload
fields the synthetic records don't carry, it typically raises
`KeyError`, `AttributeError`, or `TypeError`; the validator catches
those and surfaces `OP_REJECTS_GENERIC_PAYLOAD` — a *warning* that
skips the runtime checks for that op.  The pipeline still runs.

**Instance form.**  The canonical `BaseOp` pattern stores heavy
resources (tokenizers, model handles) as `None` in `__init__` and
populates them in `setup()`, so `process_many` against an un-setup'd
op would raise spuriously.  A defensive
`raise RuntimeError("setup() not called")` would also escalate to
`OP_RAISES_ON_SYNTHETIC_INPUT` (severity *error*) and block iteration
under `auto_validation="strict"`.  The validator avoids both by
gating the runtime checks on `validation_samples()` being overridden:
when it isn't, the validator emits a single
`OP_INSTANCE_RUNTIME_CHECKS_SKIPPED` *warning* and skips the runtime
probes.  Static AST checks (self-writes, non-deterministic stdlib
calls) still run.

To opt an op into the full runtime check suite, supply records the op
can consume *without* `setup` having run — pre-tokenize / pre-encode
payloads rather than relying on resources the real `setup` would
build:

```python
def custom_records():
    return [
        SampleRecord(
            meta=SampleMeta(sample_id=(lane, i), lane_id=lane,
                            chunk_id=i // 2, chunk_offset=i),
            payload={"tokens": [1, 2, 3]},
        )
        for lane in (0, 1)
        for i in range(3)
    ]

# Kwargs form — pass the factory as a kwarg on add_op.
pipeline.add_op(
    "needs_tokens",
    process_many=tokenize_step,
    preserves_cursor_order=True,
    validation_samples=custom_records,
)

# Instance form — override the method on your BaseOp subclass.
class MyOp(BaseOp):
    def validation_samples(self):
        return custom_records()
    ...
```
