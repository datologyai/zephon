# Sample Lifecycle

This page explains how a sample travels through a Zephon pipeline, from a
lightweight pointer produced by the WorkSource, through operators that
transform, split, filter, and pack data, to the training batch your model
consumes.  Part I builds the mental model: identities, fan-out, fan-in,
and the bookkeeping that makes checkpointing work.  Part II is aimed at
developers writing custom operators and covers the concrete APIs, contracts,
and pitfalls.

For the surrounding context (what a WorkSource is, how stages and
runners work, what the Engine does) see [Basic Concepts](../basic_concepts.md).
For how checkpoints capture and restore pipeline state, see
[Checkpointing](checkpointing.md).

---

## Part I: How Samples Flow

### From Pointer to Data

As described in [Basic Concepts](../basic_concepts.md#from-pointers-to-data),
a sample starts as a pointer, i.e., a `(dataset_id, shard_id, sample_idx)`
triple produced by the WorkSource.  The built-in
``FetchOp`` loads the actual data from disk/cache
and wraps it in a {py:class}`~zephon.types.SampleRecord`, which is a container that
pairs a **payload** (typically a Python dict of fields) with internal
tracking **metadata** ({py:class}`~zephon.types.SampleMeta`).

Everything before FetchOp operates on pointers; everything after it operates
on data.  The rest of this page is about what happens after that transition.

### The Identity of a Record

Every record flowing through the pipeline carries a
{py:class}`~zephon.types.SampleCursor`, a frozen tuple of four
components that uniquely identifies it within its lane:

```
SampleCursor sort key = (chunk_id, chunk_offset, lineage, sample_id)
                           │           │            │         │
                           │           │            │         └─ (dataset, shard, local_id)
                           │           │            └─ path through fan-out operators
                           │           └─ position within the chunk
                           └─ which chunk this sample came from
```

Note: the dataclass constructor order is `(chunk_id, chunk_offset, sample_id,
lineage)` — the sort key swaps `lineage` before `sample_id` so children of the
same offset sort together.

You can think of the cursor as a passport: it says where the record was born
(`chunk_id`, `chunk_offset`, `sample_id`) and what happened to it along the
way (`lineage`).  The cursor is **deterministic**: given the same
pipeline configuration and inputs, the same record always gets the same
cursor.

Cursors serve two purposes:

1. **Replay identity.**  When resuming from a checkpoint, the
   [ReplayFilter](checkpointing.md#the-replayfilter) uses cursor equality to
   find the boundary between already-consumed records and new ones.
2. **Eviction accounting.**  The engine uses the `(chunk_id, chunk_offset)`
   part of the cursor (the **base offset**) to track whether a chunk's
   data has been fully consumed and can be freed from memory.
   See [Chunk Eviction](checkpointing.md#chunk-eviction) for how this works.

### Operators: Map, FlatMap, and Filter

If you come from a functional-programming background, the operator types in
Zephon map directly onto familiar abstractions:

| FP concept | What it does | Zephon example |
|---|---|---|
| **map** | 1 input &rarr; 1 output, always | Tokenization (no splitting), image decode |
| **flatMap** | 1 input &rarr; 0 or more outputs | Tokenization with `split_long_samples=True` |
| **filter** | 1 input &rarr; 0 or 1 outputs | Dedup, length gating |
| **fold / reduce** | N inputs &rarr; 1 output | Sequence packing, batching |

Most Zephon operators are conceptually a **map**: they take one record in
and produce one record out, with the same identity metadata.  The important
nuance is that Zephon's `map_transform` API can also act as a filter.
When the transform UDF returns `None`, the record is dropped.
Under the hood, this is closer to a `filterMap` or `flatMap` that returns
zero or one elements:

```python
pipeline.map_transform(
    lambda payload: payload if len(payload["input_ids"]) > 128 else None,
    drop_none=True,   # enables filter behavior
)
```

When `drop_none=True` (the default), dropping a record is safe: the
framework automatically emits the bookkeeping signals (tombstones, described
[below](#tombstones-closing-what-you-dropped)) needed to keep eviction
correct. 

True fan-out operators (1 &rarr; N), like tokenization that splits a long
document into multiple training sequences, are the most interesting
case, because they create **new identities**.  The next section explains how.

### Fan-Out: One Sample, Many Records

When an operator produces multiple output records from a single input, each
output needs its own unique cursor.  Zephon achieves this through
**lineage**: a path of integer indices appended to the parent's cursor.

Consider a tokenizer that splits a long document into three sequences:

```
Input (from chunk 7, offset 2):
  cursor = (cid=7, off=2, lineage=(), sample_id=(ds0, sh5, 42))

                    tokenize (split_long_samples=True)
                    ┌──────────┼──────────┐
                    v          v          v

  child 0:  (7, 2, (0,), (ds0,sh5,42))
  child 1:  (7, 2, (1,), (ds0,sh5,42))
  child 2:  (7, 2, (2,), (ds0,sh5,42))
```

All three children share the same **base offset** `(chunk_id=7,
chunk_offset=2)` (they still "belong to" the same original sample)
but their lineage paths `(0,)`, `(1,)`, `(2,)` make each cursor unique.

Lineage can nest.  If a later operator further splits child 1, the
grandchild's lineage would be `(1, 0)`, `(1, 1)`, etc.  The tree structure
is always deterministic because child indices follow the emission order.

The key invariant for fan-out is:

> For each base offset in a lane, **exactly one** output record
> (or tombstone) must eventually signal "I am the last one."

This signal is the `is_last_child` flag on a
{py:class}`~zephon.types.ContributorRef`.  It tells the engine that
no more records will arrive for that base offset, so the corresponding chunk
slot can be marked as closed.  The helpers described in
[Part II](#the-helper-toolkit) handle this bookkeeping.

### Fan-In: Many Records, One Output

The opposite direction is equally important.  Two operators perform fan-in:

**Packing** ({py:meth}`~zephon.Pipeline.pack_sequences`) combines
fragments from *different* base offsets into a single packed record.  A
packed record gets a fresh cursor (its **primary cursor**, typically derived
from the first contributor) but carries a list of
{py:class}`~zephon.types.ContributorRef` entries that remember
which base offsets went into it.  This is what lets the engine track
per-offset completion even when offsets are mixed across records.

```
child (7,2,(0,))  ─┐
                    ├─ pack ──> packed record
child (7,3,(0,))  ─┘           cursor = (7,2,(0,0),...)
                                contributors:
                                  (7,2,(0,)) is_last_child=True
                                  (7,3,(0,)) is_last_child=True
```

A single packed record can close multiple base offsets at once, even
offsets from different chunks.  The engine processes each contributor
independently.

Packing also tracks **component heritage**: how many samples from each
mixture component (dataset) ended up in the packed record, via
`component_sample_counts` and optionally `component_token_counts` on the
metadata.  This is what allows
{py:meth}`~zephon.Pipeline.ensure_mixture` to correctly measure and
correct mixture ratios even after packing reshuffles which samples end up
together.

**Batching** ({py:meth}`~zephon.Pipeline.batch`) collects individual
`SampleRecord` objects into a
{py:class}`~zephon.types.SampleBatch`.  Unlike packing, batching
does not merge payloads or create new cursors.  Each record inside the batch
retains its own cursor and metadata.  The batch is a grouping convenience
for the training loop.

### Contributors: Who Depends on What

Contributors are Zephon's accounting system for answering the question:
*"Has the pipeline finished with base offset `(chunk_id, chunk_offset)`?"*

Every record carries contributor information, either explicitly (set by
fan-out or packing helpers) or implicitly (the default: a single
contributor pointing at the record's own cursor with `is_last_child=True`).

The `is_last_child` flag is the critical piece.  It means:

> "No more records derived from this base offset will ever be emitted."

The engine watches for this flag.  Once every offset in a chunk has seen a
closing contributor, the chunk is fully consumed and can be
[evicted](checkpointing.md#chunk-eviction).

For a simple 1:1 operator that neither drops nor splits records, you never
need to think about contributors; the defaults are correct.  Contributors
become explicit when:

- A fan-out operator creates multiple children from one base offset
  (exactly one child must set `is_last_child=True`).
- A packing operator merges contributors from multiple base offsets into one
  record (it lists all of them, marking whichever ones close their offsets).
- A filter drops the last child for an offset (it must emit a tombstone
  instead).

### Tombstones: Closing What You Dropped

Consider a filter that decides to drop a sample entirely, say a
deduplication filter that sees a duplicate document.  If that sample was the
only (or last) output for its base offset, the engine will never see a
closing contributor for that offset.  Without intervention, the containing
chunk can never be evicted, and memory grows without bound.

A **tombstone** solves this.  It is a record with no meaningful payload
whose sole purpose is to carry a closing `ContributorRef` to the engine:

```
Base offset (10, 5) → filter decides to drop it

Without tombstone:              With tombstone:
  offset (10,5) never closes      filter emits tombstone for (10,5)
  chunk 10 stuck in memory        engine marks (10,5) closed
  ✗ memory leak                   chunk 10 can evict when all offsets close
                                  ✓ correct
```

Tombstones are tagged (`meta.tombstone == True`) and flow through the
pipeline like regular records, but the engine
[strips them](checkpointing.md#replay-on-resume) before they reach the
training loop.  They participate in notification and eviction but are
invisible to your model.

The safe default rule for operator authors: **always emit tombstones when
dropping records**, regardless of pipeline configuration.  Whether the
pipeline uses monotone or epoch-based eviction is decided at compile time
based on operator traits, and your operator cannot know which path will be
chosen.  Tombstones are harmless in the monotone path (notified then
discarded) and required in the epoch-based path.

### Chunks as the Unit of State

Tying the concepts together: **chunks** are the fundamental unit of
pipeline state.  The Engine keeps a small set of **inflight chunks** per
lane: chunks whose samples have entered the pipeline but whose outputs
have not all been delivered yet.

As contributors (and tombstones) signal offset completion, the engine
[evicts](checkpointing.md#chunk-eviction) fully-consumed chunks from the
inflight set.  For non-monotonic pipelines (packing, shuffling, mixture
correction), the engine also injects **flush sentinels** every
`flush_every_k_chunks` chunks to create epoch boundaries — points where
history-dependent accumulators reset to clean state so that earlier chunks
can be safely evicted.  See
[Epoch-Based Eviction](checkpointing.md#epoch-based-eviction-packing-and-shuffling) for the
full model.

At checkpoint time, only the remaining inflight chunks (plus epoch
boundary positions for non-monotonic pipelines) are saved.  On resume,
only those chunks are replayed through the pipeline.  This is what keeps
checkpoints small and resume fast, regardless of how far into training
you are.

The lifecycle, end to end:

```
WorkSource           Engine                       Operators                Training loop
    │                   │                             │                        │
    │  work chunk       │                             │                        │
    │ ─────────────────>│  assign chunk_id            │                        │
    │                   │  add to inflight set        │                        │
    │                   │                             │                        │
    │                   │  yield sample pointers      │                        │
    │                   │ ───────────────────────────>│                        │
    │                   │                             │  fetch, transform,     │
    │                   │                             │  split, pack, filter   │
    │                   │                             │                        │
    │                   │      notify(contributors)   │                        │
    │                   │ <───────────────────────────│                        │
    │                   │  update offset bitmaps      │                        │
    │                   │  evict complete chunks      │                        │
    │                   │                             │                        │
    │                   │                             │  yield records/batches │
    │                   │                             │ ──────────────────────>│
    │                   │                             │  (tombstones stripped) │
```

For the full details on how eviction and replay interact with
checkpointing, see [Checkpointing](checkpointing.md).

---

## Part II: Operator Contracts and API

This section is for developers writing custom operators or working on
Zephon internals. 

### The Helper Toolkit

Zephon provides four builder functions in `zephon.ops.children` that
handle the fiddly parts of metadata construction:

#### `spawn_child(parent, child_idx, *, is_last_child=False, tags=None)`

Creates metadata for a child record derived from a parent.  Use this
whenever a single input produces multiple outputs (any true fan-out):

```python
from zephon.ops.children import spawn_child

# Splitting one document into three sequences:
for i, sequence in enumerate(sequences):
    meta = spawn_child(
        parent_record.meta,
        child_idx=i,
        is_last_child=(i == len(sequences) - 1),  # last one closes the offset
    )
    yield SampleRecord(meta=meta, payload=sequence)
```

`spawn_child` does several things at once:
- Extends lineage with `child_idx` for a unique cursor
- Creates a `ContributorRef` with the appropriate `is_last_child` flag
- Propagates inherited contributors if the parent was already packed
- Preserves component tracking metadata

#### `pack_meta(primary_cursor, contributors, *, lane_id, component_sample_counts, ...)`

Builds metadata for a packed record that merges multiple inputs.  The
`primary_cursor` is the packed record's identity for replay; it must
be unique per lane.  The `contributors` list gathers every input's
contributor refs (via `contribution_refs()`) so the engine knows which base
offsets the packed record closes.  See
`PackSequences._create_packed_record` for the canonical implementation.

#### `tombstone_meta(ref, lane_id)`

Creates a tombstone that closes a base offset without carrying any payload:

```python
from zephon.ops.children import tombstone_meta

# We decided to drop this record, but it was the last child for its offset.
for ref in dropped_record.meta.contribution_refs():
    if ref.is_last_child:
        tombstone = SampleRecord(
            meta=tombstone_meta(ref, lane_id=dropped_record.meta.lane_id),
            payload=None,
        )
        yield tombstone
```

#### `tombstones_for_record(record)`

Wraps the loop above: returns the tombstone records a dropped record
owes (one per `contribution_refs()` entry with `is_last_child=True`,
empty when none are owed).

```python
from zephon.ops.children import tombstones_for_record

yield from tombstones_for_record(dropped_record)
```

### Writing a 1:1 Transform

The simplest and most common case.  Use
{py:meth}`~zephon.Pipeline.map_transform`:

```python
pipeline.map_transform(lambda p: {"tokens": tokenize(p["text"])})
```

Your function receives the payload dict and returns a new (or mutated)
payload.  The framework reuses the incoming `SampleMeta` unchanged, i.e., cursor, contributors, everything stays the same.

**Filtering**: return `None` to drop a record.  With `drop_none=True`
(the default), the framework automatically emits tombstones for every
closing contributor in the dropped record.  You do not need to handle
tombstones manually:

```python
# Drop short sequences (tombstones emitted automatically)
pipeline.map_transform(
    lambda p: p if len(p["input_ids"]) >= 128 else None
)
```

This automatic handling is implemented in
``MapTransform``, which iterates
`contribution_refs()` on the dropped record and emits a tombstone for each
ref with `is_last_child=True`.

### Writing a Fan-Out Operator

When one input produces multiple outputs, use `spawn_child` for *every*
emitted record.  Do not re-emit the parent's metadata unchanged, as that
would create duplicate cursors.

The contract:
- Call `spawn_child(parent, idx, is_last_child=...)` with deterministic,
  ascending indices matching emission order.
- Set `is_last_child=True` on exactly one child per base offset (typically
  the last one emitted).
- If all children are later dropped, emit tombstones for the closing
  contributor.

### Writing a Packing Operator

A packer consumes multiple input records and produces one output:

- Use `pack_meta(...)` to build the output metadata.
- Choose a deterministic `primary_cursor` that is unique per lane (a common
  pattern: `first_input.meta.cursor.child(0)`).
- Gather contributors from all inputs via `contribution_refs()`.
  Contributors whose `is_last_child=True` close their base offsets.
- Track component heritage via `component_sample_counts` so that
  `ensure_mixture` can still measure mixture ratios correctly after
  packing.
- Set `preserves_cursor_order=False` in your
  {py:class}`~zephon.ops.OpTraits`, since packing reorders records
  across chunk boundaries, which disables the fast monotone eviction path
  and switches to per-offset tracking.

### Filtering and Dropping Records

There are two paths, with very different tombstone responsibilities:

| Approach | Tombstone handling |
|---|---|
| `map_transform(fn, drop_none=True)` | **Automatic.** Framework emits tombstones for you. |
| `stateful_transform(push=..., ...)` | **Manual.** You must emit tombstones for dropped closing contributors. |
| Custom `BaseOp` with `process_one` / `process_many` | **Manual.** Same as stateful_transform. |

For `stateful_transform`, if your `push` or `transform` function drops
records, emit the tombstones it owes via
`zephon.ops.children.tombstones_for_record`.  Alternatively,
move filtering into a preceding `map_transform` with `drop_none=True` and
keep the stateful transform focused on buffering/reordering.

### Traits That Affect the Pipeline

{py:class}`~zephon.ops.OpTraits` controls how the planner wires
your operator into the pipeline:

| Trait | Default | What it means |
|---|---|---|
| `preserves_cursor_order` | Required | Records emerge in the same chunk/offset order they entered.  Set to `False` for shuffling, packing, or any cross-chunk buffering.  This switches the engine from monotone to [epoch-based eviction](checkpointing.md#epoch-based-eviction-packing-and-shuffling). |
| `requires_serial_state` | `False` | The operator maintains cross-invocation state (e.g., shuffle buffers).  Raises `RuntimeError` if `parallelism != 1` in deterministic mode. |
| `batch_shape_sensitive` | `False` | Outputs depend on how inputs are grouped into micro-batches.  Disables latency-flush in deterministic mode. |

Getting `preserves_cursor_order` wrong is the most impactful mistake: if
your operator reorders but claims to preserve cursor order, the monotone
eviction path may evict chunks too early, losing data needed for replay.

### Common Pitfalls

- **Duplicate `is_last_child=True`.**  Emitting two closing contributors
  for the same base offset double-counts in the bitmap and can cause
  premature eviction.  Exactly one closer per offset, always.

- **Non-deterministic `primary_cursor`.**  A packed record's cursor must be
  stable across runs. 

- **Dropping tombstones.**  The pipeline strips tombstones from the training
  stream *after* notifying the engine.  If a custom operator drops
  tombstones before they reach the engine, eviction stalls. 

- **Wrong `preserves_cursor_order`.**  Claiming `True` when your operator
  reorders leads to premature chunk eviction.  Claiming `False`
  unnecessarily is safe but slightly less efficient (per-offset bitmaps
  instead of the simple watermark).
