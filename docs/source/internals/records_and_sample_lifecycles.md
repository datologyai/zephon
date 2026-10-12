# Records and Sample Lifecycles

A sample starts as three integers and ends as a row in a tensor. In between it may be
split into four sequences, three of which are packed with fragments of other documents and
one of which is thrown away by a filter. This page is about how Zephon keeps track of all
of that, because the bookkeeping is what makes checkpointing possible at all.

**From A Pointer To Data.** A `WorkSource` does not hand out samples; it hands out
pointers, each a `(dataset_id, shard_id, sample_idx)` triple, grouped into work chunks.
The built-in fetch operator is the boundary: everything before it operates on pointers,
and everything after it operates on a {py:class}`~zephon.types.SampleRecord`, which pairs
a **payload** (usually a dict of fields) with the tracking metadata in
{py:class}`~zephon.types.SampleMeta`. The rest of this page is about what happens after
that transition.

## The Identity Of A Record

Every record carries a {py:class}`~zephon.types.SampleCursor`, which is the thing that
makes it addressable. It has four components, and it sorts on them in a different order
than it stores them:

```
SampleCursor sort key = (chunk_id, chunk_offset, lineage, sample_id)
                           │           │            │         │
                           │           │            │         └─ (dataset, shard, local id)
                           │           │            └─ path taken through fan-out
                           │           └─ position within the chunk
                           └─ which chunk the sample came from
```

The constructor takes `(chunk_id, chunk_offset, sample_id, lineage)`; the sort key swaps
the last two so that children of the same offset sort next to each other rather than being
scattered by sample ID. If you are reading `zephon/types.py` and the two orders look like
a mistake, that is why.

Think of the cursor as a passport. It records where the record was born (`chunk_id`,
`chunk_offset`, `sample_id`) and what happened to it on the way (`lineage`). It is
deterministic: the same pipeline configuration and the same inputs always produce the same
cursor for the same record. That determinism is load-bearing in two places. On restore,
the replay filter uses cursor **equality** to find the boundary between records the
training loop has already consumed and records it has not. And during a run, the engine
uses the `(chunk_id, chunk_offset)` prefix — the **base offset** — to decide when a chunk
has been fully consumed and its memory can be released.

## Fan-Out: One Sample, Many Records

Operators that produce several outputs from one input need to give each output its own
identity, and they do it through **lineage**: a tuple of integer indices appended to the
parent's cursor. A tokenizer splitting a long document into three sequences produces this:

```
Input (chunk 7, offset 2):
  cursor = (cid=7, off=2, lineage=(), sample_id=(ds0, sh5, 42))

                    tokenize(split_long_samples=True)
                    ┌──────────┼──────────┐
                    v          v          v

  child 0:  (7, 2, (0,), (ds0, sh5, 42))
  child 1:  (7, 2, (1,), (ds0, sh5, 42))
  child 2:  (7, 2, (2,), (ds0, sh5, 42))
```

All three children share the base offset `(7, 2)` — they still belong to the same original
sample — but the lineage paths make each cursor unique. Lineage nests: if a later operator
splits child 1 again, the grandchildren are `(1, 0)` and `(1, 1)`. The tree is always
deterministic because child indices follow emission order, which is why operators that
split must emit in a stable order.

The invariant that matters for fan-out is this one:

> For each base offset in a lane, **exactly one** record (or tombstone) must eventually
> signal that it is the last one.

That signal is the `is_last_child` flag on a {py:class}`~zephon.types.ContributorRef`. It
tells the engine that nothing further will arrive for that base offset, so the
corresponding slot in the chunk can be marked closed. `spawn_child` does this bookkeeping
for you, and raises if you try to close a base offset that is already closed — a cheap
guard against the most damaging mistake in this area, which is closing an offset twice.

## Fan-In: Many Records, One Output

Two built-in operations go the other way.

**Packing** (`pack_sequences` and `pack_flat`) combines fragments from *different* base
offsets into one record. The packed record gets a fresh **primary cursor** — its own
replay identity, which must be unique within the lane — and carries a list of
`ContributorRef` entries remembering every base offset that went into it. One packed
record can therefore close several base offsets at once, including offsets from different
chunks, and the engine processes each contributor independently:

```
child (7, 2, (0,))  ─┐
                      ├─ pack ──>  packed record
child (7, 3, (0,))  ─┘             cursor = (7, 2, (0, 0), ...)
                                   contributors:
                                     (7, 2, (0,))  is_last_child=True
                                     (7, 3, (0,))  is_last_child=True
```

Packing also tracks **component heritage**: how many samples, and optionally how many
tokens, each mixture component contributed to the packed record, in
`component_sample_counts` and `component_token_counts`. This is what lets `ensure_mixture`
still measure and correct mixture ratios after packing has mixed several components into a
single record. The token counts are `None` when packing runs before tokenization, in which
case `ensure_mixture` falls back to distributing the record's total tokens in proportion
to the sample counts.

**Batching** (`batch`) is the gentler fan-in: it collects records into a
{py:class}`~zephon.types.SampleBatch` without merging payloads or minting new cursors.
Each record inside the batch keeps its own cursor and metadata. The batch is a grouping
for the training loop, not a new identity.

## Contributors And Tombstones

Contributors are the accounting system that answers one question: *has the pipeline
finished with base offset `(chunk_id, chunk_offset)`?*

Every record carries contributor information, either explicitly — set by the fan-out or
packing helpers — or implicitly, since a record that sets nothing is treated as having a
single contributor pointing at its own cursor with `is_last_child=True`. For a 1:1
operator that neither splits nor drops anything, the defaults are correct and you never
have to think about this. Contributors become your problem in exactly three situations: a
fan-out operator creating several children from one base offset, a packing operator
merging contributors from several offsets, and a filter dropping the last record standing
for an offset.

That third case is where **tombstones** come in. Consider a deduplication filter that
drops a document. If that record was the last one for its base offset, no closing
contributor will ever reach the engine, the offset never closes, the chunk never evicts,
and memory grows without bound for the rest of the run. A tombstone is a record with no
payload whose entire purpose is to carry the closing `ContributorRef` that the dropped
record owed:

```
Base offset (10, 5) → filter decides to drop it

Without a tombstone:              With a tombstone:
  (10, 5) never closes              filter emits a tombstone for (10, 5)
  chunk 10 pinned forever           engine marks (10, 5) closed
  ✗ unbounded memory                chunk 10 evicts once its other offsets close
```

Tombstones are tagged in the metadata and flow through the pipeline like ordinary records,
but the engine strips them at the delivery boundary, after they have been counted. They
participate in eviction and are invisible to your model.

The rule for operator authors is to **always emit tombstones when dropping records**,
regardless of what else the pipeline contains. Whether a pipeline uses the monotone or the
epoch-based eviction path is decided at compile time from operator traits, and your
operator has no way of knowing which path it will end up on. Tombstones are harmless on
the monotone path and mandatory on the epoch-based one, so emitting them unconditionally
is the only choice that is correct in both cases.

## Chunks Are The Unit Of State

Putting it together: the **chunk** is the unit in which Zephon holds and checkpoints
state. The engine keeps a small set of **inflight chunks** per lane — chunks whose samples
have entered the pipeline but whose outputs have not all been delivered. As contributors
and tombstones close offsets, fully consumed chunks leave the inflight set, and at
checkpoint time only what remains has to be written down.

```
WorkSource           Engine                       Operators            Training loop
    │                   │                             │                      │
    │  work chunk       │                             │                      │
    │ ─────────────────>│  assign chunk_id            │                      │
    │                   │  add to inflight set        │                      │
    │                   │  yield sample pointers      │                      │
    │                   │ ───────────────────────────>│                      │
    │                   │                             │  fetch, transform,   │
    │                   │                             │  split, pack, filter │
    │                   │   notify(contributors)      │                      │
    │                   │ <───────────────────────────│                      │
    │                   │  update offset bitmaps      │                      │
    │                   │  evict completed epochs     │                      │
    │                   │                             │  records / batches   │
    │                   │                             │ ────────────────────>│
    │                   │                             │ (tombstones stripped)│
```

This is why checkpoint size does not grow with training time. It is proportional to the
number of inflight chunks per lane — a handful — rather than to the number of samples
consumed. [Checkpoint Replay and Bounded Recovery](checkpoint_replay.md) covers how the
engine decides that
a chunk is safe to drop, which is considerably more subtle than "all its offsets are
closed" once packing and shuffling are in the pipeline.

## Writing Operators That Respect All This

[User-Defined Operators](../pipelines/user_defined_operators.md) covers the ways to add
your own code to a pipeline. This section covers what that code owes the bookkeeping
above. Four builders in `zephon.ops.children` do the fiddly parts.

`spawn_child(parent, child_idx, *, is_last_child=False, tags=None)` builds the metadata
for a child derived from a parent, and is what you want for any true fan-out. It extends
the lineage with `child_idx`, creates a contributor with the right flag, propagates
inherited contributors if the parent was itself a packed record, and preserves component
tracking:

```python
from zephon.ops.children import spawn_child

for i, sequence in enumerate(sequences):
    meta = spawn_child(
        parent.meta,
        child_idx=i,
        is_last_child=(i == len(sequences) - 1),
    )
    yield SampleRecord(meta=meta, payload=sequence)
```

`pack_meta(primary_cursor, contributors, *, lane_id, component_sample_counts, ...)` builds
the metadata for a packed record. The `primary_cursor` is the packed record's replay
identity and must be unique within the lane; a reliable pattern is
`first_input.meta.cursor.child(0)`. The `contributors` argument gathers every input's
`contribution_refs()` so the engine learns which base offsets the packed record closes.

`tombstone_meta(ref, lane_id)` builds a single tombstone from a closing contributor, and
`tombstones_for_record(record)` wraps the loop you would otherwise write by hand,
returning one tombstone per closing contributor and an empty list when none are owed:

```python
from zephon.ops.children import tombstones_for_record

yield from tombstones_for_record(dropped_record)
```

**Who owes tombstones.** It depends on how you added your code, and the difference catches
people out:

| How you added it | Tombstones |
|---|---|
| `map_transform(fn, drop_none=True)` | **Automatic.** Returning `None` drops the record and the framework emits what it owed. |
| `stateful_transform(...)` | **Manual.** Emit them yourself for every dropped closing contributor. |
| A custom `BaseOp` | **Manual.** Same. |

If that sounds like a trap, the way out is to keep filtering in a preceding
`map_transform` with `drop_none=True` and let your stateful operator do nothing but buffer
and reorder.

**Traits that change how the planner wires you in.** {py:class}`~zephon.ops.OpTraits`
carries what the planner needs to know:

| Trait | Default | Meaning |
|---|---|---|
| `preserves_cursor_order` | Required — no default | Records leave in the same chunk/offset order they arrived. `False` for anything that shuffles, packs, or buffers across chunks. |
| `requires_serial_state` | `False` | The operator keeps state across invocations. In deterministic mode the planner forces its parallelism to 1. |
| `batch_shape_sensitive` | `False` | Output depends on how inputs were grouped into micro-batches. Disables time-based flushing in deterministic mode. |
| `indexable` | `True` | The operator preserves indexability through the plan. |

`preserves_cursor_order` has no default on purpose: every operator author has to make the
call consciously, because getting it wrong is the most expensive mistake available here.
An operator that reorders records but claims to preserve cursor order puts the pipeline on
the monotone eviction path, which will evict chunks that replay still needs. Claiming
`False` when you did not need to is merely a little slower.

**The pitfalls, in order of how much they hurt.** Closing a base offset twice — two
records with `is_last_child=True` for the same offset — double-counts in the engine's
bitmap and can evict a chunk early; `spawn_child` raises on the obvious version of this,
but a hand-rolled operator can still do it. A non-deterministic `primary_cursor` on a
packed record means replay cannot find its target on resume. Dropping tombstones inside a
custom operator before they reach the engine stalls eviction silently. And claiming the
wrong `preserves_cursor_order`, as above, loses data outright.
