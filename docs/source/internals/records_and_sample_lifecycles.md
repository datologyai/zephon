# Records and Sample Lifecycles

A sample starts as three integers and ends as a row in a tensor. On the way it may be
split into four sequences, three of which get packed with fragments of other documents
while a filter throws the fourth away. This page is about how Zephon keeps track of all
that, because the bookkeeping is what makes checkpointing possible at all.

**From a pointer to data.** A `WorkSource` does not hand out samples, it hands out
pointers, each a `(dataset_id, shard_id, sample_idx)` triple grouped into work chunks. The
built-in `fetch` operator is the boundary: everything before it works on pointers, and
everything after it works on a `SampleRecord`, which pairs a **payload** (usually a dict
of fields) with the tracking metadata in `SampleMeta`. The rest of this page is about what
happens after that transition.

**The identity of a record.** Every record carries a `SampleCursor`, which is the thing
that makes it addressable. It has four components, and it sorts on them in a different
order than it stores them:

```
SampleCursor sort key = (chunk_id, chunk_offset, lineage, sample_id)
                           │           │            │         │
                           │           │            │         └─ (dataset, shard, local id)
                           │           │            └─ path taken through fan-out
                           │           └─ position within the chunk
                           └─ which chunk the sample came from
```

The constructor takes `(chunk_id, chunk_offset, sample_id, lineage)`. The sort key swaps
the last two so that children of the same offset sort next to each other rather than being
scattered by sample ID. If you are reading `zephon/types.py` and the two orders look like
a mistake, that is why.

You can think of the cursor as a passport: it records where the record was born
(`chunk_id`, `chunk_offset`, `sample_id`) and what happened to it on the way (`lineage`).
It is deterministic, so the same `Pipeline` definition and the same inputs always produce
the same cursor for the same record, and we lean on that in two places. On restore, the
replay filter uses cursor **equality** to find the boundary between the records your
training loop has already consumed and the ones it has not. During a run, Zephon uses the
`(chunk_id, chunk_offset)` prefix, which we call the **base offset**, to decide when a
chunk has been fully consumed and its memory can be released.

Operators can have different relationships between input and output:

| Cardinality | What happens | Examples |
|---|---|---|
| 1 &rarr; 1 | One record in, one out, identity unchanged | Tokenizing without splitting, image decode |
| 1 &rarr; 0 or 1 | A record is kept or dropped | Dedup, length gating, `map_transform` returning `None` |
| 1 &rarr; N | One record becomes several | `tokenize(split_long_samples=True)` |
| N &rarr; 1 | Several records become one | `pack_sequences`, `pack_flat`, `batch` |


**Fan-out: one sample, many records.** Each output needs its own identity, and gets it
through **lineage**: a tuple of integer indices appended to the parent's cursor. A
tokenizer splitting a long document into three sequences produces this:

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

All three children share the base offset `(7, 2)`, because they still belong to the same
original sample, but the lineage paths make each cursor unique. Lineage nests: if a later
operator splits child 1 again, the grandchildren are `(1, 0)` and `(1, 1)`. The tree is
always deterministic because child indices follow emission order, which is why operators
that split must emit in a stable order.

The invariant that matters for fan-out is this one:

> For each base offset in a lane, **exactly one** record (or tombstone) must eventually
> signal that it is the last one.

That signal is the `is_last_child` flag on a `ContributorRef`, which tells Zephon that
nothing further will arrive for that base offset and the corresponding slot in the chunk
can be marked closed. `spawn_child` does this bookkeeping for you. When you split a record
that already carries contributors, only its final child inherits the parent's closing
flag, and every earlier child gets a non-closing copy, so splitting a split record cannot
close the source early. The **Splitting Records** section of [User-Defined
Operators](../pipelines/user_defined_operators.md) walks through an example.

**Fan-in: many records, one output.** Two built-in operations go the other way.

Packing (`pack_sequences` and `pack_flat`) combines fragments from different base offsets
into one record. The packed record gets a fresh **primary cursor**, which is its own
replay identity and has to be unique within the lane, and it carries a list of
`ContributorRef` entries remembering every base offset that went into it. One packed
record can therefore close several base offsets at once, including offsets from different
chunks, and Zephon processes each contributor independently:

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

Batching (`batch`) is the gentler fan-in. It collects records into a `SampleBatch` without
merging payloads or minting new cursors. Each record inside the batch keeps its own cursor
and metadata. The batch is a grouping for the training loop, not a new identity.

(contributors-and-tombstones)=

**Contributors and tombstones.** Contributors are the accounting system that answers one
question: *has the pipeline finished with base offset `(chunk_id, chunk_offset)`?*

Every record carries contributor information, either explicitly, set by the fan-out or
packing helpers, or implicitly, since a record that sets nothing is treated as having a
single contributor pointing at its own cursor with `is_last_child=True`. If you write a
1:1 operator that neither splits nor drops anything, the defaults are already correct and
you never have to think about this. Contributors become your problem in exactly three
situations: a fan-out operator creating several children from one base offset, a packing
operator merging contributors from several offsets, and a filter dropping the last record
standing for an offset.

That third case is where **tombstones** come in. Say you write a deduplication filter and
it drops a document. If that record was the last one for its base offset, no closing
contributor will ever reach Zephon, so the offset never closes, the chunk never evicts,
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

Tombstones are tagged in the metadata, and the runners treat them as control items: a
tombstone emitted upstream of your operator bypasses its accumulator and `process_many`
entirely and travels on to delivery, so your operator never sees one. At delivery a
tombstone only marks its offset complete. It does not advance the replay cursor, and it
does not trigger eviction on its own, and your model never sees it.

The only tombstones you need to think about are
the ones your own operator creates. The rule is to **always emit them when you drop
records**, regardless of what else the `Pipeline` contains. Zephon decides at compile
time whether a `Pipeline` uses the monotone or the epoch-based eviction path, based on
operator traits. Your operator has no way of knowing which one it will end up on.
Tombstones are harmless on the monotone path and mandatory on the epoch-based one, so
emitting them unconditionally is the only choice that is correct in both cases.

(chunks-are-the-unit-of-state)=

**Chunks are the unit of state.** [How Pipelines Run](../pipelines/how_pipelines_run.md)
explains that Zephon tracks progress by chunk and that a chunk is complete once every
record derived from it has been delivered. The bookkeeping above is how it knows:
contributors and tombstones close offsets, and a chunk whose offsets are all closed leaves
the per-lane set of **inflight chunks**.

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

[Checkpoint Replay and Bounded Recovery](checkpoint_replay.md) covers how Zephon decides
that a chunk is safe to drop, which turns out to be considerably more subtle than "all its
offsets are closed" once packing and shuffling are in the `Pipeline`.

**Writing operators that respect all this.** [User-Defined
Operators](../pipelines/user_defined_operators.md) covers the ways to add your own code to
a pipeline. What follows covers what that code owes the bookkeeping above. Four builders
in `zephon.ops.children` do the fiddly parts.

`spawn_child(parent, child_idx, *, is_last_child=False, tags=None)` builds the metadata
for a child derived from a parent, and is what you want for any true fan-out. It extends
the lineage with `child_idx`, creates a contributor with the right flag, propagates
inherited contributors if the parent was itself a packed record, and preserves component
tracking:

```python
from zephon.ops.children import spawn_child
from zephon.types import SampleRecord

for i, sequence in enumerate(sequences):
    meta = spawn_child(
        parent.meta,
        child_idx=i,
        is_last_child=(i == len(sequences) - 1),
    )
    out.append(SampleRecord(meta=meta, payload=sequence))
```

`pack_meta(primary_cursor, contributors, *, lane_id, component_sample_counts, ...)` builds
the metadata for a packed record. The `primary_cursor` is that record's replay identity
and has to be unique within the lane. `first_input.meta.cursor.child(0)` is only unique
when each input leads at most one packed record. If one input can lead several, as when
the built-in packer wraps a long sequence across bins, derive the index from the input
instead: the built-in packer uses `cursor.child(first.start)`, the offset at which that
input's fragment begins. The `contributors` argument gathers every input's
`contribution_refs()` so that Zephon learns which base offsets the packed record closes.

`tombstone_meta(ref, lane_id)` builds a single tombstone from a closing contributor, and
`tombstones_for_record(record)` wraps the loop you would otherwise write by hand,
returning one tombstone per closing contributor and an empty list when none are owed:

```python
from zephon.ops.children import tombstones_for_record

out.extend(tombstones_for_record(dropped_record))
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

**Traits that change how the planner wires you in.** `OpTraits` carries what the planner
needs to know:

| Trait | Default | Meaning |
|---|---|---|
| `preserves_cursor_order` | Required, no default | Records leave in the same chunk/offset order they arrived. `False` for anything that shuffles, packs, or buffers across chunks. |
| `requires_serial_state` | `False` | The operator keeps state across invocations. In deterministic mode the runner raises unless its parallelism is 1. |
| `batch_shape_sensitive` | `False` | Output depends on how inputs were grouped into micro-batches. Intended to disable time-based flushing in deterministic mode, but currently has no effect ([#49](https://github.com/datologyai/zephon/issues/49)). |
| `indexable` | `True` | The operator preserves indexability through the plan. |
| `parallelism` | `1` | Suggested worker count, used when the plan does not override it. |

We gave `preserves_cursor_order` no default on purpose, so that you have to make the call
consciously, because getting it wrong is the most expensive mistake available here. An
operator that reorders records while claiming to preserve cursor order puts the `Pipeline`
on the monotone eviction path, which will happily evict chunks that replay still needs.
Claiming `False` when you did not have to is merely a little slower.

**The pitfalls, in order of how much they hurt.** The worst is closing a base offset too
early, meaning a record with `is_last_child=True` that reaches Zephon before its siblings
have been delivered, which lets the chunk evict while replay still needs it. Closing an
offset twice is harmless, because the bitmap ignores repeats; it is the timing that
matters. `spawn_child` hands the closing flag only to the final child, but a hand-rolled
operator can still get this wrong. A non-deterministic `primary_cursor` on a packed record
means replay cannot find its target on resume. Forgetting to emit a tombstone for a
closing record your operator drops stalls eviction silently. And claiming the wrong
`preserves_cursor_order`, as above, loses data outright.
