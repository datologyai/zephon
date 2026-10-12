# Checkpoint Replay and Bounded Recovery

[Checkpointing and Resuming Pipelines](../pipelines/checkpointing_and_resuming.md) says
that a checkpoint records which work chunks were finished and which were still in
progress, and that a restore rebuilds operator state by re-processing a bounded number of
recent chunks. This page is about how the engine knows which chunks those are, which turns
out to be the hardest problem in the system.

**Why There Is No Sample Index.** A map-style loader checkpoints by writing down an index.
Zephon cannot, because a pipeline that splits documents, packs fragments, and reorders
records has no index to write down. Progress is spread across in-flight operator state,
partially filled bins, and records that have been reordered relative to their inputs. The
strawman alternative is to replay the whole pipeline from the start on every resume, which
is correct and useless, because crashing at step 100,000 means redoing 100,000 steps of
data processing before producing one new batch.

Zephon sits in between. The goal is to **replay only the chunks that were still in flight
at checkpoint time**, which is a handful regardless of how far into training you are.

## What A Checkpoint Holds

Per lane, three things.

**Inflight chunks**, serialized in full, meaning their sample pointers, component order,
and seed. On restore these are deserialized directly and the `WorkSource` is never asked
to regenerate them. These are the only chunks that get replayed through the operator
pipeline.

**Epoch boundaries**, for pipelines that need them: the list of sentinel `chunk_id` s
marking flush points. This can include a trailing boundary just past the last inflight
chunk, which preserves the flush point between replayed data and newly fetched data. On
restore the sentinels are re-injected at these positions so accumulators flush where they
flushed originally.

**WorkSource state**, so `next_chunk()` can resume producing *new* chunks from the right
place. The `WorkSource` is not replayed and not re-advanced, because `load_state_dict()`
restores its position directly. What goes in is up to the implementation. For
`StaticMixtureWorkSource` the sample sequence is fully determined by the datasets, the
seed, the shuffle knobs, and the lane. The state is therefore a compact set of per-dataset
cursor positions plus mixture bookkeeping, and restoring it is instantaneous no matter how
far into training you are. [WorkSources and Work Chunks](worksources_and_work_chunks.md)
covers what those cursors are and what the rest of the contract requires of a source you
write yourself.

Alongside these sit the replay cursor per lane, the lane progress pointers, and the
round-robin position discussed in [Elastic
Determinism](elastic_determinism.md#what-actually-survives-a-topology-change).

Chunks already fully consumed are not in the checkpoint, because they are gone. Chunks not
yet fetched are not in it either, because the `WorkSource` will produce them. This is why
checkpoint size is bounded by inflight chunks per lane rather than by training progress.

One thing is deliberately absent. The per-offset completion bitmaps are runtime-only, and
are rebuilt as the replayed chunks flow through the pipeline and re-trigger contributor
tracking. That keeps checkpoints small at the cost of making restore time proportional to
the number of inflight chunks.

## Eviction

**Eviction** is what keeps the inflight set small. Once every sample derived from a chunk
has been delivered, the chunk leaves the set, and a checkpoint taken afterwards neither
stores nor replays it.

```
Lane 0 inflight chunks over time:

  step 100:  [ chunk 3 | chunk 4 | chunk 5 ]
  step 200:  [ chunk 4 | chunk 5 | chunk 6 ]    chunk 3 evicted
  step 300:  [ chunk 5 | chunk 6 | chunk 7 ]    chunk 4 evicted
```

Without it the inflight set grows without bound and every resume replays everything. The
difficulty is timing. Evict too early and replay loses data it needed. Evict too late and
checkpoints bloat.

The planner picks a strategy at compile time by looking at every operator's
`preserves_cursor_order` trait:

```
Pipeline compiled
       │
  Do all operators preserve cursor order?
      ╱ ╲
    yes   no  (packing, shuffling, mixture correction)
     │     │
  Monotone  Epoch-based
  eviction  eviction
```

### Monotone Eviction

When every operator preserves cursor order, records emerge in the same chunk order they
entered. If the most recently delivered record belongs to chunk *N*, everything from
chunks below *N* has already gone out, and those chunks evict immediately. A simple
watermark is enough. Most pipelines without shuffling or packing take this path.

### Epoch-Based Eviction

Packing, shuffling, and mixture correction break the watermark in two ways, and the second
is much worse than the first.

The obvious problem is reordering. A packed record can combine fragments from chunks 5 and
7, and chunk 5's last sample can be delivered *after* something from chunk 7. A rule of
"evict everything below the most recent chunk" would then drop chunk 5 while part of it is
still in flight. Per-offset tracking fixes this one.

The deeper problem is **accumulator history dependence**. The packer's bin state, meaning
which bins exist and how full they are, depends on the entire history of what it has seen
rather than just on what is currently in the bins. After a chunk's samples have all left
the bins, their influence on where subsequent samples land persists. Two packers starting
from different states and fed identical input may *never* converge. With two bins of
capacity 10 and an input stream of `[3, 3, 3, …]` , starting from `(10, 10)` versus
`(10, 5)` gives the same periodic emission cycle, permanently phase-shifted.

That creates a direct conflict. Eviction wants to discard consumed chunks to free memory.
Deterministic replay wants to reconstruct the accumulator state the original run had, and
that state was shaped by inputs from exactly the chunks eviction wants to discard. Replay
from a checkpoint would start the packer empty instead of mid-stream, produce different
packed records, and the replay filter's target cursor might never appear at all. The same
argument applies to mixture correction, whose deficit state depends on emission history,
and to shuffling, whose buffers span chunk boundaries.

**Flush sentinels** resolve it. Every `flush_every_k_chunks` chunks per lane, the engine
injects a sentinel that forces history-dependent accumulators to emit everything and reset
to a clean state, as described in [Operators and
Runners](operators_and_runners.md#flush-sentinels-and-stalling). The sentinel divides the
stream into **epochs**, and each epoch starts from a state that replay can reproduce
without any of the preceding chunks:

```
Epoch 0          Epoch 1          Epoch 2
[chunk 0, 1]  →  [chunk 2, 3]  →  [chunk 4, 5]  → ...
             ↑                ↑
          sentinel         sentinel
           (flush)          (flush)
```

This buys replay safety by spending output quality. Each flush interrupts the accumulator
mid-stride: the packer emits partially filled bins and pays for them in padding, the
shuffle buffer emits a smaller window and randomises less, and the mixture corrector
drains mid-rebalance. `flush_every_k_chunks` is the dial on that trade, and it is
genuinely three-way. Smaller K means faster resume, since fewer chunks are in flight to
replay. Larger K means fewer interruptions and better output. And memory scales with K,
since at least a full epoch stays inflight until it can be evicted. The default is 8.

For a non-monotonic pipeline, setting `flush_every_k_chunks=0` is an error. Without
sentinels, accumulator state depends on unbounded history and safe eviction is impossible.
For a monotone pipeline, setting a positive value gets you a warning and the value is
forced back to 0, since sentinels there buy nothing and would perturb batch boundaries for
free.

### Per-Offset Tracking

Within an epoch, completion is tracked per offset. Each sample occupies a position in its
chunk, and operators may split it into several children or merge children from different
offsets into one packed record. An offset closes only when all of its children have been
delivered, which the engine learns from the `is_last_child` flags on contributor metadata,
and from tombstones when a filter drops the last child standing (see [Records and Sample
Lifecycles](records_and_sample_lifecycles.md#contributors-and-tombstones)). A bitmap per
chunk records which offsets are closed, with a popcount alongside so that "is this chunk
complete?" stays O(1).

```
Chunk 5 (4 offsets):  [x] [x] [ ] [x]    3 of 4 closed
Chunk 6 (4 offsets):  [x] [x] [x] [x]    all closed, but not evicted
Chunk 7 (4 offsets):  [ ] [ ] [ ] [ ]    none closed

→ Nothing is evicted: 5 and 6 are in the same epoch, and 5 is incomplete.
```

### Atomic Per-Epoch Eviction

Chunks within an epoch share accumulator history, since where chunk 1's samples landed in
the bins depends on what chunk 0 put there. Evicting chunk 0 while chunk 1 is still
inflight would break replay for exactly the reason above. So an epoch evicts
**atomically**: all of its chunks or none of them.

The engine walks a lane's recorded boundaries in ascending order and evicts contiguously
from the bottom, stopping at the first epoch that is incomplete. Epoch boundaries are what
make this safe across epochs. Evicting epoch 0 while epoch 1 is still running is fine
precisely because epoch 1's accumulators started empty after the sentinel flush.

```
Epoch 0 (chunks 0–1):  all offsets closed   → evict together
Epoch 1 (chunks 2–3):  still in progress    → keep
Epoch 2 (chunks 4–5):  not started          → keep
```

**Only this lane's own recorded boundaries authorise eviction.** That restriction matters,
because sentinels are per-lane, so a boundary established when a different lane flushed
says nothing about this lane's accumulator history. Until a lane has recorded a boundary
of its own, its entire open epoch is retained even when every offset in it is closed.
Early chunks therefore stay inflight until the first sentinel arrives, which is a
deliberate trade of a little memory for an eviction rule with no special cases. The full
pipeline drain at end of stream handles whatever is left.

**Cursor pinning** adds the last constraint. The replay cursor is the identity of the last
delivered record, and it must always point into a chunk that is still inflight, or the
replay filter cannot find its target on resume. If the cursor sits in an epoch that is
otherwise eligible, the engine stops there and leaves it alone. This resolves on its own
within a few deliveries, as soon as a post-sentinel record goes out.

## Replay On Resume

Restoring and then iterating runs the saved state back through the full pipeline:

```
Checkpoint
  │  inflight chunks + replay cursor + WorkSource state + epoch boundaries, per lane
  v
┌──────────────────────────────────────────────────────┐
│ Engine source stream                                  │
│   Phase 1: yield restored inflight chunks, by chunk ID│
│   Phase 2: fetch new chunks from the restored source  │
└────────────────────────┬─────────────────────────────┘
                         v  sample pointers
┌─── Operator pipeline (deterministic replay) ─────────┐
│  fetch → tokenize → pack → ensure_mixture → ...       │
└────────────────────────┬─────────────────────────────┘
                         v  records
┌─── ReplayFilter ─────────────────────────────────────┐
│  Per lane: drop until cursor == target, then emit     │
└────────────────────────┬─────────────────────────────┘
                         v  new records only
                   Training loop
```

First the inflight chunks are deserialized back into the inflight set and each lane's
`WorkSource` state is restored. Then the engine yields those chunks *before* fetching
anything new, re-injecting flush sentinels at the stored boundary positions so
accumulators flush exactly where they did before. Those chunks flow through the whole
pipeline, through fetch, tokenize, pack, and batch, just as they originally did. Because
the pipeline is deterministic they produce the same records and rebuild the same
accumulator state as a side effect. Finally the replay filter drops the prefix the
training loop already consumed.

### The ReplayFilter

The planner inserts a `ReplayFilter` into every pipeline automatically, so you never add
one. If the pipeline has a `batch` operator it goes immediately before it, otherwise it is
appended at the tail, and it takes its name from the operator it sits next to. You can see
it in `Pipeline.explain()` :

```
Stage[0] place=local runner=threads cap=8 mode=microbatches
  fetch@p4 -[in_q=16]-> tokenize@p4
  --[stage_out=16]-->
Stage[1] place=local runner=inline cap=1 mode=stream_items
  ensure_mixture@p1 -> batch_replay_filter@p1 -> batch@p1
  ==[final_prefetch=3]==> pipeline_end
```

On a fresh run it is a no-op that passes everything through. It only does anything on
resume.

Its placement before `batch` is what makes `Batch(drop_last=True)` the one operator
allowed to stall at an epoch boundary. Because batching only emits complete batches, the
consumer-visible checkpoint cut falls between whole batches. On resume the prefix is
dropped *before* batching, and batching rebuilds the suffix from an empty buffer at the
same boundary. No extra batch state has to be checkpointed, and this holds even when the
pump thread has run far ahead of the consumer, because the cut is defined by delivered
batches rather than by how much work the pump had done.

When the filter drops a record during the prefix it emits **tombstones** for it. This is
not optional. Without them, per-offset tracking would never close the offsets belonging to
replayed records and their chunks could never be evicted.

**Equality, not a threshold.** At checkpoint time the engine records, per lane, the
`SampleCursor` of the most recently delivered record. On resume the filter drops records
until it sees a cursor **equal** to that target, drops the target too, and then forwards
everything after it.

Equality rather than `<=` is the right rule because the tail of a pipeline with shuffling
does not arrive in cursor order:

```
Original run:   C2  C1  C3  C4  C5  ...
                        ↑ checkpoint here, saved target = C3
```

A threshold rule would drop C1, C2 and C3, which is correct, but it would also drop any
*later* record whose cursor happens to fall below C3, and those records were never
consumed. The equality rule simply waits for C3 to reappear in the deterministic replay
and forwards everything after it, which is correct regardless of output ordering. It works
because a cursor is a unique identity rather than an ordinal position.

### What A Custom Operator Owes Replay

An operator that buffers across chunk boundaries has to cooperate with all of the above,
or the engine will evict a chunk whose data the operator is still holding. Concretely:

- Set `preserves_cursor_order=False` in `OpTraits` if the operator reorders anything.
- Propagate contributor metadata, and set `is_last_child` correctly.
- Emit tombstones for dropped final fragments.
- Implement `flush(reset=True, lane_id=...)` so that it fully resets that lane's state.
After a mid-stream flush the accumulator must report no pending data for that lane. The
runner treats anything else as a contract violation and raises.
- Do not plan on stalling. It is supported for `Batch(drop_last=True)` and nothing else.

## Multi-Rank Aggregation

Each rank's engine owns a subset of lanes, but a complete checkpoint has to cover all of
them, so the ranks merge their states through a shared directory using a leader-follower
protocol. The leader is global rank 0, DataLoader worker 0.

```
Leader                     aggregate_dir                   Followers
  │                            │                               │
  │── publish round id ───────>│                               │
  │                            │<────── read round id ─────────│
  │── write own state ────────>│                               │
  │                            │<────── write states ──────────│
  │── poll: all ranks seen,    │                               │
  │   all lanes covered? ─────>│                               │
  │── merge, write merged ────>│                               │
  │                            │<────── poll for merged ───────│
  │                            │─────── read merged ──────────>│
  │── delete own state,        │                               │
  │   previous merged file,    │                               │
  │   round file ─────────────>│                               │
```

The leader's wait condition is stricter than it might look. It holds until it has seen a
state file from every rank in the `world_size` **and** the union of their lane progress
covers every one of the `canonical_replicas` lanes. Both conditions have to hold, so a
rank that reports in with no lanes does not satisfy the coverage check on its own. If
`aggregate_timeout_s` expires first the error names the missing ranks and the missing
lanes, which is usually enough to find the rank that forgot to call `checkpoint()` .

The current round's merged file is left in place until the next checkpoint round, so a
straggler can still read it, and the leader deletes the *previous* round's merged file
when it writes the new one. State files are written with the configured serializer and
compressor (msgpack and zstd by default) rather than as plain text, and are named by rank,
DataLoader worker, and pid so that several workers on one rank do not collide. Against
object storage the polling backs off exponentially, capped at five seconds.
