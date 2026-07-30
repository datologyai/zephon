# Checkpointing

Training runs on large clusters are interrupted by node failures,
preemptions, and rescheduling.  Checkpointing lets you resume from where
you left off instead of replaying from the start.  This page covers what
you need to do to checkpoint a Zephon pipeline and how the system minimizes
wasted work on resume.

---

## What You Need to Do

### The API

Checkpointing a Zephon pipeline is two calls:

```python
# During training — capture the current state
state = pipeline.checkpoint()

# On resume — restore before iterating
pipeline.restore(state)
for batch in pipeline:
    ...
```

{py:meth}`~zephon.Pipeline.checkpoint` returns a plain Python dict
(JSON-serializable) that captures the full pipeline state: which chunks are
in flight, how far each lane has progressed, and where each lane's
WorkSource state sits.  You are responsible for persisting this dict
alongside your model checkpoint (e.g., write it to disk as JSON, or embed
it in your training framework's checkpoint payload).
{py:meth}`~zephon.Pipeline.restore` validates the checkpoint structure
eagerly — a malformed dict raises immediately — but applies the state
lazily when you start iterating, not at the `restore()` call itself.

Here is a more complete training loop with checkpoint save and resume:

```python
# --- Save ---
for step, batch in enumerate(pipeline, start=start_step):
    loss = train_step(batch)
    if step % save_interval == 0:
        ckpt = pipeline.checkpoint()      # all ranks call this
        if is_rank_zero:
            save_to_disk(model_state, optimizer_state, ckpt, step)

# --- Resume ---
model_state, opt_state, zephon_ckpt, start_step = load_latest()
pipeline.restore(zephon_ckpt)
for step, batch in enumerate(pipeline, start=start_step):
    loss = train_step(batch)
    ...
```

### Multi-Node: The Aggregation Directory

On a single rank with a single pipeline instance, `checkpoint()` already
has all the information it needs — the Engine owns all
[lanes](determinism.md#lanes-zephons-approach).
In distributed training, each rank's Engine only knows about its own lanes.
To produce a complete checkpoint, the individual per-rank states need to be
merged.

This is what the **aggregation directory** is for:

```python
pipeline = pipeline.options(
    aggregate_dir="/shared/zephon_agg/",
    run_id="run-20250301-abc",     # recommended: unique per run
)
```

The aggregation directory must be on a **shared/distributed filesystem** visible to
all ranks (or a cloud path like `s3://...`).  During `checkpoint()`, each
rank writes its local state to this directory.  A leader rank merges the
per-rank state files, and every rank reads back the merged result.  The
details of this protocol are described [below](#multi-rank-aggregation).

The reason Zephon aggregates state globally instead of having each rank
write only its own local state is **elastic resumption**: when you resume
on a different number of GPUs, lane ownership is redistributed.  A rank
that did not own lane 3 during the original run might own it after
resumption — and it needs that lane's state to continue correctly.  Only
a globally merged checkpoint makes this possible.  See
[Elastic Determinism](determinism.md) for background on lane remapping.

```{warning}
If two runs share the same aggregation directory and their checkpoint calls
overlap in time, stale files from a previous run could interfere.  Always ensure to supply
a unique `run_id` per run (or use a fresh aggregation directory) to avoid
this.
```

```{note}
For single-rank setups the aggregation directory is not
needed — `checkpoint()` returns the local state directly.
```

### Checkpoint Is a Collective Operation

`checkpoint()` must be called by **every rank**.  The leader rank waits
for all ranks to write their local state before merging.  If some ranks
call `checkpoint()` and others do not, the leader will time out and raise a
`RuntimeError` listing which lanes are missing.  The timeout is
configurable via `aggregate_timeout_s` in
{py:meth}`Pipeline.options() <zephon.Pipeline.options>`.

### Elastic Resumption

The merged checkpoint is self-contained: it includes the state for every
lane.  Any rank can restore from it, regardless of which lanes it owned
when the checkpoint was taken.  This means you can checkpoint on 8 GPUs
and resume on 4 (or vice versa), as long as `canonical_replicas` stays
the same.

```{note}
Changing `canonical_replicas` between checkpoint and restore is **not
supported** — `restore()` will raise a `RuntimeError`.  See [Elastic Determinism](determinism.md) for background
on `canonical_replicas`.
```

---

## How It Works

### The Challenge: Iterable Pipelines

As discussed in
[Why an Iterable Pipeline?](../basic_concepts.md#why-an-iterable-pipeline),
Zephon cannot map a training step back to a sample index without actually
running the pipeline.  The approach  of storing a sample index does
not work for iterable pipelines with online processing.

A strawman alternative is to replay the entire pipeline from the beginning
on every resume.  This is correct but expensive: crashing at step 100,000
means repeating all 100,000 steps of data processing before producing a
single new batch.

Zephon's chunk-based design falls in the middle of both.  The goal is to
**replay only the chunks that were in flight (not yet fully consumed) at
checkpoint time**, which is typically a handful of chunks, regardless of how far
into training you are.

### Chunks and Inflight State

Recall that the WorkSource produces **work chunks**, i.e., fixed-size groups of
sample pointers.  The Engine requests chunks sequentially per lane and
feeds their samples through the operator pipeline.  At any given moment,
each lane has a small number of **inflight chunks**: chunks whose samples
have entered the pipeline but whose results have not all been delivered to
the training loop yet.

The checkpoint captures, per lane:

1. **Inflight chunks** — serialized in full (sample pointers, component
   order, seed).  On restore these are deserialized directly; the
   WorkSource is never asked to regenerate them.  Only inflight chunks
   are replayed through the operator pipeline.

2. **Epoch boundaries** — for non-monotonic pipelines, the list of
   sentinel `chunk_id`s that mark flush points.  This can include a
   trailing boundary just beyond the last inflight chunk, which preserves
   the flush point between replayed inflight data and newly fetched data.
   On restore these are re-injected at the same positions so accumulators
   flush at the same points as the original run (see
   [Epoch boundary persistence](#epoch-boundary-persistence)).

3. **WorkSource state** — each per-lane WorkSource serializes its internal
   state via `state_dict()` so that `next_chunk()` can resume producing
   *new* chunks from where it left off.  The WorkSource is not replayed or
   re-advanced; `load_state_dict()` restores its position directly.
   What goes into this dict is up to the implementation.  For the built-in
   {py:class}`~zephon.work.StaticMixtureWorkSource`, the sample sequence
   is deterministic (fixed by datasets, seed, shuffle knobs, and lane ID),
   so the state is just a per-dataset position integer — restoring it is
   instantaneous regardless of how far into training you are.  A custom
   WorkSource must serialize whatever internal state its `next_chunk()`
   needs to continue correctly.

Chunks that have already been fully consumed are not in the checkpoint.
They are gone (evicted).  Chunks that have not been fetched yet do not need to be
stored either — the WorkSource will produce them from its restored state.

This is why the checkpoint size stays small and bounded: it is proportional
to the number of inflight chunks per lane, not to the
total number of samples consumed.  In practice, a checkpoint is quite small. 
Each inflight chunk stores `chunk_size` sample pointers (`(dataset_id, shard_id, sample_idx)` triples),
plus a small amount of per-lane metadata.

### Chunk Eviction

The mechanism that keeps the inflight set small is **chunk eviction**.
Once all samples from a chunk have been delivered to the training loop,
that chunk is removed from the inflight set.  If a checkpoint is taken
after eviction, the chunk is no longer part of the saved state and does not
need to be replayed on resume.

```
Lane 0 inflight chunks over time:

  step 100:  [ chunk 3 | chunk 4 | chunk 5 ]
                                    ^^ being processed

  step 200:  [ chunk 4 | chunk 5 | chunk 6 ]
               ^^ chunk 3 evicted (fully delivered)

  step 300:  [ chunk 5 | chunk 6 | chunk 7 ]
               ^^ chunk 4 evicted
```

Eviction is what makes Zephon's checkpointing efficient.  Without it, the
inflight set would grow unboundedly and every resume would replay
everything.  The challenge is evicting chunks at exactly the right time:
too early and we lose data; too late and checkpoints are unnecessarily
large.

Zephon selects an eviction strategy during pipeline compilation based on
operator properties:

```
Pipeline compiled
       |
  All operators preserve cursor order?
      / \
    Yes   No (packing, shuffling, ...)
     |     |
  Monotone   Epoch-based
  eviction   eviction
  (simple    (flush sentinels +
  watermark)  per-offset bitmaps)
```

The planner determines this by checking each operator's
`preserves_cursor_order` trait.  If every operator in the pipeline
preserves it, the fast monotone path is used.  Otherwise the epoch-based
path kicks in.

#### Monotone Eviction (Simple Pipelines)

When the pipeline preserves cursor order — meaning samples emerge in the
same chunk-ID order they entered — eviction is straightforward.  If the
most recently delivered sample belongs to chunk *N*, then every sample from
chunks *< N* has already been delivered.  Those older chunks can be evicted
immediately.

Most pipelines without shuffling or packing qualify for this fast path.

(epoch-based-eviction)=
#### Epoch-Based Eviction (Packing and Shuffling)

Operators like `pack_sequences`, `shuffle`, and `ensure_mixture` can
reorder samples across chunk boundaries.  A packed record might combine
fragments from chunks 5 and 7, and chunk 5's last sample might be
delivered *after* something from chunk 7.  The simple "evict everything
older than *N*" rule would prematurely evict chunk 5.

These operators also introduce a deeper problem: **accumulator history
dependence**.  The packing accumulator's bin state — which bins exist,
their sizes, their remaining capacity — depends on the **full history** of
inputs it has seen, not just the samples currently in the bins.  After a
chunk's samples have all left the bins (been packed and emitted), their
influence on bin placement persists.  Two packer instances starting from
different states and processing identical subsequent input may **never
converge**: with 2 bins of capacity 10 and an input stream of
`[3, 3, 3, …]`, starting from states `(10, 10)` vs `(10, 5)` produces the
same periodic bin-emission cycle but permanently phase-shifted.

This creates a tension:

- **Chunk eviction** must free memory by discarding fully consumed chunks.
- **Deterministic replay** must reconstruct identical accumulator state on
  resume — but the accumulator state was shaped by inputs from the now-evicted
  chunks.

Without intervention, replaying from a checkpoint would start the packer
from empty state instead of the state it had in the original run, producing
different packed record boundaries.  The ReplayFilter's target cursor may
never appear in the replay stream, causing data loss.  The same reasoning
applies to `EnsureMixtureAccumulator` (whose SWRR deficit state depends on
full emission history) and `ShuffleBuffer` (whose buffer contents span
chunk boundaries).

Zephon solves this with **flush sentinels** and **per-epoch atomic
eviction**.  The core idea is to periodically force history-dependent
accumulators to emit all buffered data and reset to a clean state,
creating boundaries where replay can start fresh without needing evicted
chunks.  This introduces an inherent tradeoff: the periodic flush
interrupts the accumulator's natural operation — a packer emits partially
filled bins (more padding waste), a shuffle buffer emits a
smaller-than-usual batch (less randomisation), a mixture corrector drains
mid-rebalance.  The epoch size (`flush_every_k_chunks`) controls how often
this interruption happens, balancing output quality against replay safety.

##### Flush sentinels and epoch boundaries

A **flush sentinel** is a lightweight record that the engine injects into
the source stream every `flush_every_k_chunks` chunks per lane.  The
sentinel flows through the pipeline like any other record, but at each
operator boundary the runner intercepts it:

- The runner calls `flush(reset=True, lane_id=lane)` on the accumulator,
  which emits and **fully resets** that lane's state — sentinels are per-lane,
  so other lanes are untouched.  For
  history-dependent accumulators (`preserves_cursor_order=False`) this is
  the load-bearing reset.  For order-preserving accumulators the flush is
  harmless (a no-op or trivial drain).  The only built-in operator that
  intentionally stalls instead of flushing is `Batch(drop_last=True)`; all
  non-monotonic built-ins flush to a fresh state at the sentinel (see
  [Accumulator stalling](accumulators_operators.md#stalling-at-epoch-boundaries)).

The sentinel divides the source stream into **epochs** — windows of K
chunks between consecutive flush points.  Within an epoch, the
accumulator builds up state normally.  At the epoch boundary, the
sentinel forces a flush that clears all state, making the next epoch
independent of all prior ones:

```
Epoch 0          Epoch 1          Epoch 2
[chunk 0, 1]  →  [chunk 2, 3]  →  [chunk 4, 5]  → ...
             ↑                ↑
         sentinel          sentinel
          (flush)            (flush)
```

On resume, only the inflight chunks need to be replayed.  The engine
re-injects sentinels at the same positions (stored in the checkpoint),
so the accumulator hits the flush at the same point and starts the next
epoch from empty state — identical to the original run.

**Choosing `flush_every_k_chunks`.**  The epoch size K is a three-way
tradeoff:

- **Replay speed.**  On resume, all inflight chunks must be replayed
  through the operator pipeline.  The number of inflight epochs depends
  on buffering depth, prefetch, and how far the pump runs ahead of the
  consumer — replay cost grows with K.  Smaller K means faster resume.
- **Output quality.**  Each flush interrupts the accumulator's natural
  operation.  A packer emits partially filled bins (more padding waste),
  a shuffle buffer emits a smaller-than-usual batch (less randomisation),
  a mixture corrector resets its deficit tracking.  Larger K means fewer
  such interruptions and higher output quality.
- **Memory.**  Inflight chunks consume memory (fetched data, operator
  buffers, prefetch queues).  Since at least one full epoch must stay
  inflight until eviction, memory usage is proportional to K.

For non-monotonic pipelines, `flush_every_k_chunks` must be positive —
setting it to 0 is an error, since without flush sentinels the
accumulator's state would depend on the full input history and safe
eviction would be impossible.  The default is 8, which is a reasonable
balance for most workloads.  You can tune it via
{py:meth}`Pipeline.options() <zephon.Pipeline.options>`.  Monotone pipelines do not need flush sentinels.  If you explicitly set a
positive value for a monotone pipeline, the engine emits a warning but
still injects sentinels unnecessarily.  That can perturb batch boundaries
and Batch stalling behavior without improving replay safety.  Remove the
explicit setting or set `flush_every_k_chunks=0` to avoid this.

##### Per-offset tracking

Within an epoch, Zephon tracks completion at the **per-offset** level.
Each sample in a chunk occupies a specific offset (its position within the
chunk).  As a sample flows through the pipeline, operators may split it
into multiple **children** (e.g., a long document tokenized into several
sequences) or merge children from different offsets into a single packed
record.  An offset is only closed once *all* of its children have been
delivered.  To track this, every record carries **contributor metadata** —
a `ContributorRef` that references the base offset(s) it was derived from,
plus an `is_last_child` flag indicating whether it is the final child for
that offset.  When the last child for every offset in a chunk has been
delivered, the chunk is fully consumed.  See
[Sample Lifecycle](sample_lifecycle.md) for more on how samples spawn
children as they flow through operators.

```
Chunk 5 (4 offsets):  [x] [x] [ ] [x]    ← 3 of 4 offsets closed
Chunk 6 (4 offsets):  [x] [x] [x] [x]    ← all closed, but NOT evicted yet
Chunk 7 (4 offsets):  [ ] [ ] [ ] [ ]    ← none closed

→ Nothing is evicted: chunks 5 and 6 are in the same epoch,
  and chunk 5 is incomplete.
```

When an operator drops all fragments for a given offset (e.g., a filter
that removes invalid samples), it emits a **tombstone** — a special record
that signals completion of that offset without carrying any training
payload.  Tombstones flow through the operator pipeline but are stripped
at the Engine's delivery boundary before reaching the training loop.

##### Atomic per-epoch eviction

Chunks within an epoch share accumulator history — the bin placement for
chunk 1's samples depends on what chunk 0 put into the bins.  Evicting
chunk 0 while chunk 1 is still inflight would break replay (replay starts
the packer from empty state instead of the state shaped by chunk 0).
Therefore, chunks within an epoch must be evicted **atomically**: either
all chunks in the epoch are done, or none are evicted.

Once all offsets in all chunks of an epoch are closed (all bitmaps full),
the epoch can be evicted as a unit.  Epoch boundaries established by flush
sentinels guarantee that no accumulator state from the evicted epoch
influences subsequent output.  This makes cross-epoch eviction safe:
evicting epoch 0 while epoch 1 is still in-flight is correct because
epoch 1's accumulator started from empty state (after the sentinel flush).

This guarantee assumes that history-dependent accumulators actually flush
and reset at the sentinel.  Zephon intentionally does **not** support
stalled non-monotonic operators at the moment: if an operator both
reorders and stalls, the delayed reset point depends on cross-boundary
consumption history that is not encoded in the checkpoint payload.
`Batch(drop_last=True)` is the special exception because it preserves
cursor order and is handled by the ReplayFilter-before-Batch path
described below.

```
Epoch 0 (chunks 0-1):  all offsets closed  → evict together
Epoch 1 (chunks 2-3):  still in progress   → keep
Epoch 2 (chunks 4-5):  not started yet     → keep
```

The engine also tracks an **epoch floor** per operator — the lowest
`chunk_id` that could still influence a `preserves_cursor_order=False`
accumulator's state.  After a sentinel flush the floor advances to the
first chunk of the new epoch; between sentinels it is lowered if a record
with a smaller `chunk_id` enters.  This prevents considering epochs whose
chunks are still being processed by the accumulator.

**Before the first sentinel fires** (i.e., during the first K chunks of a
lane), no epoch boundary exists yet and all chunks belong to a single
open epoch.  The engine falls back to treating everything below the epoch
floor as one atomic group: if all chunks below the floor are fully done
(all offsets closed), they can be evicted together.  This matters in
practice because the pump thread in the thread and process runners runs
ahead of the consumer — by the time the first record is delivered, the
pump may have processed many chunks and the floor may have advanced well
past them.  Without this fallback, those early chunks would stay inflight
until the first sentinel creates a proper epoch boundary, unnecessarily
inflating memory.  The atomic "all below floor" check is safe because all
those chunks are in a single epoch with shared accumulator history — either
they all evict (replay starts from scratch, which is correct since nothing
was evicted before them) or none do.

##### Cursor pinning

The replay cursor (the last delivered record's identity) must always
reference an inflight chunk, otherwise the ReplayFilter cannot find its
target on resume.  If the cursor references a chunk in an epoch that is
eligible for eviction, the engine skips that epoch's eviction until the
cursor advances past it.  This typically resolves within a few deliveries
after the sentinel flush, when the first post-sentinel record is
delivered.

##### Epoch boundary persistence

Epoch boundary positions (the list of sentinel `chunk_id`s per lane) are
persisted in the checkpoint.  On replay, the engine re-injects sentinels at
these stored positions during Phase 1 (inflight chunk replay).  Without
this, the accumulator on replay would process pre-sentinel and
post-sentinel chunks as one continuous stream, building up state that
differs from the original run.

When the inflight set spans multiple epochs (possible when the pipeline has
deep prefetch buffers), all intermediate sentinel positions are stored and
re-injected.

The bitmap state is runtime-only and **not** persisted in the checkpoint.
On resume, it is rebuilt as the replayed inflight chunks flow through the
pipeline and re-trigger contributor tracking.  This keeps checkpoints small
but means resume time is proportional to the number of inflight chunks
times the per-chunk processing cost.

### Replay on Resume

When you call `restore(ckpt)` and start iterating, the Engine restores the
saved state and replays inflight chunks through the full operator pipeline:

```
Checkpoint
  │
  │  saved: inflight chunks + replay cursor per lane
  │         + WorkSource state per lane
  │         + epoch boundaries (non-monotonic only)
  v
┌──────────────────────────────────────────────────────┐
│ Engine source stream                                  │
│                                                       │
│  Phase 1: yield restored inflight chunks (by chunk ID)│
│  Phase 2: fetch new chunks from restored WorkSource   │
└────────────────────────┬─────────────────────────────┘
                         │
                         v  sample pointers
┌─── Operator pipeline (deterministic replay) ─────────┐
│  fetch → tokenize → pack → ensure_mixture → ...       │
└────────────────────────┬─────────────────────────────┘
                         │
                         v  records
┌─── ReplayFilter ─────────────────────────────────────┐
│  Per lane: drop until cursor == target, then emit      │
└────────────────────────┬─────────────────────────────┘
                         │
                         v  new records only
                   Training loop
```

1. **Restore.**  Inflight WorkChunk objects are deserialized and placed
   back into the inflight set.  Each lane's WorkSource state is restored
   so that subsequent `next_chunk()` calls resume from the right position.

2. **Replay inflight chunks.**  The Engine yields the restored inflight
   chunks first, before fetching any new ones.  For non-monotonic
   pipelines, flush sentinels are re-injected at the stored epoch
   boundary positions so that accumulators flush at exactly the same
   points as the original run.  These chunks (and sentinels) flow through
   the full operator pipeline — fetch, tokenize, pack, batch — just as
   they did originally.  Because the pipeline is deterministic, the
   replay produces the exact same sequence of output records, including
   any stateful accumulator state (e.g., packing bins) that gets rebuilt
   as a side effect of processing the same inputs.

3. **Filter the prefix.**  The ReplayFilter drops the already-consumed
   prefix so the training loop only sees new records.

(the-replayfilter)=
#### The ReplayFilter

The ReplayFilter is an operator that Zephon **automatically inserts** into
every pipeline during compilation. You never add it yourself.  If the
pipeline contains a `batch` operator, the ReplayFilter is placed
immediately before it; otherwise it is appended as the tail operator.  You
can see it in the {py:meth}`~zephon.Pipeline.explain` output.  Its
name is derived from the adjacent operator — for example,
`batch_replay_filter` when placed before a `batch` operator:

```
Stage[0] place=local runner=threads cap=8 mode=microbatches
  fetch@p4 -[in_q=16]-> tokenize@p4
  --[stage_out=16]-->
Stage[1] place=local runner=inline cap=1 mode=stream_items
  ensure_mixture@p1 -> batch_replay_filter@p1 -> batch@p1
  ==[final_prefetch=3]==> pipeline_end
```

On a fresh run (no checkpoint), the ReplayFilter is a no-op — it passes
every record through unchanged.  It only activates on resume.

This placement immediately before `Batch` is what makes
`Batch(drop_last=True)` the only supported stalling special case today.
Because Batch only emits complete batches, the consumer-visible checkpoint
cut is between full batches.  On resume, ReplayFilter drops the replay
prefix before batching, so Batch rebuilds the suffix from an empty buffer
at the same batch boundary.  No extra Batch state needs to be
checkpointed.
This remains timing-safe even if the pump thread runs far ahead of the
consumer: the checkpoint cut is defined by delivered full batches, and the
replay prefix is removed before Batch ever rebuilds its suffix state.

When the ReplayFilter drops a record during the replay prefix, it emits
**tombstones** for that record (via `tombstones_for_record`).  This is
essential for correctness: without these tombstones, per-offset completion
tracking would never close offsets for replayed records, and their chunks
could never be evicted.

At checkpoint time, the Engine records the **replay cursor** per lane, i.e., a
{py:class}`~zephon.types.SampleCursor` identifying the most
recently delivered record.  A SampleCursor is a tuple of
`(chunk_id, chunk_offset, lineage, sample_id)` that uniquely and
deterministically identifies every record within a lane.  Because the
cursor is a unique identity rather than an ordinal position, this works
regardless of whether the pipeline preserves cursor order.  On resume, the
ReplayFilter reads these saved cursors and uses them as replay targets:

- For each lane, drop every record until the record whose cursor
  **equals** the saved sentinel.  Drop the sentinel itself too.
- Then emit everything that follows.

The filter uses **equality** matching (`==`), not a threshold (`<=`).
Consider a pipeline with shuffling where the tail output
for a lane arrives in non-monotonic cursor order:

```
Original run:   C2  C1  C3  C4  C5  ...
                         ^^ checkpoint taken here
                         saved sentinel = C3
```

A threshold rule ("drop while cursor <= C3") would drop C1, C2, *and*
C3 — but C4 and C5 were never consumed, so that is correct.  However, it
would *also* drop any future record with a cursor below C3, even if that
record was never consumed in the original run.  With equality matching,
the filter simply waits for C3 to reappear in the deterministic replay,
drops everything up to and including C3, and then forwards the rest.  This
is correct regardless of output ordering.


### Cross-Chunk Packing and Shuffling

Packing and shuffling interact with checkpointing through the
[epoch-based eviction model](#epoch-based-eviction) described above.  Safe
eviction of a chunk requires two conditions: all samples delivered
([per-offset tracking](#per-offset-tracking)) and no accumulator influence
remaining ([flush sentinels](#flush-sentinels-and-epoch-boundaries)).
A packed record that combines fragments from chunks 5 and 7 carries
contributor references for both; chunk 5 is not complete until its last
offset closes, even if all of chunk 7 is done.  For monotone pipelines
the second condition is trivially met — order-preserving accumulators have
no cross-invocation history, so replay reconstructs the same state
regardless of prior eviction.  Non-monotonic accumulators require the
epoch flush to make this verifiable.
[Cursor pinning](#cursor-pinning) adds a further constraint: the replay
cursor's chunk is never evicted even if both conditions are met, preventing
the edge case where a cross-chunk packed record's delivery completes an
earlier chunk that the cursor references.

Together these guarantee **replay correctness**: flush sentinels ensure the
accumulator starts each epoch from the same clean state as the original
run, all inflight chunks are preserved, so replay produces identical
output and the ReplayFilter finds its target cursor.

```{warning}
**Known limitation: cross-lane shuffle determinism with multi-lane
workers.**  When a single worker serves multiple lanes, the internal
round-robin that interleaves records from those lanes is not restored on
resume.  For most accumulators (packing, mixture correction) this is
irrelevant because they are keyed per-lane.  However, the shuffle buffer
uses a shared `CountingAccumulator` whose microbatch composition — and
therefore `batch_seed` — can differ after resume, producing a different
shuffle permutation within the buffer window.

In a `shuffle → pack` pipeline this can break replay: different shuffle
order → different bin composition → `pack_meta` produces different output
cursors → the ReplayFilter's target cursor never appears.  The typical
`pack → shuffle` ordering is safe because shuffle preserves existing
cursors (it only reorders), so the ReplayFilter always finds its target
regardless of permutation.  We need to fix this.
```

Operators that **buffer samples across chunk boundaries** must cooperate
with the eviction protocol.  Specifically, they must track which base
offsets their buffered and emitted records derive from via contributor
metadata, so the Engine does not evict a chunk while buffered samples from
that chunk still exist.  Zephon's built-in `pack_sequences`, `shuffle`,
and `ensure_mixture` operators satisfy this.  A custom operator that
buffers across chunks must propagate contributor metadata in the same
way — the requirements are:

- Set `preserves_cursor_order=False` in `OpTraits` if the operator
  reorders records.
- Propagate or set `is_last_child` on contributor metadata.
- Emit tombstones for dropped final fragments.
- Implement `flush(reset=True)` to fully reset accumulator
  state (see [Accumulators and Operators](accumulators_operators.md)).
  Do not rely on intentional stalling unless your operator is
  `Batch(drop_last=True)` or a future replay-capsule mechanism exists.
  After a mid-stream flush, your accumulator must no longer report
  pending data; Zephon treats any remaining pending state as a contract
  violation and raises.

See [Sample Lifecycle](sample_lifecycle.md) for the full
contract.

### Multi-Rank Aggregation

In distributed training, each rank's Engine owns a subset of lanes.  A
complete checkpoint must contain the state for *all* lanes.  Zephon uses a
**leader-follower protocol** to merge per-rank states:

```
Rank 0 (leader)          Shared FS            Rank 1 (follower)
     |                      |                       |
     |-- publish round ID ->|                       |
     |                      |<-- read round ID -----|
     |-- write state_r0 --->|                       |
     |                      |<--- write state_r1 ---|
     |-- poll: all lanes? ->|                       |
     |-- merge + write ---->|                       |
     |                      |<---- poll merged? ----|
     |                      |---- read merged ----->|
     |-- cleanup ---------->|<------ cleanup -------|
```

1. The leader (global rank 0) publishes a round ID to the aggregation
   directory.
2. All ranks write their local state as a JSON file, tagged with the
   round ID.
3. The leader polls until it sees state files covering all
   `canonical_replicas` lanes, then merges them and writes the merged
   result.
4. Follower ranks poll until the merged file appears, then read and
   return it.
5. Both leader and followers clean up their own local state files.  The
   leader also removes the round file and the merged file from the
   *previous* checkpoint round (if any).  The current round's merged file
   persists until the next checkpoint, so late-arriving followers can still
   read it.

For cloud storage (S3, GCS), the protocol uses exponential backoff
(capped at 5 seconds) when polling.
