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

{py:meth}`~zephon.api.Pipeline.checkpoint` returns a plain Python dict
(JSON-serializable) that captures the full pipeline state: which chunks are
in flight, how far each lane has progressed, and where each lane's
WorkSource state sits.  You are responsible for persisting this dict
alongside your model checkpoint (e.g., write it to disk as JSON, or embed
it in your training framework's checkpoint payload).
{py:meth}`~zephon.api.Pipeline.restore` validates the checkpoint structure
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
{py:meth}`Pipeline.options() <zephon.api.Pipeline.options>`.

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
checkpoint time**, which typically a handful of chunks, regardless of how far
into training you are.

### Chunks and Inflight State

Recall that the WorkSource produces **work chunks**, i.e., fixed-size groups of
sample pointers.  The Engine requests chunks sequentially per lane and
feeds their samples through the operator pipeline.  At any given moment,
each lane has a small number of **inflight chunks**: chunks whose samples
have entered the pipeline but whose results have not all been delivered to
the training loop yet.

The checkpoint captures two things per lane:

1. **Inflight chunks** — serialized in full (sample pointers, component
   order, seed).  On restore these are deserialized directly; the
   WorkSource is never asked to regenerate them.  Only inflight chunks
   are replayed through the operator pipeline.

2. **WorkSource state** — each per-lane WorkSource serializes its internal
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

Zephon selects one of two eviction strategies during pipeline compilation,
based on operator properties:

```
Pipeline compiled
       |
  All operators preserve cursor order?
      / \
    Yes   No (packing, shuffling, ...)
     |     |
  Monotone   Per-offset
  eviction   eviction
  (faster)   (bitmap tracking)
```

The planner determines this by checking each operator's
`preserves_cursor_order` trait.  If every operator in the pipeline
preserves it, the fast path is used. Otherwise the per-offset path
kicks in.

#### Monotone Eviction (Simple Pipelines)

When the pipeline preserves cursor order — meaning samples emerge in the
same chunk-ID order they entered — eviction is straightforward.  If the
most recently delivered sample belongs to chunk *N*, then every sample from
chunks *< N* has already been delivered.  Those older chunks can be evicted
immediately.

Most pipelines without shuffling or packing qualify for this fast path.

#### Per-Offset Eviction (Packing and Shuffling)

Operators like `pack_sequences` and `shuffle` can reorder samples across
chunk boundaries.  A packed record might combine fragments from chunks 5
and 7, and chunk 5's last sample might be delivered *after* something from
chunk 7.  The simple "evict everything older than *N*" rule would
prematurely evict chunk 5.

For these pipelines, Zephon tracks completion at the **per-offset** level.
Each sample in a chunk occupies a specific offset (its position within the
chunk).  As a sample flows through the pipeline, operators may split it
into multiple **children** (e.g., a long document tokenized into several
sequences) or merge children from different offsets into a single packed
record.  An offset is only closed once *all* of its children have been
delivered.  To track this, every record carries **contributor metadata** —
a `ContributorRef` that references the base offset(s) it was derived from,
plus an `is_last_child` flag indicating whether it is the final child for
that offset.  When the last child for every offset in a chunk has been
delivered, the chunk is fully consumed and can be evicted.  See
[Sample Lifecycle](sample_lifecycle.md) for more on how samples spawn
children as they flow through operators.

```
Chunk 5 (4 offsets):  [x] [x] [ ] [x]    ← 3 of 4 offsets closed
Chunk 6 (4 offsets):  [x] [x] [x] [x]    ← all closed, but NOT evicted yet
Chunk 7 (4 offsets):  [ ] [ ] [ ] [ ]    ← none closed

→ Nothing is evicted: chunk 5 is incomplete and blocks chunk 6.

Later:
Chunk 5 (4 offsets):  [x] [x] [x] [x]    ← all closed
Chunk 6 (4 offsets):  [x] [x] [x] [x]    ← all closed

→ Chunks 5 and 6 are evicted together.
```

The Engine maintains a per-chunk bitmap tracking which offsets are closed.
When the bitmap is full, the chunk is complete.  Eviction proceeds
**from the front** — only the oldest contiguous run of completed chunks is
evicted.  This ensures that on resume, all inflight chunks form a gapless
sequence that can be replayed in order without skipping any chunks in
between.

When an operator drops all fragments for a given offset (e.g., a filter
that removes invalid samples), it emits a **tombstone** — a special record
that signals completion of that offset without carrying any training
payload.  Tombstones flow through the operator pipeline but are stripped
at the Engine's delivery boundary before reaching the training loop.  They
allow the Engine to mark the
offset as closed so the containing chunk can eventually be evicted.

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
│  Per lane: drop until cursor == sentinel, then emit   │
└────────────────────────┬─────────────────────────────┘
                         │
                         v  new records only
                   Training loop
```

1. **Restore.**  Inflight WorkChunk objects are deserialized and placed
   back into the inflight set.  Each lane's WorkSource state is restored
   so that subsequent `next_chunk()` calls resume from the right position.

2. **Replay inflight chunks.**  The Engine yields the restored inflight
   chunks first, before fetching any new ones.  These chunks flow through
   the full operator pipeline — fetch, tokenize, pack, batch — just as
   they did in the original run.  Because the pipeline is deterministic,
   the replay produces the exact same sequence of output records, including
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
can see it in the {py:meth}`~zephon.api.Pipeline.explain` output.  Its
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

At checkpoint time, the Engine records the **replay cursor** per lane, i.e., a
{py:class}`~zephon.core.constants.SampleCursor` identifying the most
recently delivered record.  A SampleCursor is a tuple of
`(chunk_id, chunk_offset, lineage, sample_id)` that uniquely and
deterministically identifies every record within a lane.  Because the
cursor is a unique identity rather than an ordinal position, this works
regardless of whether the pipeline preserves cursor order.  On resume, the
ReplayFilter reads these saved cursors and uses them as sentinels:

- For each lane, drop every record until the record whose cursor
  **equals** the saved sentinel.  Drop the sentinel itself too.
- Then emit everything that follows.

The filter uses **equality** matching (`==`), not a threshold (`<=`).
Consider a pipeline with shuffling where the tail output
for a lane arrives in non-monotone cursor order:

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

Packing and shuffling interact with checkpointing in two ways:

1. **Eviction correctness.**  The per-offset eviction path with
   contributor tracking handles cross-chunk packing correctly.  A packed
   record that combines fragments from chunks 5 and 7 carries contributor
   references for both, and chunk 5 is not evicted until its last offset
   is closed — even if chunk 7's offsets close first.

2. **Replay correctness.**  The ReplayFilter operates on the final output
   stream, after packing and shuffling.  It does not care about chunk
   boundaries or contributor tracking — it only looks at the record-level
   cursor.  Because the cursor is deterministic and unique per record per
   lane, replay deduplication works the same way regardless of how records
   were assembled.

```{warning}
When a packed record combines samples from two chunks and its delivery
completes the earlier chunk, that chunk evicts before the checkpoint is
taken.  On resume the replay cursor references an evicted chunk, which
disables the ReplayFilter for that lane — causing samples from the later
chunk that were already delivered as part of the cross-chunk packed record
to appear a second time.  This is an edge case that requires the packed
record to straddle the exact chunk boundary where eviction occurs; it does
not affect monotone-eviction pipelines or packing that stays within a single
chunk.  The issue is isolated by
`test_pack_sequences_cross_chunk_data_correctness_after_checkpoint` in
`tests/zephon/ops/test_pack_sequences_integration.py` (currently xfail).
```

Operators that **buffer samples across chunk boundaries** must cooperate
with the eviction protocol.  Specifically, they must track which base
offsets their buffered and emitted records derive from via contributor
metadata, so the Engine does not evict a chunk while buffered samples from
that chunk still exist.  Zephon's built-in `pack_sequences` and `shuffle`
operators satisfy this.  A custom operator that buffers across chunks must
propagate contributor metadata in the same way — the requirements are:

- Set `preserves_cursor_order=False` in `OpTraits` if the operator
  reorders records.
- Propagate or set `is_last_child` on contributor metadata.
- Emit tombstones for dropped final fragments.

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
