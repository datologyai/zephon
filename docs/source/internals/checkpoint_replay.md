# Checkpoint Replay and Bounded Recovery

[Checkpointing and Resuming Pipelines](../pipelines/checkpointing_and_resuming.md) says
that a checkpoint records which work chunks were finished and which were still in
progress, and that a restore rebuilds operator state by re-processing a bounded number of
recent chunks. This page is about how Zephon works out which chunks those are, which is
the trickiest part of the whole system to get right.

**Why there is no sample index.** A map-style loader checkpoints by writing down an index,
and Zephon cannot, because a `Pipeline` that splits documents, packs fragments and
reorders records has no index to write down. Progress is spread across in-flight operator
state, partially filled bins, and records that have been reordered relative to their
inputs. We could replay the whole `Pipeline` from the start on every resume, which would
be correct, but crashing at step 100,000 would then mean redoing 100,000 steps of data
processing before you get one new batch.

<figure class="zephon-post zephon-embed">
  <iframe src="../_static/embeds/nonindexable-packing.html" height="371" loading="lazy"
          title="One document, three output IDs"></iframe>
  <figcaption>
    <strong>One document, three output IDs.</strong> Four documents are tokenized, split,
    and first-fit packed. Document A ends up spread across three different packed outputs,
    C overtakes B on the way through, and when the input runs out two bins are still part
    full. There is no index into that, and no way to reconstruct it other than replaying
    what came before.
  </figcaption>
</figure>

So Zephon sits in between. What we aim for is to **replay only the chunks that were still
in flight at checkpoint time**, which is a handful of them regardless of how far into
training you are.

**What a checkpoint holds.** Three things, per lane:

- Inflight chunks, serialized in full: their sample pointers, component order, and seed.
  On restore these are deserialized directly and the `WorkSource` is never asked to
  regenerate them. They are the only chunks that get replayed through the operator
  pipeline.
- Epoch boundaries, for pipelines that need them: the list of sentinel chunk IDs marking
  flush points. This can include a trailing boundary just past the last inflight chunk,
  which preserves the flush point between replayed data and newly fetched data. On restore
  the sentinels are re-injected at these positions so accumulators flush where they
  flushed originally.
- WorkSource state, so `next_chunk()` can resume producing *new* chunks from the right
  place. The `WorkSource` is not replayed and not re-advanced, because `load_state_dict()`
  restores its position directly.

What goes into that last one is up to the implementation. For `StaticMixtureWorkSource`
the sample sequence is fully determined by the datasets, the seed, the shuffle knobs and
the lane, so the state is a compact set of per-dataset cursor positions plus mixture
bookkeeping, and restoring it is instantaneous no matter how far into training you are.
[WorkSources and Work Chunks](worksources_and_work_chunks.md) covers what those cursors
are and what the rest of the contract requires of a source you write yourself.

Alongside these sit the replay cursor per lane, the lane progress pointers, and the
round-robin position discussed in
{ref}`Elastic Determinism <what-actually-survives-a-topology-change>`.

One thing is deliberately absent. The per-offset completion bitmaps are runtime-only, and
Zephon rebuilds them as the replayed chunks flow through the `Pipeline` and re-trigger
contributor tracking. That keeps your checkpoints small, at the cost of making restore
time proportional to the number of inflight chunks.

**Eviction.** What keeps the inflight set small is eviction. Once every sample derived
from a chunk has been delivered, the chunk leaves the set, and a checkpoint taken
afterwards neither stores nor replays it.

```
Lane 0 inflight chunks over time:

  step 100:  [ chunk 3 | chunk 4 | chunk 5 ]
  step 200:  [ chunk 4 | chunk 5 | chunk 6 ]    chunk 3 evicted
  step 300:  [ chunk 5 | chunk 6 | chunk 7 ]    chunk 4 evicted
```

Without eviction the inflight set grows without bound and every resume replays everything,
so the only real difficulty is timing: evict too early and replay loses data it needed,
evict too late and your checkpoints bloat.

Zephon picks a strategy at compile time by looking at every operator's
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

**Monotone eviction.** When every operator preserves cursor order, records come out in the
same chunk order they went in. If the most recently delivered record belongs to chunk *N*,
then everything from chunks below *N* has already gone out and those chunks evict
immediately, so a simple watermark is all we need. Most pipelines without shuffling or
packing take this path.

**Epoch-based eviction.** Packing, shuffling and mixture correction break that watermark
in two ways, and the second is much worse than the first.

The obvious problem is reordering. A packed record can combine fragments from chunks 5 and
7, and chunk 5's last sample can be delivered after something from chunk 7, so a rule of
"evict everything below the most recent chunk" would drop chunk 5 while part of it is
still in flight. Per-offset tracking fixes that one.

The deeper problem is **accumulator history dependence**. The packer's bin state, meaning
which bins exist and how full they are, depends on the entire history of what it has seen
rather than just on what is currently in the bins. After a chunk's samples have all left
the bins, their influence on where subsequent samples land persists. Two packers starting
from different states and fed identical input may *never* converge. With two bins of
capacity 10 and an input stream of `[3, 3, 3, …]`, starting from `(10, 10)` versus
`(10, 5)` gives the same periodic emission cycle, permanently phase-shifted.

That leaves us with a direct conflict. Eviction wants to discard consumed chunks to free
memory, while deterministic replay wants to reconstruct the accumulator state the original
run had, and that state was shaped by inputs from exactly the chunks eviction wants to
discard. Replaying from a checkpoint would start the packer empty instead of mid-stream
and produce different packed records, and the replay filter's target cursor might never
appear at all. The same argument applies to mixture correction, whose deficit state
depends on emission history, and to shuffling, whose buffers span chunk boundaries.

Our resolution is the **flush sentinel**. Every `flush_every_k_chunks` chunks per lane,
Zephon injects one that forces the history-dependent accumulators to emit everything and
reset to a clean state, as described in
{ref}`Operators and Runners <flush-sentinels-and-stalling>`. Sentinels divide the stream
into **epochs**, and each epoch starts from a state that replay can reproduce without any
of the preceding chunks:

```
Epoch 0          Epoch 1          Epoch 2
[chunk 0, 1]  →  [chunk 2, 3]  →  [chunk 4, 5]  → ...
             ↑                ↑
          sentinel         sentinel
           (flush)          (flush)
```

[How Pipelines Run](../pipelines/how_pipelines_run.md) covers the trade that
`flush_every_k_chunks` controls. What that page leaves implicit is what a flush costs at
the operator level, since each one interrupts an accumulator mid-stride: the packer emits
partially filled bins and pays for them in padding, the shuffle buffer emits a smaller
window and randomises less, and the mixture corrector drains mid-rebalance.

On a non-monotonic `Pipeline`, setting `flush_every_k_chunks=0` is an error, because
without sentinels the accumulator state depends on unbounded history and we cannot evict
anything safely. On a monotone one, setting a positive value gets you a warning and Zephon
forces the value back to 0, since sentinels buy you nothing there and would perturb your
batch boundaries for free.

**Per-offset tracking.** Within an epoch, Zephon tracks completion per offset. Each sample
occupies a position in its chunk, and operators may split it into several children or
merge children from different offsets into one packed record, so an offset only closes
once all of its children have been delivered. Zephon learns that from the `is_last_child`
flags on contributor metadata, and from tombstones when a filter drops the last child
standing (see {ref}`Records and Sample Lifecycles <contributors-and-tombstones>`). A
bitmap per chunk records which offsets are closed, with a popcount alongside so that "is
this chunk complete?" stays O(1).

```
Chunk 5 (4 offsets):  [x] [x] [ ] [x]    3 of 4 closed
Chunk 6 (4 offsets):  [x] [x] [x] [x]    all closed, but not evicted
Chunk 7 (4 offsets):  [ ] [ ] [ ] [ ]    none closed

→ Nothing is evicted: 5 and 6 are in the same epoch, and 5 is incomplete.
```

**Atomic per-epoch eviction.** Chunks within an epoch share accumulator history, since
where chunk 1's samples landed in the bins depends on what chunk 0 put there, so evicting
chunk 0 while chunk 1 is still inflight would break replay for exactly the reason above.
An epoch therefore evicts **atomically**: all of its chunks, or none of them.

Zephon walks a lane's recorded boundaries in ascending order and evicts contiguously from
the bottom, stopping at the first epoch that is incomplete. The epoch boundaries are what
make this safe across epochs, because evicting epoch 0 while epoch 1 is still running is
fine precisely when epoch 1's accumulators started empty after the sentinel flush.

```
Epoch 0 (chunks 0–1):  all offsets closed   → evict together
Epoch 1 (chunks 2–3):  still in progress    → keep
Epoch 2 (chunks 4–5):  not started          → keep
```

Only a lane's own recorded boundaries authorise eviction for that lane. The restriction
matters because sentinels are per-lane, so a boundary established when a different lane
flushed says nothing about this lane's accumulator history. Until a lane has recorded a
boundary of its own, Zephon retains its entire open epoch even when every offset in it is
closed, which means your early chunks stay inflight until the first sentinel arrives. We
consider that a fair trade of a little memory for an eviction rule with no special cases,
and the full `Pipeline` drain at end of stream handles whatever is left.

Cursor pinning adds the last constraint. The replay cursor is the identity of the last
delivered record, and it has to point into a chunk that is still inflight or the replay
filter cannot find its target on resume. If the cursor sits in an epoch that is otherwise
eligible, Zephon stops there and leaves it alone, which resolves on its own within a few
deliveries, as soon as a post-sentinel record goes out.

**Replay on resume.** When you restore and then iterate, Zephon runs the saved state back
through the full `Pipeline`:

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
┌─── Replay filter ────────────────────────────────────┐
│  Per lane: drop until cursor == target, then emit     │
└────────────────────────┬─────────────────────────────┘
                         v  new records only
                   Training loop
```

First the inflight chunks are deserialized back into the inflight set and each lane's
`WorkSource` state is restored. Zephon then yields those chunks before fetching anything
new, re-injecting flush sentinels at the stored boundary positions so that the
accumulators flush exactly where they did before. Those chunks flow through the whole
`Pipeline`, through fetch, tokenize, pack and batch, just as they originally did, and
because the `Pipeline` is deterministic they produce the same records and rebuild the same
accumulator state as a side effect. Finally the replay filter drops the prefix your
training loop has already consumed.

**The replay filter.** Zephon inserts a replay filter into every `Pipeline` for you, so
this is never something you add yourself. If the `Pipeline` has a `batch` operator the
filter goes immediately before it, otherwise it is appended at the tail, and it takes its
name from the operator it sits next to. You can see it in `Pipeline.explain()`:

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

Putting it before `batch` is what makes `batch(drop_last=True)` the one operator allowed
to stall at an epoch boundary. Because batching only emits complete batches, the
consumer-visible checkpoint cut falls between whole batches, so on resume the prefix is
dropped before batching and batching rebuilds the suffix from an empty buffer at the same
boundary. No extra batch state has to be checkpointed, and that holds even when the pump
thread has run far ahead of the consumer, because the cut is defined by delivered batches
rather than by how much work the pump had done.

When the filter drops a record during the prefix it emits **tombstones** for it, which is
not optional: without them, per-offset tracking would never close the offsets belonging to
replayed records and their chunks could never be evicted.

The match is on equality, not on a threshold. At checkpoint time Zephon records, per lane,
the `SampleCursor` of the most recently delivered record, and on resume the filter drops
records until it sees a cursor **equal** to that target, drops the target too, and then
forwards everything after it.

We use equality rather than `<=` because the tail of a `Pipeline` with shuffling does not
arrive in cursor order:

```
Original run:   C2  C1  C3  C4  C5  ...
                        ↑ checkpoint here, saved target = C3
```

A threshold rule would drop C1, C2 and C3, which is correct, but it would also drop any
later record whose cursor happens to fall below C3, and you never consumed those. The
equality rule waits for C3 to reappear in the deterministic replay and forwards everything
after it, which is correct whatever the output ordering looks like, because a cursor is a
unique identity rather than an ordinal position.

**What a custom operator owes replay.** If your operator buffers across chunk boundaries
it has to cooperate with all of the above, or Zephon will evict a chunk whose data your
operator is still holding. The obligations are spelled out on the two pages before this
one; gathered together, they are:

- Set `preserves_cursor_order=False` in `OpTraits` if your operator reorders anything.
- Propagate contributor metadata, and set `is_last_child` correctly.
- Emit tombstones for dropped final fragments.
- Implement `flush(reset=True, lane_id=...)` so that it fully resets that lane's state.
  After a mid-stream flush your accumulator has to report no pending data for that lane,
  and the runner treats anything else as a contract violation and raises.
- Do not plan on stalling. We support it for `batch(drop_last=True)` and nothing else.

**Multi-rank aggregation.** Each rank owns a subset of lanes, but a complete checkpoint
has to cover all of them, so the ranks merge their states through shared coordination
storage using a leader-follower protocol. That storage is either a shared directory
(`aggregate_dir`, a local or shared filesystem or object storage) or a key-value store
(`aggregate_store`, where `"torch://"` uses the TCPStore that torchrun already runs). The
protocol is the same over both. The leader is global rank 0, DataLoader worker 0.

```
Leader                     shared storage                  Followers
  │                            │                               │
  │── announce round id ──────>│                               │
  │                            │<────── read round id ─────────│
  │── write own state ────────>│                               │
  │                            │<────── write states ──────────│
  │── wait: every rank covers  │                               │
  │   all its lanes? ─────────>│                               │
  │── close round              │                               │
  │   (delete announcement) ──>│                               │
  │── merge, publish merged ──>│                               │
  │── delete own state ───────>│                               │
  │                            │<────── wait for merged ───────│
  │                            │─────── read merged ──────────>│
  │                            │<── delete own state, ack ─────│
  │── wait for every ack ─────>│                               │
  │── delete acks, previous    │                               │
  │   round's merged ─────────>│                               │
```

The leader's wait condition is stricter than it might look. It is not enough for the
lanes reported across all ranks to add up to every one of the `canonical_replicas`: each
rank has to report every lane its DP group owns, so that a model-parallel peer in the same
DP group can't stand in for a DataLoader worker that is running late. The leader also
checks that every state it receives agrees with its own lane mapping, and raises on a
mismatch rather than waiting for it to resolve itself.

`aggregate_timeout_s` is one budget for the whole round, not a limit per wait. If it runs
out while the leader is still collecting states, the error tells you which ranks and lanes
are missing, which is usually enough to find the rank that forgot to call `checkpoint()`.
If the leader fails before publishing the merged result, it publishes a failure notice
instead, so the followers fail straight away rather than waiting out the timeout.

Because the leader waits for every ack, nobody is still reading the merged result when
`aggregate()` returns. We keep it around anyway as a marker that the round is closed, and
delete it once the next round has been fully acknowledged or the `Pipeline` shuts down; if
the acks time out, it stays for the followers that haven't read it yet. States are written
with msgpack and zstd by default. Against object storage the polling backs off
exponentially, capped at five seconds, while the TCPStore uses its own blocking waits.
