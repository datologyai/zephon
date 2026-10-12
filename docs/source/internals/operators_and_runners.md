# Operators and Runners

[How Pipelines Run](../pipelines/how_pipelines_run.md) makes a claim in passing that is
worth taking seriously: an operator running across sixteen workers makes exactly the same
decisions it would have made on one. This page explains the two pieces of machinery that
deliver that, and what they ask of an operator in return.

**The Operator Contract.** An operator is a unit of transformation (tokenizing, packing,
batching, shuffling) and it neither knows nor cares how it is executed. Five methods make
up the contract, of which three matter most.

`accumulator(*, deterministic, ctx)` returns the {ref}`accumulator <accumulators>` that
decides how incoming elements are grouped into micro-batches. The `deterministic` flag
comes from the runner and controls whether time-based flushing is allowed.

`process_many(elems)` is the workhorse, and the part that runs in parallel. It takes a
micro-batch and returns a list of the *same* element types. The elements are
`SampleRecord` objects, or `SampleBatch` objects downstream of `batch()`. To transform a
sample you rebuild its `payload` while preserving its `meta`, which carries the identity
and ordering that [Records and Sample Lifecycles](records_and_sample_lifecycles.md)
describes. Returning a bare dict or a raw payload gets you a
`TypeError: Unsupported element type`.

`traits()` declares the `OpTraits` that guide compilation and scheduling, including the
all-important `preserves_cursor_order`.

Two more round it out. `setup(ctx)` runs once before any data arrives, receiving an
`OpContext` with runtime services and plan metadata. `process_one(elem)` handles a single
element and by default just calls `process_many` with a one-element list. Override it only
if you have a genuinely faster scalar path.

**The invariant is that `process_many` must be stateless across calls.** It must not
depend on anything left over from a previous invocation. Every piece of cross-invocation
state belongs in the accumulator, including buffers, counters, and open packing bins. This
is the whole reason it is safe to hand `process_many` calls to an arbitrary number of
workers. Each call is independent, so results can be reordered back into input order
without changing anything.

How the work splits between the accumulator and the workers varies a lot. For tokenization
or fetching, the accumulator does nothing but group elements into fixed-size micro-batches
and all the real work happens in `process_many` across the pool. For operators whose core
logic is inherently serial, such as packing, mixture correction, and shuffling, the
accumulator does the heavy lifting and `process_many` is currently a passthrough that
returns its input unchanged. Some of that work (assembling packed payloads, for instance)
is stateless in principle and could move onto the workers. It has not, and that is a known
throughput limitation rather than a design requirement. What *is* a design requirement is
that any logic needing to see elements in stream order lives in the accumulator.

**Micro-Batches.** When an operator runs with several workers, the runner does not hand
out elements one at a time. It groups them into micro-batches processed together in a
single `process_many` call. That amortises dispatch overhead and lets operators use
batch-level optimisations, such as a tokenizer encoding sixty-four texts in one call
rather than sixty-four calls.

The sizes are per-operator and chosen for the work involved: fetching and tokenization use
64, text decoding uses 128, a `map_transform` uses 64 unless you say otherwise, and
`batch` uses whatever `microbatch_size` you asked for. With four workers on a tokenizer,
four micro-batches of 64 are in flight at once. That is 256 samples being tokenized
concurrently, with each call still seeing a sensible 64.

:::{note}
**Two kinds of batch.** Micro-batches here are the internal groups of elements
dispatched to operator workers. They are not the training batches that `batch()`
produces, and the two sizes have nothing to do with each other. The name collision is
unfortunate and ours.
:::

Who decides the boundaries? That is the accumulator's entire job.

(accumulators)=

**Accumulators.** An accumulator is a small, single-threaded object sitting between an
operator's input stream and its worker pool. It receives elements from upstream in order,
decides when enough of them are ready, and hands a micro-batch off to be dispatched.

The guarantee is that it runs **serially**. The runner never calls into it from two
threads at once, and elements always arrive in upstream order. In the current runners this
is a dedicated pump thread per operator, but the architectural contract is just the
serialness. Workers run the stateless `process_many`, and the accumulator decides what
each worker sees.

```
Upstream elements
       │
       v
┌──────────────────────────────────────────────────────────┐
│  Accumulator (serial)                                     │
│                                                           │
│  push_many(elems)                                         │
│      ├── not enough yet  → buffer internally              │
│      └── ready           → emit micro-batch               │
│                                                           │
│  Runner tags the micro-batch with a sequence number        │
└─────────────────────────┬─────────────────────────────────┘
                          v
          ┌───────────────────────────┐
          │        Worker pool         │
          │   process_many()  × N      │
          └─────────────┬─────────────┘
                        │  results, possibly out of order
                        v
            Reorder by sequence number (deterministic mode)
                        │
                        v
                   Downstream
```

**Why this buys determinism.** Take sequence packing. The packer keeps open bins, which
are partially filled sequences waiting for short samples to complete them, and which bin a
given sequence lands in depends on what is already in the bins. If each worker kept its
own bins, bin assignment would depend on which worker happened to receive which sample,
which is to say on thread scheduling. One worker and four workers would produce different
packed sequences. Instead every bin lives in the packing accumulator, which sees elements
in stream order on one thread, so the stateful decision is always the same one. The same
argument applies to batching, mixture correction, and shuffle buffering.

The bins are also per-lane, which is what connects this to [elastic
determinism](elastic_determinism.md): samples from different lanes are never packed
together, so each lane's packing decisions are independent of how many GPUs are running.

**The interface** is four methods and a flag, defined in `zephon.ops.accumulators`:

`push_many(elems)` consumes elements from upstream, buffers them, and returns zero or more
ready micro-batches.

`flush(*, reset=False, lane_id=None)` emits what is still buffered, in one of two
contexts. At end of stream (`reset=False`, `lane_id=None`) every lane is finished and any
remaining partial micro-batches go out. Mid-stream (`reset=True`, with a `lane_id`) a
flush sentinel has arrived, and the accumulator must emit **and fully reset** that one
lane's state while leaving every other lane untouched. Flush sentinels are injected per
lane, so flushing every lane at one lane's boundary corrupts the others and breaks replay.

`has_pending_data(lane_id=None)` reports whether a flush would produce anything, which the
runner uses to avoid terminating early and to detect stalling.

`try_epoch_reset(boundary_chunk_id, lane_id)` asks whether a stalled sentinel can now be
released. The question it answers is whether the accumulator's future output is equivalent
to that of a fresh instance fed only the records still buffered. The default
implementation is conservative and says yes only when nothing is pending.

`reads_payload` is the flag. It defaults to `False`, which declares that the accumulator
only ever looks at metadata (the `lane_id`, cursors, and so on) and lets the runner leave
payloads as unresolved shared-memory handles. Set it to `True` if `push_many` inspects
payloads, and the runner resolves them first. Forget to, and you get a lazy handle where
you expected a dict.

`flush()` is deterministic in the sense that matters. Its output depends only on the
elements the accumulator has seen, never on timing.

**The built-ins.** `PassthroughAccumulator` emits whatever arrives, immediately, with no
buffering and no state, and the replay filter uses it. `CountingAccumulator` buffers per
lane and emits when a lane reaches `max_batch`, and is what most operators use: fetch,
decode, tokenize, map transforms, and batching, which uses a subclass that adds
`drop_last`. Lane keying lives in `CountingAccumulator` itself, which is what makes a
per-lane flush sentinel able to reset one lane alone.

In non-deterministic mode `CountingAccumulator` can also flush on a latency bound
(`max_latency_ms`), emitting a partial micro-batch rather than waiting for a bursty stream
to fill one. In deterministic mode that is disabled, so micro-batch boundaries depend only
on counts and never on the clock.

The specialised ones ship with the operators that need them, and you never construct one
yourself. The accumulator behind `pack_sequences` keeps per-lane bins and emits when a
bin's remaining capacity drops below the minimum useful sequence length or when too many
bins are open. The one behind `ensure_mixture` keeps per-lane buffers grouped by mixture
component and uses smooth weighted round-robin to emit in the order that best tracks the
target ratios. `shuffle` has two, picked by its `algorithm` argument: a reservoir
accumulator for the default `"streaming"`, and a block accumulator for `"block"` and
`"block_warmup"`.

**How Runners Keep The Order.** Zephon ships three runner types for local execution,
namely `inline`, `threads`, and `process`, which [How Pipelines
Run](../pipelines/how_pipelines_run.md) describes. There is also a Ray-backed `remote`
runner for stages added with `placement="remote"`. From the accumulator's point of view
they all behave identically.

When parallelism is greater than one, workers finish in whatever order they finish, and a
slow micro-batch can complete after a fast one dispatched later. In deterministic mode the
runner tags each micro-batch with a monotonically increasing **sequence number** before
dispatch, holds out-of-order completions in a reorder buffer, and forwards a result only
once every preceding sequence number has gone out. The downstream operator sees exactly
what a single-threaded execution would have produced.

In non-deterministic mode the runner skips the reordering and forwards in completion
order, trading reproducibility for latency.

Accumulator-defined boundaries plus sequence-number reordering are, together, the reason
execution parameters are
{ref}`pure performance knobs <execution-parameters-are-not-data-parameters>`.

**Data-Derived Randomness.** The shuffle buffer needs randomness, and the obvious approach
would be wrong. An RNG advanced once per micro-batch ties the permutation to micro-batch
boundaries, and those boundaries shift across a restart because replay starts from a
different point. The same data would shuffle differently after a resume.

So the shuffle derives its randomness from the **data itself**. The block algorithms hash
the cursor keys of every element in the batch together with the user's base seed. The
streaming algorithm derives a per-element salt from the cursor identity and combines it
with the seed, the lane, an emission counter, and the reservoir size to pick a
deterministic victim on each arrival. Either way the same set of elements produces the
same result regardless of how they were grouped or which restart they turn up on.

(flush-sentinels-and-stalling)=

**Flush Sentinels And Stalling.** Pipelines containing packing, shuffling, or mixture
correction use **flush sentinels** to create epoch boundaries that make checkpoint
eviction safe. Every `flush_every_k_chunks` chunks per lane, the engine injects a sentinel
that triggers `flush(reset=True, lane_id=lane)` on each accumulator as it passes.
[Checkpoint Replay and Bounded Recovery](checkpoint_replay.md) covers why. This section
covers what it means for an accumulator.

**What gets flushed.** Every accumulator, scoped to the sentinel's lane, with one
exception noted below. For history-dependent accumulators (`preserves_cursor_order=False`)
the reset is the load-bearing part. For order-preserving ones the flush is harmless, being
either a no-op or a trivial drain, and only the bookkeeping differs:

| Operator | At a sentinel | Why |
|---|---|---|
| `pack_sequences`, `pack_flat` | Flush: emit padded partial bins, reset all bin state | Bin state depends on the full input history |
| `ensure_mixture` | Flushes | Round-robin deficit state depends on emission history |
| `shuffle` | Flushes: drains the buffer | Buffer contents span chunk boundaries |
| `fetch`, `decode_text`, `tokenize`, `map_transform` | Flush, returning whatever is buffered | Harmless, because grouping is stateless |
| `batch(drop_last=False)` | Flushes: emits the partial batch | Harmless, because a partial batch is acceptable |
| `batch(drop_last=True)` | **Stalls** | Cannot emit a partial batch |
| The replay filter | Flushes, as a no-op | No state to reset |

**What "fully reset" means.** After `flush(reset=True, lane_id=lane)`, that lane must be
indistinguishable from a freshly constructed accumulator, with its buffers, counters, and
derived state cleared and every other lane untouched. For the packer that means the lane's
bins are emptied and partial bins emitted with padding. For the mixture corrector, that
lane's deficit weights are reset. This is what lets each lane's next epoch replay from a
clean start.

The contract is enforced rather than assumed. If `flush(reset=True, lane_id=lane)` returns
while `has_pending_data(lane)` is still true, the runner raises. Leaving residual
cross-boundary state would break replay silently, so it fails loudly instead.

**Stalling is deliberately narrow.** Exactly one built-in is allowed to hold a sentinel
back instead of flushing: `batch(drop_last=True)`, because it cannot emit a partial batch
without changing its output. The runner derives this from the operator instance rather
than from a trait, which is a signal that it is a special case and not an extension point.

The reason the door is shut is worth spelling out. For a general stalled operator, the
effective reset point is *later* than the recorded epoch boundary. After eviction, replay
would need the cross-boundary consumption state that determined that delayed point. Zephon
checkpoints inflight chunks, epoch boundaries, and replay cursors, but it does not
checkpoint a general operator replay capsule. An operator that both reorders and stalls
would need one.

`batch(drop_last=True)` escapes this because it preserves input order, emits only complete
batches, and has the replay filter inserted immediately before it by the planner. The
consumer-visible checkpoint cut therefore falls between complete batches, so the batch
buffer is empty at the replay cursor and nothing extra needs saving. Live execution can
still produce a cross-epoch batch like `[old, old, new]` before the sentinel is released.
That is expected and harmless, because on resume the prefix is filtered out *before*
batching and the suffix is rebuilt from an empty buffer at the same boundary.

When the runner sees a sentinel for a stalling accumulator with pending data on that lane,
it skips the flush and holds the sentinel behind the buffered data. The sentinel is
released once `try_epoch_reset(boundary, lane)` succeeds for that lane. Lanes stall
independently, so a blocked lane never holds back another lane's ready sentinel.
