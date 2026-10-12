# WorkSources and Work Chunks

[WorkSources: The Training Curriculum](../worksources/index.md) describes a `WorkSource`
from the outside: you hand it datasets and mixture weights, and it decides what the model
sees and in what order. This page describes the contract underneath that — what the engine
actually calls, what it is allowed to assume, and how the one implementation that ships
with Zephon satisfies it.

This is the base case of the induction that [Elastic
Determinism](elastic_determinism.md) rests on. Every claim about reproducible data order
reduces, eventually, to the claim that a `WorkSource` bound to a given lane produces the
same pointers in the same order for a given seed. Everything downstream is a deterministic
function of that.

Unlike most of what this section covers, `WorkSource` and `WorkChunk` are public API —
they live in `zephon.work`, not in `zephon/_internal/`. You need them if you want a
curriculum that the built-in static mixture cannot express.

## The Contract

The engine only ever calls a handful of methods, and they fall into four groups.

**Chunk production.** `next_chunk()` returns the next {py:class}`~zephon.work.WorkChunk`
for this source's lane, or `None` when the source is exhausted. Everything else in the
interface exists to support this one method. The engine calls it on a per-lane clone, in a
loop, and it is the only method that is allowed to advance any internal position.

**Component vocabulary.** `component_ids()` publishes the source's static mixture
vocabulary: every component name the source will *ever* emit, including components that a
curriculum only activates later, mapped to a unique non-negative integer. The ids need not
be dense. The engine stamps these ids onto every sample, so publishing them as a pure
function of configuration is what keeps component ids identical across ranks, topology
changes, and resumes — the alternative, assigning ids in encounter order, would produce a
different numbering on a rank that happens to see a rare component later. Emitting a
component that was never declared raises.

**Lane binding.** Before pulling any chunks, the engine clones the source once per lane it
owns with `clone_for_lane(lane_id, canonical_replicas)`, and each clone produces only the
chunks assigned to its lane. The default implementation deep-copies, so a subclass has to
be deepcopy-safe or override the method. The clone is bound once and refuses to be cloned
again; the original instance is never used to produce chunks.

**Checkpointing.** `state_dict()` and `load_state_dict()` serialize whatever drives
`next_chunk()`. What goes in is entirely up to the implementation. The one hard
requirement is that restoring from a saved state and then calling `next_chunk()` produces
the same sequence as if the source had never been interrupted. The base class stores the
lane identity and refuses a restore whose `lane_id` or `canonical_replicas` does not
match, which is the first line of defence against the silent curriculum change that
[Checkpoint Replay and Bounded Recovery](checkpoint_replay.md) describes.

Three further hooks are optional, and the built-in source implements all of them:

`chunk_size_hint()` returns a fixed chunk size when the source has one. The engine asserts
it is present whenever the plan carries a batch-size hint, because deterministic batch
composition needs to know how many samples a chunk holds.

`requires_token_priming` and `prime()` give a source a driver-side calibration pass before
the workers are forked. The token-aware mixture uses it to measure a tokens-per-byte ratio
by actually tokenizing a sample of the data. A restore skips the measurement and keeps the
checkpointed ratios, so a tokenizer upgrade cannot silently shift the mixture mid-run.

`supports_indexing()`, `__len__()` and `sample_id_at(index)` let a source back a map-style
dataset. Together with a plan whose operators are all `indexable`, they are what
`Pipeline.to_indexable_torch_dataset()` needs; a streaming-only source leaves them raising
`NotImplementedError` and `Pipeline.is_indexable` reports `False`.

## Work Chunks

A `WorkChunk` is the unit of work that travels from the source to the engine. It is a
bundle of sample *pointers* grouped by mixture component — no payloads, nothing read from
storage yet:

```python
WorkChunk(
    components={
        "fineweb": [(0, 3, 17), (0, 3, 18), (0, 1, 42), ...],   # 11469 pointers
        "dclm":    [(1, 0, 5),  (1, 2, 99), ...],               #  4915 pointers
    },
    seed=42,
)
```

Mapping insertion order is the stable tie-breaker anywhere behaviour depends on component
order, which is why the source builds the dict in a fixed order rather than whatever order
the datasets were discovered in.

**Iteration order is not storage order.** Iterating a chunk yields
`(sample_id, component_name)` pairs interleaved by **Smooth Weighted Round Robin**: the
emitter tracks a per-component deficit and always emits from whichever component is
furthest behind its target share. The point is that the mixture holds not just across the
chunk as a whole but approximately across every prefix of it, so a consumer that stops
early — or a chunk that is only partly delivered before a checkpoint — still sees roughly
the right proportions. The alternative mode, `weighted_random`, samples a component per
draw instead and gets the ratio right only in expectation.

The traversal is described by {py:class}`~zephon.work.MixtureReadConfig`: the mode, a
seed, whether to shuffle within a component, and whether to materialize the whole order up
front instead of streaming it. The engine iterates chunks with the defaults — smooth
weighted round robin, the chunk's own seed, components in their stored order, streamed
rather than precomputed.

**Why components survive at all.** Grouping pointers by component, rather than handing the
engine one flat list, is what lets `ensure_mixture` know which component a sample came
from after tokenization and packing have destroyed every other trace of its origin. The
chunk's `mixture` property reports the ratios its own counts imply; a token-aware source
additionally stamps `target_mixture` with the user's *token* mixture, because its chunks
deliberately carry a different sample composition — components with longer documents
contribute fewer pointers for the same token share.

Chunks are also the granularity at which the engine tracks progress, which is what keeps
checkpoints small: only the chunks currently in flight are saved and replayed, never the
whole history. [Chunks Are The Unit Of
State](records_and_sample_lifecycles.md#chunks-are-the-unit-of-state) covers that side of
it.

## Filling A Chunk

{py:class}`~zephon.work.StaticMixtureWorkSource` is the only implementation that ships
with Zephon. Filling a chunk is two steps: decide how many pointers each component
contributes, then pull that many from each component's cursor.

**Quota allocation** is the first step, and the default strategy is incremental rather
than fixed. Each component carries an accumulator; every chunk adds `weight * chunk_size`
to it, takes the integer part as that component's base quota, and leaves the fractional
remainder behind. A largest-remainder correction then forces the quotas to sum to exactly
`chunk_size`, and — the part that matters — each correction is debited back into the
accumulator, so a component that got rounded up this chunk is correspondingly behind on
the next one. Accumulators may go briefly negative after a `+1` correction and recover as
the next chunk's share is added; negative accumulators sort last in the correction, which
prevents the same component being rounded up twice in a row.

The effect is that the mixture is exact in the limit rather than per chunk. A component
with weight 0.001 and `chunk_size=16384` draws 16 pointers from most chunks and 17 from
some, instead of being rounded to a fixed 16 forever and drifting from its target.

Two other strategies exist. A **token-aware** strategy replaces sample quotas with token
quotas when the source is built with a `TokenEstimation`, so the weights describe the
share of *tokens* each component contributes rather than the share of documents. A
**legacy fixed** strategy computes quotas once at construction, gives every component at
least one slot per chunk, and distributes the remainder by largest remainder; it exists so
that checkpoints written before the accumulator strategy still restore.

**Per-dataset cursors** are the second step. Each dataset has a cursor that walks a
deterministic traversal of its shards, and the chunk pulls `quota[component]` pointers
from each. The cursor is where the three shuffle levels described in [Shuffling and
Repeating Samples](../worksources/shuffling_and_repeating_samples.md) are applied — shard
order, within-shard offsets, then a block-wise mosaic over the flattened sequence — each
seeded from a different derivation of the run seed so the three permutations do not
correlate with one another.

What the cursor deliberately does *not* do is mix datasets. Each cursor produces its own
internally shuffled sequence, and the interleaving across components happens later, when
the engine reads the chunk. Keeping those two concerns apart is why the mixture ratio is a
property of chunk traversal rather than something baked into a shared permutation.

The cursor streams shard by shard and generates within-shard permutations lazily, so
memory is proportional to the largest shard rather than to the dataset. With block shuffle
active, a block-sized buffer is added on top. Nothing anywhere scales with the total
sample count, which is the property that lets a run start against a multi-billion-sample
catalogue without a materialization step.

## Chunk-To-Lane Routing

Each lane's clone has to work out which chunks are *its* chunks. It does this without
asking anyone: the clone enumerates the **global** chunk schedule — the same sequence
every other lane enumerates — assigns each global chunk index to a lane, keeps the chunks
that map to its own lane, and discards the rest.

```python
# The shape of StaticMixtureWorkSource.next_chunk()
while True:
    chunk = self._next_chunk()            # deterministic global schedule
    g = self._global_chunk_index
    self._global_chunk_index += 1
    if self._lane_for_chunk(g) != self._lane:
        continue                          # belongs to another lane; drop it
    return chunk
```

Compute-everywhere-then-discard sounds wasteful, and it would be if chunks held data. They
hold pointers, so the discarded work is a quota computation and some cursor arithmetic.
What it buys is that **no rank ever talks to another rank to decide what to read**. The
schedule is a pure function of the seed and the lane assignment, so every rank derives the
same answer locally. Coordination happens only at checkpoint time, through the aggregation
directory.

**The mapping is permuted, not modular.** The obvious assignment is
`g % canonical_replicas`, and it is available as `lane_assignment="modulo"`, but it is not
the default, because it resonates badly with the allocator. A rare component is emitted
roughly every `1 / weight` chunks, and when that period shares a factor with
`canonical_replicas` the component lands on only `gcd(period, canonical_replicas)` of the
lanes — the rest never see it at all. A downstream gate waiting on that component, such as
a strict `ensure_mixture`, then stalls on the starved lanes.

`lane_assignment="permute"`, the default, routes each block of `canonical_replicas`
consecutive chunks through a seeded permutation of the lanes:

```
canonical_replicas = 4, modulo:          canonical_replicas = 4, permute:

  g:    0  1  2  3  4  5  6  7            g:    0  1  2  3  4  5  6  7
  lane: 0  1  2  3  0  1  2  3            lane: 2  0  3  1  1  3  0  2
        └── block 0 ──┘└─ block 1 ─┘            └── block 0 ──┘└─ block 1 ─┘
```

Every block still hands each lane exactly one chunk, so load balance and the exactly-once
property are untouched, and the permutation is a pure function of `(seed, block)`, so
replay is unaffected. A periodic emitter now sprays the lanes roughly uniformly instead of
hitting the same few. The block seed composes integers rather than hashing a tuple,
because `hash()` is not stable across processes unless `PYTHONHASHSEED` is pinned — and a
data order that depends on an environment variable is not a data order.

## Why The Schedule Is Not Pre-Computed

Most distributed loaders materialize the full sample order up front: build a permutation
of every index, shard it across ranks, iterate. That works when the dataset size and the
rank count are both known before training starts, and it makes "skip to 73% of the epoch"
a one-line operation.

A `WorkSource` is a streaming generator instead. It produces chunks on demand and never
holds the whole sequence, and three things follow from that.

A source does not have to be finite. The built-in one can repeat its catalogue for a fixed
number of passes or indefinitely, and a source that grows while training runs is
expressible without changing anything else.

Changing the rank count does not require recomputing a global permutation. The lane
abstraction absorbs it — the per-lane sequences are unchanged, and only the question of
which group owns which lane has a new answer.

Memory does not scale with the sample count, per the cursor model above.

The cost is the operation that a materialized permutation makes trivial. There is no
global index to seek to, so progress is tracked through opaque cursors inside the source,
and the only way to arrive at a position is to have been handed its checkpoint. That is a
deliberate trade: the positions you actually want to resume from are the ones you
checkpointed, and those restore in time independent of how far into training they are.
