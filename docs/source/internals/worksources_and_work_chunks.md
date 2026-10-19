# WorkSources and Work Chunks

[WorkSources: The Training Curriculum](../worksources/index.md) describes a `WorkSource`
from the outside. You hand it datasets and mixture weights, and it decides what your model
sees and in what order. This page describes the interface underneath that: what Zephon
actually calls, what it is allowed to assume, and how the one implementation that ships
with it satisfies that interface.

This is the base case of the induction that [Elastic Determinism](elastic_determinism.md)
rests on. Every claim we make about reproducible data order reduces, eventually, to the
claim that a `WorkSource` bound to a given lane produces the same pointers in the same
order for a given seed, and everything downstream is a deterministic function of that.

`WorkSource` and `WorkChunk` live in `zephon.work`. Unlike most of what the Internals
section describes, they are an interface you would implement against if you wanted a
curriculum that the built-in static mixture cannot express.

**The interface.** This is the whole of what Zephon currently calls on a source:

```python
class MyWorkSource(WorkSource):
    def next_chunk(self) -> WorkChunk | None: ...
    def component_ids(self) -> Mapping[str, int]: ...

    @property
    def datasets_by_id(self) -> Mapping[int, Dataset]: ...

    def clone_for_lane(self, lane_id: int, canonical_replicas: int) -> WorkSource: ...
    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, state: dict[str, Any]) -> None: ...

    # Optional below here.
    def chunk_size_hint(self) -> int | None: ...

    @property
    def requires_token_priming(self) -> bool: ...
    def prime(self, **kwargs: Any) -> None: ...

    def supports_indexing(self) -> bool: ...
    def __len__(self) -> int: ...
    def sample_id_at(self, index: int) -> SampleId: ...
```

The required ones fall into five groups:

- **Chunk production.** `next_chunk()` returns the next `WorkChunk` for this source's
  lane, or `None` when the source is exhausted. Everything else in the interface exists to
  support this one method. The engine calls it on a per-lane clone, in a loop, and it is
  the only method that is allowed to advance any internal position.
- **Component vocabulary.** `component_ids()` publishes the source's static mixture
  vocabulary: every component name the source will *ever* emit, including components that
  a curriculum only activates later, mapped to a unique non-negative integer. The ids need
  not be dense. The engine stamps these ids onto every sample, so publishing them as a
  pure function of configuration is what keeps component ids identical across ranks,
  topology changes, and resumes. The alternative, assigning ids in encounter order, would
  produce a different numbering on a rank that happens to see a rare component later.
  Emitting a component that was never declared raises.
- **Dataset catalogue.** `datasets_by_id` maps each dataset id appearing in your pointers
  to the `Dataset` that can read it. The engine reads it once at startup, builds the
  node-local shard catalogue from it, and passes it to the operators through their
  context, so this is the bridge between a pointer and the bytes it names. Fetching and
  prefetching raise without it.
- **Lane binding.** Before pulling any chunks, the engine clones the source once per lane
  it owns with `clone_for_lane(lane_id, canonical_replicas)`, and each clone produces only
  the chunks assigned to its lane. The default implementation deep-copies, so a subclass
  has to be deepcopy-safe or override the method. The clone is bound once and refuses to
  be cloned again, and the original instance is never used to produce chunks.
- **Checkpointing.** `state_dict()` and `load_state_dict()` serialize whatever drives
  `next_chunk()`. What goes in is entirely up to the implementation. The one hard
  requirement is that restoring from a saved state and then calling `next_chunk()`
  produces the same sequence as if the source had never been interrupted. The base class
  stores the lane identity and refuses a restore whose `lane_id` or `canonical_replicas`
  does not match, which is the first line of defence against the silent curriculum change
  that [Checkpoint Replay and Bounded Recovery](checkpoint_replay.md) describes.

Three further hooks are optional, and the built-in source implements all of them:

- `chunk_size_hint()` returns a fixed chunk size when the source has one. The engine
  asserts it is present whenever the plan carries a batch-size hint, because deterministic
  batch composition needs to know how many samples a chunk holds.
- `requires_token_priming` and `prime()` give a source a driver-side calibration pass
  before the workers are forked. The token-aware mixture uses it to measure a
  tokens-per-byte ratio by actually tokenizing a sample of the data. A restore skips the
  measurement and keeps the checkpointed ratios, so a tokenizer upgrade cannot silently
  shift the mixture mid-run.
- `supports_indexing()`, `__len__()` and `sample_id_at(index)` let a source back a
  map-style dataset. Together with a plan whose operators are all `indexable`, they are
  what `Pipeline.to_indexable_torch_dataset()` needs. A streaming-only source leaves
  `supports_indexing()` returning `False` and the other two raising `NotImplementedError`,
  and `Pipeline.is_indexable` reports `False`.

**Work chunks.** A `WorkChunk` is the unit of work that travels from your source into
Zephon. It is a bundle of sample pointers grouped by mixture component, with no payloads
and nothing read from storage yet:

```python
WorkChunk(
    components={
        "fineweb": [(0, 3, 17), (0, 3, 18), (0, 1, 42), ...],   # 11469 pointers
        "dclm":    [(1, 0, 5),  (1, 2, 99), ...],               #  4915 pointers
    },
    seed=42,
)
```

Wherever behaviour depends on component order, Zephon falls back on the mapping's
insertion order as a stable tie-breaker, which is why your source should build that dict
in a fixed order rather than whatever order the datasets happened to be discovered in.

**Iteration order is not storage order.** Iterating a chunk yields
`(sample_id, component_name)` pairs interleaved by **Smooth Weighted Round Robin**, which
tracks a per-component deficit and always emits from whichever component is furthest
behind its target share. We do it this way so that the mixture holds not just across the
chunk as a whole but approximately across every prefix of it, which means a consumer that
stops early, or a chunk that is only partly delivered before a checkpoint, still sees
roughly the right proportions. The alternative mode, `weighted_random`, samples a
component per draw instead and gets the ratio right only in expectation.

`MixtureReadConfig` describes the traversal: the mode, a seed, whether to shuffle within a
component, and whether to materialize the whole order up front instead of streaming it.
Zephon iterates chunks with the defaults, which are smooth weighted round robin, the
chunk's own seed, components in their stored order, and streaming rather than
precomputing.

**Why components survive at all.** Grouping pointers by component, rather than handing
Zephon one flat list, is what lets `ensure_mixture` know which component a sample came
from once tokenization and packing have destroyed every other trace of its origin. The
chunk's `mixture` property reports the ratios its own counts imply, and a token-aware
source additionally stamps `target_mixture` with your token mixture, because its chunks
deliberately carry a different sample composition: components with longer documents
contribute fewer pointers for the same token share.

Chunks are also the granularity at which Zephon tracks progress, which is what keeps your
checkpoints small, because only the chunks currently in flight are saved and replayed and
never the whole history.
{ref}`Chunks are the unit of state <chunks-are-the-unit-of-state>` covers that side of it.

**Filling a chunk.** `StaticMixtureWorkSource` is the only WorkSource implementation that ships with Zephon today and covers the common case: a fixed set of datasets mixed in fixed proportions. It fills a chunk in two steps: decide how many pointers each component
contributes, then pull that many from each component's cursor.

Quota allocation is the first step, and the default strategy is incremental rather than
fixed. Each component carries an accumulator, and every chunk adds `weight * chunk_size`
to it, takes the integer part as that component's base quota, and leaves the fractional
remainder behind. A largest-remainder correction then forces the quotas to sum to exactly
`chunk_size`. The part that matters is what happens next: each correction is debited back
into the accumulator, so a component that got rounded up this chunk is correspondingly
behind on the next one. An accumulator may go briefly negative after a `+1` correction and
recover as the next chunk's share is added, and negative accumulators sort last in the
correction so that the same component is not rounded up twice in a row.

The effect is that your mixture comes out exact in the limit rather than per chunk. A
component with weight 0.001 and `chunk_size=16384` draws 16 pointers from most chunks and
17 from some, instead of being rounded to a fixed 16 forever and drifting away from its
target.

Two other strategies exist. A **token-aware** strategy replaces sample quotas with token
quotas when you build the source with a `TokenEstimation`, so the weights describe the
share of tokens each component contributes rather than the share of documents. A **legacy
fixed** strategy computes quotas once at construction, gives every component at least one
slot per chunk, and distributes the remainder by largest remainder; it is there so that
checkpoints written before the accumulator strategy still restore.

Per-dataset cursors are the second step. Each dataset has a cursor that walks a
deterministic traversal of its shards, and the chunk pulls `quota[component]` pointers
from each one. The cursor is where the three shuffle levels described in [Shuffling and
Repeating Samples](../worksources/shuffling_and_repeating_samples.md) are applied, namely
shard order, within-shard offsets, and then a block-wise mosaic over the flattened
sequence. Each level is seeded from a different derivation of your run seed, so the three
permutations do not correlate with one another.

What the cursor deliberately does not do is mix datasets. Each cursor produces its own
internally shuffled sequence and the interleaving across components happens later, when
Zephon reads the chunk, and keeping those two concerns apart is why the mixture ratio is a
property of chunk traversal rather than something baked into a shared permutation.

The cursor streams shard by shard and generates within-shard permutations lazily, so
memory is proportional to the largest shard rather than to the dataset, with a block-sized
buffer on top when block shuffle is active. Nothing anywhere scales with the total sample
count, which is what lets you start a run against a multi-billion-sample catalogue without
a materialization step.

(chunk-to-lane-routing)=

**Chunk-to-lane routing.** Each lane's clone has to work out which chunks are its own, and
it does that without asking anyone. The clone enumerates the **global** chunk schedule,
the same sequence every other lane enumerates, assigns each global chunk index to a lane,
keeps the chunks that map to its own lane, and discards the rest.

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

Computing everywhere and then discarding sounds wasteful, and it would be if chunks held
data. They hold pointers, so the discarded work is a quota computation and some cursor
arithmetic. What it buys us is that **no rank ever talks to another rank to decide what to
read**, because the schedule is a pure function of the seed and the lane assignment, so
every rank derives the same answer locally. Coordination only happens at checkpoint time,
through the aggregation directory.

**The mapping is permuted, not modular.** The obvious assignment is
`g % canonical_replicas`, and you can have it as `lane_assignment="modulo"`, but we do not
use it by default because it resonates badly with the allocator. A rare component comes
out roughly every `1 / weight` chunks, and when that period shares a factor with
`canonical_replicas` the component lands on only `gcd(period, canonical_replicas)` of the
lanes, so the rest never see it at all. A downstream gate waiting on that component, such
as a strict `ensure_mixture`, then stalls on the starved lanes.

`lane_assignment="permute"`, the default, routes each block of `canonical_replicas`
consecutive chunks through a seeded permutation of the lanes:

```
canonical_replicas = 4, modulo:          canonical_replicas = 4, permute:

  g:    0  1  2  3  4  5  6  7            g:    0  1  2  3  4  5  6  7
  lane: 0  1  2  3  0  1  2  3            lane: 2  0  3  1  1  3  0  2
        └── block 0 ──┘└─ block 1 ─┘            └── block 0 ──┘└─ block 1 ─┘
```

Every block still hands each lane exactly one chunk, so the load balance and the
exactly-once property are untouched, and the permutation is a pure function of
`(seed, block)`, so replay is unaffected. A periodic emitter now sprays the lanes roughly
uniformly instead of hitting the same few. We compose integers for the block seed rather
than hashing a tuple, because `hash()` is not stable across processes unless you pin
`PYTHONHASHSEED`, and we would rather your data order did not depend on an environment
variable.

**Why the schedule is not pre-computed.** Most distributed loaders materialize the full
sample order up front: build a permutation of every index, shard it across ranks, iterate.
That works when the dataset size and the rank count are both known before training starts,
and it makes "skip to 73% of the epoch" a one-line operation.

A `WorkSource` is a streaming generator instead, producing chunks on demand and never
holding the whole sequence, and three things follow from that. The first is that a source
does not have to be finite: the built-in one can repeat its catalogue for a fixed number
of passes or indefinitely, and you could write a source that grows while training runs
without changing anything else. The second is that changing the rank count does not
require recomputing a global permutation, because the lane abstraction absorbs it and the
per-lane sequences are unchanged, so only the question of which group owns which lane has
a new answer. The third is that memory does not scale with the sample count, for the
reasons the cursor model above gives.

What it costs you is the one operation a materialized permutation makes trivial. There is
no global index to seek to, so Zephon tracks progress through opaque cursors inside the
source and the only way to arrive at a position is to have been handed its checkpoint. We
think that is a fair trade, because the positions you actually want to resume from are the
ones you checkpointed, and those restore in time independent of how far into training they
are.
