# WorkSources

The curriculum (which datasets, in what proportions, in what order) is an important
training decision that lives in the WorkSource.  The [Basic Concepts](../basic_concepts.md) page
introduced the WorkSource at a high level; this page goes deeper into the
abstraction, the built-in implementation, and where we see it heading.

---

## The Curriculum

In a training run, the **curriculum** is the global ordering of training
samples.  Some data loaders treat the curriculum as a byproduct of how data happens to be
stored and partitioned.  In Zephon, the curriculum is an explicit concept
owned by the WorkSource.
These choice of curriculum (potentially) affects what the model learns.  
By placing it in a dedicated component, it
is easy to inspect, configure, checkpoint, and eventually swap out
without touching any of the downstream processing logic.

Note that shuffling is part of the curriculum: the decision to visit shards in random
order or to permute samples within shards changes what the model sees and
when.  The built-in {py:class}`~zephon.work.StaticMixtureWorkSource` exposes
this through three orthogonal knobs (see
[Basic Concepts](../basic_concepts.md#shuffling) for details).

The WorkSource expresses the curriculum as a stream of
**work chunks**: fixed-size groups of sample **pointers**, each a
`(dataset_id, shard_id, sample_idx)` triple.  No I/O happens here; the
WorkSource only decides *which* samples to train on.  Actual data loading
starts later, when the pipeline's built-in
{py:class}`~zephon.ops.FetchOp` resolves those pointers into data.

```
WorkSource                              Pipeline
┌──────────────────────┐    pointers    ┌───────────────────────┐
│                      │ ─────────────> │ FetchOp → Tokenize →  │
│  "what to train on"  │  (no I/O)      │ ... → Batch           │
│                      │                │                       │
│  curriculum logic    │                │  "how to process it"  │
└──────────────────────┘                └───────────────────────┘
```

This separation of *what to train on* from *how to process it* is one of
Zephon's central design principles, as discussed in
[Basic Concepts](../basic_concepts.md#declaring-your-data).

```{note}
The *effective* curriculum (what the model actually trains on) can
diverge from what the WorkSource produces.  Downstream operators alter it:
tokenization with splitting turns one document into a variable number of
sequences, packing merges short samples, and filtering drops some entirely.
After these steps, the token-level mixture ratio may no longer match the
WorkSource's target.  Zephon provides pipeline-level tools like
{py:meth}`Pipeline.ensure_mixture() <zephon.api.Pipeline.ensure_mixture>` and
{py:meth}`Pipeline.shuffle() <zephon.api.Pipeline.shuffle>` to correct for
this drift.  See
[Keeping Mixtures on Track](../basic_concepts.md#keeping-mixtures-on-track)
for details.
```

---

## The WorkSource Interface

Every WorkSource must satisfy a small contract defined by the
{py:class}`~zephon.work.WorkSource` base class:

```python
class WorkSource(ABC):
    def next_chunk(self) -> WorkChunk | None: ...

    @property
    def datasets_by_id(self) -> Mapping[int, Dataset]: ...

    def clone_for_lane(self, lane_id, canonical_replicas) -> WorkSource: ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state: dict) -> None: ...
```

The key methods fall into three groups:

### Chunk production

{py:meth}`~zephon.work.WorkSource.next_chunk` is the core method.  The Engine
calls it repeatedly on each per-lane clone to pull the next
{py:class}`~zephon.work.WorkChunk` (described in the
[next section](#work-chunks)).  When the WorkSource is exhausted, it returns
`None`.  Everything else in the interface exists to support this one method.

### Lane binding

Zephon distributes work across **lanes**, logical, deterministic
sub-streams of the global sample order (see
[Elastic Determinism](determinism.md) for background).  Before the Engine
starts pulling chunks, it **clones** the WorkSource once per lane via
{py:meth}`~zephon.work.WorkSource.clone_for_lane`.  Each clone is bound to a
specific `(lane_id, canonical_replicas)` pair and produces only the chunks
assigned to that lane.  The default implementation deep-copies the instance,
but subclasses can override this for efficiency.

### Checkpointing

{py:meth}`~zephon.work.WorkSource.state_dict` and
{py:meth}`~zephon.work.WorkSource.load_state_dict` serialize and restore the
WorkSource's internal state so that `next_chunk()` can resume producing
new chunks from where it left off.  What goes into the dict is entirely up
to the implementation.  The only hard requirement is that restoring from a
saved state and then calling `next_chunk()` must produce the same sequence
as if the WorkSource had never been interrupted.

For the built-in {py:class}`~zephon.work.StaticMixtureWorkSource`, the
sample sequence is deterministic (fixed by datasets, seed, shuffle knobs,
and lane ID), so the essential mutable state is just a per-dataset position
integer and a global chunk counter.  The checkpoint also carries
configuration fields (seed, knobs, weights) for validation on restore.
Restore time is independent of how far into training you are: the
cursors are rebuilt deterministically from the datasets and knobs, then
advanced to the saved position.

A hypothetical future WorkSource that makes online decisions (e.g., adjusting
mixture weights based on training signals) would need to serialize whatever
internal state drives those decisions.  The contract does not prescribe
what that state looks like.

For the full picture of how WorkSource state fits into Zephon's checkpoint
protocol, see [Checkpointing](checkpointing.md).

---

## Work Chunks

A {py:class}`~zephon.work.WorkChunk` is the unit of work that flows from the
WorkSource to the Engine.  It bundles sample pointers grouped by **mixture
component**:

```python
WorkChunk(
    components={
        "fineweb": [(0, 3, 17), (0, 3, 18), (0, 1, 42), ...],  # 70 pointers
        "dclm":    [(1, 0, 5),  (1, 2, 99), ...],               # 30 pointers
    },
    seed=42,
)
```

When the Engine iterates over a chunk, the chunk's default iteration order
uses {py:class}`Smooth Weighted Round Robin
(SWRR) <zephon.utils.swrr.SmoothWeightedRoundRobin>`.
SWRR tracks a per-component deficit and always emits from the most
underrepresented component next, ensuring that the per-component ratio is
maintained not just across the chunk as a whole but approximately within any
prefix of it.

The component grouping also enables downstream operators like `ensure_mixture`
to know which mixture component a sample belongs to, even after
transformations like tokenization or packing have altered sample boundaries.

Chunks also play a key role in reudcing checkpointing time.  Because the Engine tracks
progress at chunk granularity, only the small number of chunks currently
in flight need to be saved and replayed on resume, not the entire
history of consumed samples.   See
[Checkpointing](checkpointing.md#chunks-and-inflight-state).

---

## StaticMixtureWorkSource

{py:class}`~zephon.work.StaticMixtureWorkSource` is the only WorkSource
implementation that ships with Zephon today.  It covers the common case:
a fixed set of datasets mixed in fixed proportions.

```python
ws = StaticMixtureWorkSource(
    datasets=[fineweb, dclm],
    mixture=MixtureSpec({"fineweb": 0.7, "dclm": 0.3}),
    chunk_size=1024,
    seed=42,
)
```

The sections below describe how this implementation works internally.
For the user-facing API (constructor parameters, shuffling knobs), see
[Basic Concepts](../basic_concepts.md#declaring-your-data).

### Chunk quota allocation

Given a `chunk_size` and a {py:class}`~zephon.work.MixtureSpec`, the
WorkSource computes a **quota** per component: how many sample pointers each
component contributes to every chunk.  The allocation guarantees that every
component gets at least one slot, and the sum of quotas equals `chunk_size`.

For example, with `chunk_size=1024` and weights `{"fineweb": 0.7, "dclm":
0.3}`, the quotas would be approximately `fineweb=717, dclm=307`.  The
exact split uses a largest-remainder method to distribute rounding leftovers
fairly.

### Per-dataset cursors

Internally, the WorkSource maintains a **cursor** per dataset: a position
into a pre-computed, deterministic sequence of all sample pointers for that
dataset.  The sequence is built once at construction time from the dataset's
shard index and the shuffle knobs (shard order, within-shard permutation,
block shuffle).  After construction, producing the next batch of pointers
is a simple array slice with no RNG calls or shard metadata lookups.

When a chunk is requested, the WorkSource pulls `quota[component]` pointers
from each cursor and assembles them into a work chunk.  If any cursor has
fewer remaining samples than its quota, the WorkSource is 
exhausted and returns `None`.

### Lane partitioning

As described in
[Distributed Training](distributed_training.md#compute-everywhere-discard-locally),
the {py:class}`~zephon.work.StaticMixtureWorkSource` uses a
*compute-everywhere-then-discard* strategy for lane assignment.  Every lane
enumerates the same global chunk sequence deterministically, then keeps
only chunks whose global index maps to its lane:

```
chunk_lane = global_chunk_index % canonical_replicas
```

This means no rank needs to communicate with any other rank to figure out
what data to read.  The schedule is a pure function of the seed and the
lane assignment.

---

## Future Directions

The {py:class}`~zephon.work.StaticMixtureWorkSource` covers the common case
of a fixed dataset mixture, but training curricula can be more sophisticated
than a static set of weights. 
A few directions we are thinking about exploring:

### Dynamic and learned curricula

Static mixtures decide dataset proportions before training begins.  In
practice, the optimal ratio may change over time: early training might
benefit from broad, diverse data, while later stages focus on
domain-specific corpora.  A dynamic WorkSource could accept a schedule of
weight changes (e.g., "after 10B tokens, shift from 70/30 to 50/50") or
even adjust proportions in response to training signals like per-domain
loss.

A WorkSource with more explicit curriculum control (e.g., a scripted
schedule of topics) would also face an inherent tension between
explicit ordering and shuffling, that is interesting to explore.

### Declarative data selection with a DSL

As training data grows in scale and diversity, selecting the right
subset becomes a data-management problem in its own right.  Rather
than listing datasets and weights by hand, a declarative WorkSource
could accept a query in a domain-specific language:

```
# Hypothetical DSL — not implemented today
SELECT * FROM corpora
WHERE language IN ('en', 'de')
  AND quality_score > 0.8
MIX BY language WEIGHTS {'en': 0.7, 'de': 0.3}
```

The idea is inspired by systems like
[Mixtera](https://anakli.inf.ethz.ch/papers/mixtera_sigmod2026.pdf), which
model training data as a queryable corpus and let users express curricula as
composable queries over metadata rather than enumerating datasets.  

### Server-based work distribution

The compute-everywhere-then-discard strategy works well when the
schedule is a pure function of the seed, but it requires every rank to
redundantly enumerate the full global schedule.  A server-based
WorkSource could centrally compute the schedule and push chunks to
ranks, avoiding redundant work and enabling richer coordination, e.g., rebalancing load across ranks, 
integrating with a corpus server,
or supporting curricula that depend on global training state.

```{note}
These directions are forward-looking and not yet implemented.  The
WorkSource interface is designed to accommodate them without breaking
existing pipelines.
```
