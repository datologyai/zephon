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
The choice of curriculum affects what the model learns.
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

Chunks also play a key role in reducing checkpointing time.  Because the Engine tracks
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
    chunk_size=16384,
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

For example, with `chunk_size=16384` and weights `{"fineweb": 0.7, "dclm":
0.3}`, the quotas would be approximately `fineweb=11469, dclm=4915`.  The
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
from each cursor and assembles them into a work chunk.  What happens when a
cursor runs out of samples depends on the **exhaustion policy**.

### Exhaustion policies

By default the source runs `stop_after_passes=1`: every dataset repeats and
the stream ends once all of them have completed one full pass.  The largest
dataset is seen exactly once and triggers the stop; smaller datasets loop in
the meantime so the mixture ratio holds throughout.  Pass a larger integer to
require more passes over every dataset.  Passing `exhausted_policy` or
`max_repeats` turns the floor off and opts into the per-dataset policy below;
the two are mutually exclusive.

The `exhausted_policy` parameter controls what
{py:class}`~zephon.work.StaticMixtureWorkSource` does when a mixture
component can no longer fill its per-chunk quota:

| Policy | Behaviour |
|---|---|
| `"stop"` | Return `None` as soon as **any** component is exhausted. The training loop sees end-of-data. |
| `"repeat"` | Reset the exhausted component's cursor to position 0 and keep going. Each component restarts independently — a small dataset may cycle several times while a large one is still on its first pass. The source never returns `None` (unless `max_repeats` is set). |

```python
ws = StaticMixtureWorkSource(
    datasets=[large_corpus, small_corpus],
    mixture={"large": 0.6, "small": 0.4},
    chunk_size=16384,
    seed=42,
    exhausted_policy="repeat",      # restart components that run out
    reshuffle_on_repeat=True,       # different sample order each epoch
    max_repeats=3,                  # stop after 3 restarts (optional)
)
```

**`reshuffle_on_repeat`** (default `True`): when a component restarts, rebuild
its sample-order array with a deterministic epoch-derived seed so each pass
through the data sees a different traversal order.  Set to `False` to replay
the exact same order every epoch.

**`max_repeats`** (default `None` = unlimited): cap the number of times any
single component may restart.  Once a component hits the cap, the source
returns `None` just like `"stop"`.  This is useful as a safety valve when
you want repetition but do not want to rely solely on `max_steps` to stop
training.

**Per-dataset policies.** `exhausted_policy`, `reshuffle_on_repeat`, and
`max_repeats` each accept either a scalar (broadcast to every dataset) or a
`{dataset_name: value}` mapping, so datasets in the same mix can differ — for
example one repeating forever as padding while another stops when exhausted.

```python
ws = StaticMixtureWorkSource(
    datasets=[web_corpus, instruction_data],
    mixture={"web": 0.7, "instructions": 0.3},
    chunk_size=16384,
    seed=42,
    # web pads forever; the stream ends once instructions is exhausted
    exhausted_policy={"web": "repeat", "instructions": "stop"},
    reshuffle_on_repeat={"web": True, "instructions": False},
)
```

```{note}
A `"redistribute"` policy (shift an exhausted component's quota
proportionally to the remaining components) is planned but not yet
implemented.
```

**Tail-drop semantics.** When a component exhausts with fewer remaining
samples than its quota, those leftover samples are dropped — the same
`drop_last` semantics already implicit in `"stop"` mode.  For typical
dataset sizes this is negligible (at most `quota - 1` samples per epoch
boundary).

### Token-aware mixtures

By default the mixture is enforced in **samples**: 50/50 means half the
sample pointers come from each dataset.  Whether that is also 50/50 in
*tokens* depends on your corpus.  If every dataset is preprocessed into
equal-length sequences, sample and token proportions are the same thing and
the default is exactly right.  When samples are whole documents instead
(raw text, or pretokenized without splitting into training sequences),
length varies wildly across domains: one arXiv paper can span several of
the model's sequences, while a short email fills only a fraction of one.
Mixed 50/50 by sample, the model sees far more paper tokens than email
tokens, so the de-facto mixture at the model diverges from the one you
declared.  Truncating the long documents would restore the ratio, but only
by throwing away most of their content.

```{warning}
Token-aware mixtures do not solve the problem of turning variable-length
documents into equal-length trainable samples.  In the canonical LLM
pipeline that is the job of packing (`tokenize` → `pack` → `shuffle`), and
with imbalanced corpora the packing step needs thought of its own.
Dedicated packing documentation is in the works.
```

Passing a `token_estimation` makes the declared weights **token**
proportions instead.  The WorkSource estimates how many tokens each sample
pointer is worth and hands out pointers such that the *tokens* arrive at the
declared ratios:

```python
from zephon.work.token_estimation import TokenEstimation

ws = StaticMixtureWorkSource(
    datasets=[fineweb, arxiv],
    mixture={"fineweb": 0.5, "arxiv": 0.5},   # token proportions
    token_estimation=TokenEstimation(),
)
```

How it works:

- **Priming.**  Once per run, in the driver, the pipeline calibrates a
  tokens/byte ratio per dataset by fetching a few samples and tokenizing
  them with the pipeline's tokenizer configuration (truncation and special
  tokens are accounted for analytically).  Ratios persist in checkpoints, so
  restored runs never re-measure.  When the heuristics do not fit your data
  (binary payloads, VLM cost units), pin the ratios or provide a `measure=`
  callable via {py:class}`~zephon.work.token_estimation.TokenEstimation`.
- **Per-draw allocation.**  Instead of computing fixed per-chunk quotas
  (see [above](#chunk-quota-allocation)), the WorkSource fills each chunk
  one sample at a time, always drawing from the component that is furthest
  behind on tokens.  Each draw is charged the sample's estimated cost: the
  shard's average bytes/sample from the shard catalog, times the primed
  tokens/byte ratio.

The estimation is intentionally coarse.  Shard-average resolution is enough
to remove the systematic length difference *between* datasets; length
variation *within* a dataset remains as noise around the target.  Operators
between fetch and tokenize that change token mass (aggressive filters,
dedup) are invisible to priming and show up as mixture drift; if you know
their expected pass rates, fold them into pinned ratios.

Token-aware mixtures do not require
{py:meth}`Pipeline.ensure_mixture() <zephon.api.Pipeline.ensure_mixture>`,
but the two compose.  Each chunk carries the declared token mixture as its
`target_mixture`, so a downstream `ensure_mixture` aims at your token target
rather than the chunk's (deliberately skewed) sample composition.  Because
the source already delivers the target on average, a small bounded buffer
suffices to smooth the per-document noise, and nothing needs to be dropped.
If you add one, place it *after* `tokenize` and keep a token-unit
`weight_by`; the pipeline checks this before any data flows.

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
