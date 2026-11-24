# Changing the mixture over training

`StaticMixtureWorkSource` can change its dataset proportions at fixed points in training,
for example shifting from web data toward code. The datasets stay the same, and the whole
schedule is fixed before training starts.

## Phases in optimizer steps

Write the curriculum as a `StepMixturePlan` of `MixturePhase`s. Each phase is a complete
mixture that holds from its `step` until the next phase. Pass the plan as the `mixture`,
and set `Pipeline.options(global_batch_size=...)` to the trainer's global batch size.

```{literalinclude} ../../../examples/guide/worksources/step_curriculum.py
:language: python
:caption: examples/guide/worksources/step_curriculum.py
```

Phases start at step 0, increase strictly, stay below `total_steps`, name every dataset,
and sum to 1. A dataset may start at 0 but must be active in some phase. `total_steps`
only bounds the plan: training can run longer, and the last mixture keeps holding.

## Ramps

Instead of `phases`, pass `schedules`: one per dataset, each starting at step 0.
`PhaseLinearSchedule` interpolates between `(step, weight)` breakpoints;
`PhaseConstantSchedule` holds each weight until the next breakpoint. Weights are relative
and normalized at every step. A linear schedule's last breakpoint may equal `total_steps`
to mark the end of a ramp. Every other breakpoint must be below it.

```{literalinclude} ../../../examples/guide/worksources/linear_step_curriculum.py
:language: python
:caption: examples/guide/worksources/linear_step_curriculum.py
```

## How optimizer steps map to the data stream

`global_batch_size` is the number of records one optimizer step consumes across all
data-parallel ranks:

```text
global_batch_size = micro_batch_size × dp_degree × gradient_accumulation_steps
```

Pass that, not the per-rank batch, the microbatch, or a token count. Zephon assumes one
record is one training example, as in a pretokenized, fixed-window pipeline.

Zephon hands out data in chunks of `chunk_size` records to `canonical_replicas` lanes. It
changes the mixture only between chunk rounds (one chunk per lane), so every lane uses the
same mixture for a round, however many workers, ranks, or resumes are involved:

```text
records_per_round = chunk_size × canonical_replicas
steps_per_round   = records_per_round / global_batch_size
```

Each round trains on the average of the per-step mixtures of the steps it covers. A
breakpoint is exact when it starts a round, that is, when `step × global_batch_size` is a
multiple of `records_per_round`. By default (`boundary="exact"`), a plan with any other
breakpoint refuses to start, and the error names the nearest exact steps. This includes a
ramp's breakpoints and a ramp end at `total_steps`. With
`StepMixturePlan(boundary="warn")`, Zephon logs a warning and starts anyway. The round
containing such a breakpoint then trains on a blend of both mixtures, weighted by how many
of its steps fall on each side. In either mode, linear ramps become staircases
`steps_per_round` steps wide.

With `boundary="warn"`, a switch from 100% `a` to 100% `b` lands like this:

| Global batch (micro × DP × GA) | chunk_size | canonical_replicas | Steps per round | Phase at step 1000 | Phase at step 8000 |
|---|---|---|---|---|---|
| 512 (8 × 8 × 8) | 16384 | 8 | 256 | steps 768–1023 train on 91% a / 9% b | steps 7936–8191: 25% a / 75% b |
| 512 (8 × 8 × 8) | 16384 | 64 | 2048 | steps 0–2047 train on 49% a / 51% b: the first phase is blended away entirely | steps 6144–8191: 91% a / 9% b |
| 512 (8 × 8 × 8) | 1024 | 8 | 16 | steps 992–1007: 50% / 50% | exact |
| 256 (8 × 8 × 4) | 16384 | 8 | 512 | steps 512–1023: 95% a / 5% b | steps 7680–8191: 62.5% / 37.5% |

Blending is deterministic, but it is not "this mixture until step X": in the second row the
first phase never trains on its own. Changing `chunk_size` changes both the record order
and where blends fall.

`describe()` shows where every breakpoint lands, so you can check a plan before launching:

```python
print(plan.describe(global_batch_size=512, chunk_size=16384, canonical_replicas=8))
```

## Choosing settings

- Put breakpoints on exact steps (with 256 steps per round, use 768 or 1024 rather than
  1000), and keep every phase several rounds long.
- For a finer curriculum, lower `chunk_size` or `canonical_replicas`. Lowering
  `canonical_replicas` caps how far you can scale out on resume; see
  [Distributed training](../pipelines/distributed_training.md).
- Set `canonical_replicas` explicitly. If unset, it equals the first run's `dp_degree`, so
  the curriculum's resolution would depend on cluster size.

## What step plans require

Zephon checks these when the pipeline starts:

- **Whole microbatches per lane.** The pipeline must end in `.batch(micro_batch_size)`.
  `global_batch_size / micro_batch_size` must be a multiple of `canonical_replicas`, and
  `canonical_replicas` a multiple of `dp_degree`, so every lane and every rank contributes
  equally to each step. The error suggests values that work. This check applies to any
  pipeline that sets `global_batch_size`, with or without a curriculum.
- **Records stay in order.** `shuffle`, `ensure_mixture`, `pack_sequences`, and
  `pack_flat` are rejected. To shuffle, set `shuffle_within_shard=True` on the work source.
- **One record per training example.** Operators that may split, merge, or drop records
  are rejected: `tokenize(split_long_samples=True)`, `expand_chat`, `stateful_transform`,
  and custom operators that do not declare `one_to_one_records=True`.
- **No `TokenEstimation`.**

Some operators drop records only for certain inputs: a `map_transform` that returns
`None`, a `map_batch` that drops or changes its batch, or a `tokenize_chat` sample it
cannot tokenize faithfully. Under a step plan these raise an error naming the lane, chunk, and
offset, rather than silently shifting every later breakpoint.
`map_transform(..., drop_none=False)` keeps such records and is allowed. To filter, do it
before loading the dataset, or use a chunk-round `MixtureSchedule`.

## Schedules in chunk rounds

To work in Zephon's own units, pass a `MixtureSchedule` as the `mixture`. Breakpoints are
then chunk rounds, and nothing is averaged. Each dataset takes a `PhaseConstantSchedule`, a
`PhaseLinearSchedule`, a `ConstantSchedule`, or a plain number (a constant weight).
Filtering and packing work here; `TokenEstimation` does not. A round's size depends on
`chunk_size` and `canonical_replicas`, so prefer `StepMixturePlan` unless you control both.

```{literalinclude} ../../../examples/guide/worksources/chunk_round_schedule.py
:language: python
:caption: examples/guide/worksources/chunk_round_schedule.py
```

The schedule is fixed for the life of the work source. Lanes evaluate it when they produce
a chunk, rounds ahead of the trainer and in their own processes, so a schedule that reacts
to training could not produce a deterministic stream.

`ensure_mixture()` without `mixture=` follows the scheduled mixture. Passing `mixture=`
would replace the curriculum, so Zephon rejects it.

## Resuming, repetition, and stopping

- **Resuming.** Checkpoints store a fingerprint of the schedule, not the schedule itself.
  Before calling `restore`, rebuild the work source with the same plan, `chunk_size`,
  `canonical_replicas`, and `global_batch_size`; a missing or different schedule is
  rejected. Use `to_dict()`/`from_dict()` to save the plan alongside your checkpoints.
- **Repeating datasets.** With `exhausted_policy="repeat"`, each dataset must be large
  enough for the largest per-chunk share the schedule ever gives it.
- **Stopping.** A dataset whose final weight is 0 never finishes a pass or runs out. Under
  `stop_after_passes` (the default is 1), every dataset must finish, so a dataset that ends
  at 0 is rejected. With `stop` policies or `max_repeats`, the source is rejected only if
  every dataset that could stop it ends at 0. To end at weight 0, use
  `exhausted_policy="repeat"` and stop training by step count. `len()` is undefined while
  a schedule is attached.
