# User-defined operators

The previous pages of this guide covered the operators that Zephon ships with: fetching,
decoding, tokenizing, shuffling, etc. Most training pipelines can be expressed wholly with
those built-ins plus the `map_transform` operator (see
[Transforming and filtering records](preparing_samples.md)). We will now
examine the case where these options are not enough; e.g. your data has a peculiar schema,
or where a filtering decision depends on samples you have already seen, or where a
transformation is expensive or different enough that you need fine-tuned control over how
it is batched and parallelized.

Whenever you call any operator such as `tokenize` or `pack_flat`, the machinery is exactly
the same whether it is a Zephon-shipped operator or a custom one. This means that a custom
operator can be as efficient as a built-in one, though it can also break determinism and
checkpointing if not written carefully.

**Your own functions.** There are multiple ways in Zephon to run custom pipeline code.
Suppose the documents in your dataset include some boilerplate (e.g. website navigation
bars) that you would like to protect your model from. Furthermore, you would like to
remove anything that is too short after stripping this boilerplate. Similar to what we
have seen in [Transforming and filtering records](preparing_samples.md), this can be
expressed via a `map_transform`:

```{literalinclude} ../../../examples/guide/pipelines/map_transform_filter.py
:language: python
:caption: examples/guide/pipelines/map_transform_filter.py
```

The above Python snippet introduces a function `clean_document` that receives a sample's
payload and returns the payload with the transformations applied. Furthermore, it drops
payloads we do not want, in this case documents that are too short, by returning `None`.
Everything else is handled by Zephon: running your function on workers in Zephon's
infrastructure, putting the results back in the right order, and ensuring stability for
checkpointing and resumption (as long as your function is deterministic!).

A close relative is `map_batch`, which hands your function a whole batch instead of a single
payload. This means that it has to come after the `batch` operator in the pipeline.

Handing functions to `map_transform` and `map_batch` is convenient, but there are limits to
what they can achieve, which are covered by Custom Operators.

**Remembering things between samples.** There are plenty of cases where you would want
to see more than one sample for your transformation. Holding back records until you can
pair them with another one, or merging records that belong together. In these cases, we
can sub out our `map_transform` for `stateful_transform`:

```{literalinclude} ../../../examples/guide/pipelines/stateful_transform_pair.py
:language: python
:caption: examples/guide/pipelines/stateful_transform_pair.py
```

`stateful_transform` requires a few extra pieces of information to manage its state. You
provide `init_state` to create the state and `push` advances it. The latter receives the
current state and some records, and returns the new state with whatever should be emitted.
We can also hold on to documents this way. Finally, `flush` is used to emit any leftovers.
The hard work is done under `transform`, which runs on parallel workers:

```{literalinclude} ../../../examples/guide/pipelines/stateful_transform_workers.py
:language: python
:caption: examples/guide/pipelines/stateful_transform_workers.py
```

A few things should be noted here:

- If your `push` buffers records, you must supply a `flush` to empty this buffer.
  Otherwise, these records are discarded silently.
- You can change the order of records, but you must set `preserves_cursor_order` to
  `False` if you do so. This includes shuffling, pairing, and anything else that holds a
  record back past a later one.
- State is per lane. A lane is an independent data stream, which means that your
  transformation can only hold state within one. `init_state` runs independently for every
  lane. Furthermore, state is reset periodically on flush.
- Unlike `map_transform`, if you do use `stateful_transform` for dropping records, you
  must emit tombstones so that Zephon does not keep waiting for them.

**The limits of transformations.** The examples above work well as simple
transformations, but sometimes you may require something more complicated, such as
running documents through a classifier for scoring. You could write this as a filter
transformation, but it would be slow as it would run the classifier on individual
records instead of batches of records. Any operations that require different mapping
like this, e.g. once per worker instead of once per record, require writing an operator.

**Writing your own operators.** An operator you write yourself is the same kind of
object as the ones Zephon ships. The operators behind `tokenize`, `shuffle`, and
`pack_flat` are all `BaseOp` subclasses, with the same lifecycle, grouping machinery,
and traits your own operators will have. We can define our own custom operators via
`add_op` with two paths: a light one taking a function, and a class-based one for
operators that need the full lifecycle.

Before the `batch` operator, a pipeline carries `SampleRecord` objects, which contain your
data. After `batch`, the pipeline carries `SampleBatch` objects instead, which is why
`map_batch` exists for that position.

Operators receive and return records; they transform the payload and carry some Zephon
bookkeeping through. Rebuild `record.payload` as needed and return the record that carried
it. Returning a bare dictionary or a raw tensor instead is the most common first-operator
mistake, and Zephon will reject it:

```{literalinclude} ../../../examples/guide/pipelines/records_in_records_out.py
:language: python
:caption: examples/guide/pipelines/records_in_records_out.py
```

**Attaching a function.** As discussed above, there are two ways to add operators. The
light form of `add_op` takes a name and a `process_many` function. Here is the previous
classifier, rating sixty-four documents per call across four workers. It writes a score
onto each record and leaves the dropping to a `map_transform` downstream:

```{literalinclude} ../../../examples/guide/pipelines/add_op_quality_score.py
:language: python
:caption: examples/guide/pipelines/add_op_quality_score.py
```

`process_many` receives a list of records and returns a list of records. The grouping is
decided by the operator's accumulator. Passing a `CountingAccumulator` asks for groups of
at most `max_batch` records, buffered per lane so a group never mixes lanes. Without one,
your function receives whatever group arrives from upstream. Note that `accumulator` takes
a factory rather than an instance, as it is rebuilt by Zephon when the pipeline starts and
resets.

Additionally, you can pass a `process_one` function if a single-record path is meaningful
to your function, and a `validation_samples` function to let Zephon check your operator
properly, which we come back to below.

One more detail if you do attach a function after batching: the stream there carries
Zephon's bookkeeping records alongside the batches, so pass anything that is not a
`SampleBatch` through unchanged.

When designing functions and operators, setting `parallelism` higher runs your function on
more than one worker at once. Due to Zephon's determinism, however, the pipeline still
produces the same records in the same order. That holds because `process_many` must not
rely on anything from a previous invocation.

**Writing a full operator class.** The function form covers operators whose dependencies
fit in a closure. The classifier example discussed previously is great right up until
the classifier is a real model that takes seconds to load: the function form does not
allow you to load it once, so you would be needing to load it per call. This can be
circumvented by subclassing `BaseOp` instead.

```{literalinclude} ../../../examples/guide/pipelines/baseop_subclass.py
:language: python
:caption: examples/guide/pipelines/baseop_subclass.py
```

Two methods are required for a `BaseOp` subclass. As seen before, `process_many` is the
workhorse, but this time we need a `traits` that informs Zephon how the operator behaves.
There is also the optional `accumulator` for special grouping, `__init__` for
configuration, `setup` for per-worker initialization, and `teardown` to undo it.

`__init__` and `setup` can be the cause of some confusion, as they may feel quite similar.
`__init__` runs once in your process when the operator is built, and it should be solely
used for configuration. `setup` runs once per worker, which is e.g. where we can load our
model.

`teardown` is the counterpart to `setup`: it runs once per worker when the pipeline
shuts down, and it is where you release what `setup` acquired, such as open files,
network connections, or a model holding GPU memory. The default does nothing. Zephon
also calls `teardown` when `setup` fails partway through, and may call it more than
once, so it must cope with resources that were never created and do nothing the second
time. Resetting the attribute to `None`, as in the example above, is safe in both
cases; for a resource with a `close()` method, check for `None` before closing it.

Treat `teardown` as best-effort cleanup, not a guarantee. A worker that crashes, or is
terminated during a hard shutdown, may never run it, so do not rely on it to save
results or anything else that must happen.

Once we have defined our subclass, we can pass it on by passing an instance:

```{literalinclude} ../../../examples/guide/pipelines/add_op_instance.py
:language: python
:caption: examples/guide/pipelines/add_op_instance.py
```

Configuring `traits` is essential to get the operator to work the way you want in Zephon:

- `preserves_cursor_order` is one we have seen before and is mandatory. Set it to `True`
  if your operator emits records in the same order it received them in, or `False`
  otherwise. Be careful; if you do set this to `True` while not preserving order, it will
  silently break checkpointing.
- `calibration_replayable`: set `True` if processing each fetched record independently
  produces the same payloads and record counts as normal execution, including drops;
  otherwise leave `False` (default). Order may differ. See {class}`~zephon.ops.OpTraits`
  for replay constraints and examples. If you leave this `False` on an operator before
  the first tokenizer, you need to supply token ratios or a custom
  `TokenEstimation(measure=...)`.
  Calibration supplies real `sample_id` values, but placeholder lane/chunk positions,
  mixture-component counts, and stage metadata. Leave `False` if replacing runtime
  values in those fields with placeholders would change the emitted payloads, number
  of records, or which records are dropped; copying metadata through with helpers
  such as `spawn_child` is fine.
- `indexable` says whether the pipeline can be accessed by position (à la PyTorch
  map-style dataset).
- `parallelism` notes the preference for how many workers should be given to the operator.
- `one_to_one_records` says every record leaves your operator as exactly one record with
  the same identity. `StepMixturePlan` refuses to start unless every operator sets it to
  `True`; leave it unset if unsure.
- `requires_serial_state` causes the operator to run with parallelism 1 in deterministic
  mode.
- `batch_shape_sensitive` indicates your output depends on how records were grouped, which
  disables latency-based grouping.

**Splitting records.** When an operator turns one record into several, construct each
child's metadata with `spawn_child` from `zephon.ops.children`. Give the children
deterministic indices in emission order and set `is_last_child=True` only on the final
child of that immediate parent. The helper appends the index to the parent's lineage
so that every child has its own replay identity.

A contributor's closing flag means that this record carries responsibility for
completing the original source sample; it does not mean completion has already been
reported. If you split a record again, only its final child inherits that responsibility.
Splitting a non-closing parent produces no closing children, even though you still mark
its locally final child with `is_last_child=True` when calling the helper.

For example, splitting a source into children `(0,)` and `(1,)`, then splitting each
again, gives:

| Child lineage | Carries the source's closing flag |
| --- | --- |
| `(0, 0)` | No |
| `(0, 1)` | No |
| `(1, 0)` | No |
| `(1, 1)` | Yes |

For packed inputs, the same rule applies separately to each contributor: the final child
inherits the parent's existing flags, including any non-closing ones. Earlier children
keep the contributor cursors but clear all closing flags. If you drop a record instead
of splitting it, use `tombstones_for_record` from the same module to emit tombstones for
the closing responsibilities it carries.

(reacting-to-source-exhaustion)=

**When a mixture component runs out.** Suppose our operator holds back records from B
while waiting for more records from A. These could be languages within one dataset.
If A runs out, we may want to let B continue. Zephon reports this through an end-of-data
(EOD) sentinel. Your accumulator can react by overriding
`on_source_exhausted(lane_id, component_id)`; by default, it does nothing.

This tells us something different from a change to the target mixture. The target says
which proportions we want to produce. A curriculum might remove A for a while and bring
it back later. An EOD sentinel says that A will supply no more source samples in this
lane. That fact is permanent for the run, regardless of what the target mixture says.

For example, `ensure_mixture` can stop waiting for A and continue correcting the mixture
of the remaining components. An EOD sentinel lets it do this before records carrying an
updated target arrive. It also works when the operator has an explicit `mixture`
override instead of following the source's target. It only relaxes the target for
components listed in `WorkSource.continues_after_exhaustion()`. The built-in
`StaticMixtureWorkSource` lists components with `exhausted_policy="redistribute"`.
EOD sentinels do not relax mixture enforcement under its stop or repeat policies.

There is one catch: A can still have records buffered upstream, for example in a shuffle
or packer. Those records may reach your operator after the sentinel, so you still need to
decide how to handle them. `ensure_mixture` interleaves them with records from the remaining
components. Its bounded mode preserves leftovers at flush, whereas strict mode drops them.
Your operator may need a different policy, but it should not assume that receiving an
EOD sentinel means it has already seen every record from that component.

Zephon remembers which components have run out, including when your accumulator's own
state is reset. The [accumulator API](../api/operators.rst) explains how to read that
information and implement the hook. Keep the usual `flush` handling too: an EOD sentinel
describes one component running out, rather than the whole pipeline finishing.

**Operator validation.** The first time you let Zephon iterate on your `Pipeline`, it
will check whether your operators are valid. The validation suite checks the results and
the source of a test suite of a few small synthetic records. Besides checking for
malformed and unexpected output, this checks determinism, namely whether calling your
operator twice with the same input gives the same output. It also checks whether there
are any wrongful dependencies by checking whether the output depends on earlier input.
Finally, the validator runs checks on writing to attributes and unseeded randomness.
These last two are reported as warnings; the rest will error and stop the pipeline.

If you set `calibration_replayable=True`, Zephon also checks that processing records one
at a time or replacing their metadata with placeholders does not change the output.
For operators added through keyword arguments, it also checks whether the accumulator
changes the result. These checks use your `validation_samples()` or generated records,
without calling `setup()`. Zephon warns when it cannot complete a check. You should still
test your operator on representative data, since these checks only cover a few records.

The validation framework is exclusive to custom operators (those you add with `add_op`).
Anything that you run with `map_transform` and `stateful_transform` runs inside Zephon's
own operators and is not validated this way.

If your operator splits records, merges them, or drops them, it owes the engine some
bookkeeping so that progress tracking and checkpointing keep working.
[Records and sample lifecycles](../internals/records_and_sample_lifecycles.md) covers the
helpers that handle it and the contract behind them.
