# User-Defined Operators

The previous pages of this guide covered the operators that Zephon ships with: fetching,
decoding, tokenizing, shuffling, etc. Most training pipelines can be expressed wholly with
those built-ins plus the `map_transform` operator (see
[Transforming and Filtering Records](preparing_samples.md)). We will now
examine the case where these options are not enough; e.g. your data has a peculiar schema,
or where a filtering decision depends on samples you have already seen, or where a
transformation is expensive or different enough that you need fine-tuned control over how
it is batched and parallelised.

Whenever you call any operator such as `tokenize` or `pack_flat`, the machinery is exactly
the same whether it is a Zephon-shipped operator or a custom one. This means that a custom
operator can be as efficient as a built-in one, though it can also break determinism and
checkpointing if not written carefully.

**Your Own Functions.** There are multiple ways in Zephon to run custom pipeline code.
Suppose the documents in your dataset include some boilerplate (e.g. website navigation
bars) that you would like to protect your model from. Furthermore, you would like to
remove anything that is too short after stripping this boilerplate. Similar to what we
have seen in [Transforming and Filtering Records](preparing_samples.md), this can be
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

**Remembering Things Between Samples.** There are plenty of cases where you would want
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

**The Limits of Transformations.** The examples above work well as simple
transformations, but sometimes you may require something more complicated, such as
running documents through a classifier for scoring. You could write this as a filter
transformation, but it would be slow as it would run the classifier on individual
records instead of batches of records. Any operations that require different mapping
like this, e.g. once per worker instead of once per record, require writing an operator.

**Writing Your Own Operators.** An operator you write yourself is the same kind of
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

**Attaching A Function.** As discussed above, there are two ways to add operators. The
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

**Writing a full Operator class.** The function form covers operators whose dependencies
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
There is also the optional `accumulator` for special grouping, and `__init__` for
configuration and `setup` for per-worker initialization.

These last two can be the cause of some confusion, as they may feel quite similar.
`__init__` runs once in your process when the operator is built, and it should be solely
used for configuration. `setup` runs once per worker, which is e.g. where we can load our
model.

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
- `indexable` says whether the pipeline can be accessed by position (à la PyTorch
  map-style dataset).
- `parallelism` notes the preference for how many workers should be given to the operator.
- `requires_serial_state` causes the operator to run with parallelism 1 in deterministic
  mode.
- `batch_shape_sensitive` indicates your output depends on how records were grouped, which
  disables latency-based grouping.

**Operator Validation.** The first time you let Zephon iterate on your `Pipeline`, it
will check whether your operators are valid. The validation suite checks the results and
the source of a test suite of a few small synthetic records. Besides checking for
malformed and unexpected output, this checks determinism, namely whether calling your
operator twice with the same input gives the same output. It also checks whether there
are any wrongful dependencies by checking whether the output depends on earlier input.
Finally, the validator runs checks on writing to attributes and unseeded randomness.
These last two are reported as warnings; the rest will error and stop the pipeline.

The validation framework is exclusive to custom operators (those you add with `add_op`).
Anything that you run with `map_transform` and `stateful_transform` runs inside Zephon's
own operators and is not validated this way.
