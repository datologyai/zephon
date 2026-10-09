# Shuffling and Maintaining Mixtures

The `WorkSource` determines the initial curriculum for a training run and determines how
the underlying `Dataset`s are traversed and their samples are interleaved. But once a
`Pipeline` has filtered, split, tokenized, or otherwise transformed those samples, the
resulting record stream may have a different local order or mixture than the one that the
`WorkSource` originally planned. This page covers the operations that the `Pipeline` API
supports to address these effects before records are prepared for training.

**Shuffling Records.** Given that we already covered shuffling at the WorkSource level
earlier in the guide, you may be surprised (or even delighted?) to encounter another page
on shuffling within our `Pipeline` documentation. While `WorkSource` shuffle settings
control how the samples within a particular `Dataset` are ordered overall, it is possible
for the filtering or transformation operations that were applied during the early stages
of a pipeline to introduce unintended correlations or structure between the stream of
samples that go into a training batch. The `shuffle` operator addresses these correlations
by creating a buffer of samples (with a default size of 8192), populating the buffer with
samples as they stream in, and then in a streaming-shuffle fashion choosing one item from
the buffer to emit downstream. Unlike the per-`Dataset` shuffling performed by the
`StaticMixtureWorkSource`, this operator shuffles the stream of records itself, including
records from different `Dataset`s, but it comes at the cost of the memory and throughput
overheads of needing to buffer a set of records during processing. Setting a larger buffer
size than the default allows for a better shuffle at the cost of additional time and
resources. Like other stateful operators, `shuffle` will emit all of the records in its
buffer at each flush boundary.

```{literalinclude} ../../../examples/guide/pipelines/shuffle.py
:language: python
:caption: examples/guide/pipelines/shuffle.py
```

**Keeping The Mixture on Target.** For most pipelines, the `StaticMixtureWorkSource` is
sufficient to keep your mixture on target, with `TokenEstimation` enabled if you need a
token-aware mix. However, there are a couple of situations where you may want to add an
`ensure_mixture` operator to your pipeline. It works similarly to a shuffle; it keeps a
buffer of records and can reorder them so that its output tracks the overall target
mixture defined in the `StaticMixtureWorkSource`. By default, it weights the samples in
its buffer via their token counts, so be sure that it is placed in the `Pipeline` after
the samples have already been tokenized, or you can specify the `weight_by="samples"`
option to weight all records equally.

One situation is when upstream operations produce uneven output. For example, splitting a
book into chunks can turn one sample into hundreds of records, and a filter might
eliminate most of the samples from one source. The mixture can then be correct on average
but lopsided locally, causing the model to see a long run of tokens from a single
`Dataset`:

```{literalinclude} ../../../examples/guide/pipelines/ensure_mixture.py
:language: python
:caption: examples/guide/pipelines/ensure_mixture.py
```

By default, the `max_buffer_size` parameter for the `ensure_mixture` operator is set to
1000. If the buffer is full, `ensure_mixture` will simply emit whatever is in it rather
than stalling the pipeline, which means that a mixture component that remains scarce in
the stream can deviate from its target. The buffer is also drained at each flush, so the
mixture can drift slightly just before a flush boundary. You can track when this happens
by setting a `warn_tolerance` value for the operator which will cause a message to be
logged if a component drifts from its target (e.g. setting `warn_tolerance=0.05` will log
a message whenever the component's share is off by more than 5 percentage points). If you
would rather drop samples completely in order to enforce a mixture that is as close as
possible to your settings, you can set `max_buffer_size=None` which will cause excess
records from oversupplied components to be dropped at each flush interval (at the cost of
lost data and more memory use):

```{literalinclude} ../../../examples/guide/pipelines/ensure_mixture_unbounded_buffer.py
:language: python
:caption: examples/guide/pipelines/ensure_mixture_unbounded_buffer.py
```

**When a Dataset Runs Out.** When a non-repeating dataset contributes its last source
sample, Zephon inserts a source-exhaustion notification immediately after that sample.
The same happens on the final allowed pass of a dataset with `max_repeats`. An ordinary
repeat boundary does not trigger this notification.

The notification is a source fact; how `ensure_mixture` reacts depends on the source's
policy. `WorkSource.continues_after_exhaustion()` declares the components whose exhaustion
allows the source to keep producing. For those components, both bounded and strict modes
stop waiting when no contributing records are available locally and renormalize the
surviving target shares. For example, if A runs out in a 50:30:20 A/B/C target, the
remaining B/C target is 60:40. Custom sources can opt in now; `StaticMixtureWorkSource`
currently declares no such components. A source that opts in must retain the exhausted
components in `WorkChunk.target_mixture` so their late records remain on target.

For components outside that set, the notification does not change mixture enforcement.
Strict mode (`max_buffer_size=None`) still drops surplus it cannot place on target at
flush boundaries, including end of stream. Bounded mode still drains its remaining
buffer at the final flush. This preserves the behavior of the existing
[stop and repeat policies](../worksources/shuffling_and_repeating_samples.md).
An ordinary repeat boundary or the `stop_after_passes` run boundary does not announce
permanent exhaustion of repeating datasets.

The notification can arrive before records held by an upstream shuffle or packer.
Those records remain on target and are handled normally when they arrive; exhaustion
does not make them obsolete. When the source continues, ratios are approximate during
this transition, even in strict mode. The notification is not proof that processing has
finished, and observing the next flush is not a general substitute: a notification can
pass a stalled batch flush while records from later chunks are still buffered upstream.
