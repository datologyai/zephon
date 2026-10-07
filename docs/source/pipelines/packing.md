# Packing

Packing operators combine multiple short tokenized records into a single sequence for
training. The goal of packing is to ensure that the model spends less time processing
padding tokens, which do not contain data that we want the model to learn but are simply
there so that the structure of our training sequence matches what the model expects. For
example, if a model is trained for an input sequence length of 4096 tokens, but many of
the individual samples are only a few hundred tokens long, then much of the space
allocated for the input sequence would be wasted on padding, which makes our overall
training pipeline much less efficient than it could be. Instead of this, many training
stacks support a mechanism to indicate that tokens from multiple documents are included in
the same input sequence.

In the general case, optimal packing is an NP-hard problem, but there are several useful
heuristics that seem to work well in practice. That said, packing is not always necessary
for a training pipeline; if the input records are generally all the same size and are
close to a fixed target length, or if the model training pipeline cannot correctly handle
document boundaries within a single training sequence, or if keeping a single record per
example is important for the workload, then packing can be excluded from your pipeline
configuration.

**Choosing A Packing Operator.** Zephon ships with two built-in packing operators that
support a common set of packing algorithms and configuration options: `pack_flat` and
`pack_sequences`. Both packing operators take a `max_length` argument that defines the
maximum token capacity of each packed sequence that will be created. This is typically
the sequence length of the model plus one, because we need a final label for the last
token in each sequence. For example, if you use
`SampleBatch.to_training(return_labels=True)`, you will find that if you pack with a
maximum length of 4097, that there will be 4096 input tokens and 4096 label tokens,
shifted one to the right.

The `pack_flat` operator is for the simple and common case in which you want to
concatenate the tokenization fields in the record into fixed-length sequences:

```{literalinclude} ../../../examples/guide/pipelines/pack_flat_output.py
:language: python
:caption: examples/guide/pipelines/pack_flat_output.py
```

The output payload of `pack_flat` is designed to work directly with the
`SampleBatch.to_training()` conversion methods and includes a `positions` array that marks
the boundaries of the tokens from each input sample in the output payload.

The `pack_sequences` operator should be used when you have code downstream of packing that
needs to be able to access the individual records that make up each packed result, which
is common in pipelines for multimodal data:

```{literalinclude} ../../../examples/guide/pipelines/pack_sequences_output.py
:language: python
:caption: examples/guide/pipelines/pack_sequences_output.py
```

Here, the output payload contains a `packed_samples` list rather than an assembled
training sequence. With the bin-packing algorithms (`first_fit` and `best_fit`, described
below), this list preserves the individual input payloads, but with `wrap` and
`best_fit_wrap`, it contains *slices* of the token field and other length-aligned fields,
so those algorithms should not be used when you need to retain arbitrary fields from the
original records.

**Choosing A Packing Algorithm.** While the packing operator determines the shape of the
packed output records, the packing algorithm determines how Zephon chooses to fill those
packed outputs. Each of the algorithms below can be used with either `pack_flat` or
`pack_sequences`; the main choice that you need to make is whether you need an algorithm
that will maximize the number of useful tokens in each packed example (even if it
requires splitting up the tokens from a single record across multiple packed outputs) or
if you want to be sure that the input records stay intact in the packed outputs (even if
doing this imposes a cost on the efficiency of the packing).

The `wrap` algorithm treats the input data as a first-in, first-out stream of tokens and
fills each packed sequence to `max_length`, splitting input records across packed
sequences when necessary. At each flush, any remaining tokens that cannot fill a complete
sequence are dropped, and Zephon logs a warning with the number of tokens that were
discarded.

```{literalinclude} ../../../examples/guide/pipelines/pack_flat_wrap.py
:language: python
:caption: examples/guide/pipelines/pack_flat_wrap.py
```

If you need to prioritize keeping input records intact, the `first_fit` and `best_fit`
bin-packing based algorithms are your go-to options because they will never split an input
record across different packed outputs. Instead, they maintain a configurable number of
bins and place each incoming record into one of the bins that has enough space to fit the
sample. By convention, the wrap algorithm is commonly used in pretraining, whereas bin
packing algorithms are used in SFT.

As their names suggest, the `first_fit` algorithm will choose the first bin it encounters
that has enough capacity to fit the sample, while the `best_fit` algorithm selects the bin
that would have the least remaining capacity once the input record was placed inside of
it. Both of these algorithms require a `num_bins` argument that controls how many
partially filled sequences are kept open for each packing group. Because `first_fit` and
`best_fit` do not split records, any input records that are longer than the `max_length`
value cannot be packed and will be dropped. At each flush, every open bin is emitted
regardless of whether or not it is full. The `pack_flat` operator will pad these partial
sequences to `max_length`, and `pack_sequences` will emit them as-is, so you can expect up
to `num_bins` partially filled sequences per flush.

```{literalinclude} ../../../examples/guide/pipelines/pack_bin_fit.py
:language: python
:caption: examples/guide/pipelines/pack_bin_fit.py
```

Finally, `best_fit_wrap` is a hybrid packing algorithm that maintains a buffer of
candidate records and selects records that fit together efficiently, but also has the
option to split up input records across packed outputs when it is necessary in order to
finish a packed sequence.

When a partially filled sequence needs to be emitted, `pack_flat` pads it to `max_length`
using the required `pad_token_id` argument. This also applies to the partial sequence
produced by `best_fit_wrap` at each flush. You don't need to mask this padding yourself:
`pack_flat` records how much padding it added, and `SampleBatch.to_training` excludes
those positions from the loss (see
[Building Training Batches](building_training_batches.md)). The `pack_sequences` operator
simply returns the constituent payloads without doing any additional padding.

Most users should opt for the `wrap` algorithm when they are trying to maximize token
utilization and the `first_fit` algorithm when they must keep records intact. The
`pack_flat` and `pack_sequences` entries in the [API reference](../api/pipeline.rst)
cover the `best_fit`, `best_fit_wrap`, and other packing options.

**Packing Homogeneity.** By default, Zephon can pack samples from different upstream
`Dataset` components together into a single training sequence. For many training
pipelines this is perfectly fine, but there are cases where you might want to enforce
some rules about the structure of the samples that the model will encounter in the
training sequence. For example, you might not want to allow general web text to share a
context window with samples that include detailed instructions or reasoning traces, or
where your downstream processing code needs to handle the transformation of different
kinds of data formats in different ways.

To support these use cases, Zephon's packing operators support a `homogeneity` setting
which will enforce rules about how the packing algorithms are allowed to combine samples
from different mixture components into a single packed record. Setting
`homogeneity="full"` requires that each packed output is comprised only of samples that
come from a single mixture component, while setting `homogeneity="group"` and adding in a
dictionary of `groups` that organizes the input `Dataset` names into groups that are
allowed to be packed together will allow samples from the same group to be packed
together:

```{literalinclude} ../../../examples/guide/pipelines/pack_homogeneity.py
:language: python
:caption: examples/guide/pipelines/pack_homogeneity.py
```

The downside of adding the homogeneity constraints is utilization; by default, any sample
can help fill any open packed sequence. With a homogeneity constraint in place, the
packing operators need to maintain separate pools for each permitted group of samples,
which adds overhead and reduces the efficiency of packing for components that are
relatively rare in the training data. Each group is also flushed separately, so every
flush can produce a partial sequence for each group.
