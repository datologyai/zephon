# Building training batches

The final step of a `Pipeline` is turning the stream of prepared records into the batches
that the training loop consumes. This happens in Zephon in two parts: the `batch` operator
on the `Pipeline` groups records together and returns a `SampleBatch` object, and the
`SampleBatch.to_training` method converts each of these objects into the tensors and
fields that your model training loop expects. This page covers both of these essential
bits of functionality to bridge the gap between your data loader pipeline and your model.

**Grouping records into batches.** The `batch` operator should generally be the last
operator in a `Pipeline`, and it is the only one that is always required for training. It
collects records into groups of `microbatch_size` records each and returns a `SampleBatch`
object that wraps those records and the metadata they accumulated as they moved through
the pipeline operators:

```{literalinclude} ../../../examples/guide/pipelines/batch_operator.py
:language: python
:caption: examples/guide/pipelines/batch_operator.py
```

By default, the `batch` operator will drop any incomplete final batches that are left over
at the end of a pipeline run, since most training loops expect every batch to have the
same shape. If your training loop can handle smaller batches, you can set `drop_last=False`
in order to return incomplete batches as well. Note that in a `Pipeline` that uses flushes,
this includes a short batch at each flush — not just at the end of the run. With the
default `drop_last=True` setting, flushes never produce short batches.

**From a packed record to a training batch.** Here is what the `to_training()` method
produces from a single packed record that contains two (extremely) short documents, `A`
and `B`:

```text
packed record:  BOS  a1   a2   EOS  BOS  b1   b2   EOS

inputs:         BOS  a1   a2   EOS  BOS  b1   b2
labels:         a1   a2   EOS  BOS  b1   b2   EOS
counts toward   ✓    ✓    ✓    ✗    ✓    ✓    ✓
the loss?                      ↑ optional: the EOS → BOS prediction

positions:      0    1    2    3    0    1    2      (per document)
            or: 0    1    2    3    4    5    6      (continuous)
documents:      [0, 4, 7]                            (where each one starts and ends)
```

Each row of this diagram is something that the model needs from the batch: the tokens that
it reads, the tokens that it should predict, which of those predictions should be used to
update the model weights, and how the tokens in the record are related to each other.

**What the model reads and predicts.** A language model learns by predicting each token
from the tokens that came before it, so the inputs to the model and the labels it predicts
come from the same sequence, just offset by one position. For our example, this starts
when the model reads `BOS` and should predict `a1`, then reads `a1` and should predict
`a2`, and so on. Setting the `return_labels=True` option on the `to_training()` method
produces both halves of this input/label pair from each record, as the `input_ids` and
`labels`. Note that because of this, the `input_ids` and `labels` will be one token shorter
than the input record that they are constructed from, so if you are training a model with
a context length of `L`, you should pack your sequences to `L + 1` tokens, which is what
we do in our own training framework integrations.

**Which predictions matter.** Some of the labels in a batch are predictions that we don't
actually want the model to learn from. The `to_training()` method needs to mark each of
these with an "ignore" value (`-100` by default) so that they do not contribute to the
loss. There are three different types of labels that we do not want to predict on:

1. **Padding tokens.** Padding tokens are simply there to make the records into a fixed
   length, they do not provide information that we want the model to learn. Zephon records
   how much padding each record contains when it adds padding, so these labels are always
   masked.
2. **Tokens that you do not want to train on.** For example, in a chat conversation, we
   want the model to learn to produce the assistant's replies but not the user's messages.
   The `tokenize_chat` operator records which tokens should be trained on, and the
   `to_training` method masks the rest.
3. **Predictions across a document boundary.** The general idea here is that we do not
   want the tokens at the end of one document to be used to learn things about the
   beginning of the next, unrelated document in the sequence. Some training recipes exclude
   this prediction. Set `mask_segment_starts=True` to mask targets at zero in the
   original scalar `positions`, independently of token IDs. The first token of each
   packed document is already outside the target slice, including wrap continuations.
   This composes with existing supervision and padding masks and does not change
   attention or model positions. Alternatively, `eos_mask_loss=True` with an explicit
   EOS ID masks predictions **from EOS tokens**; if BOS and EOS share an ID, it also
   masks predictions from BOS, so it is not an exact boundary policy.

Different training frameworks want this masking information represented in different ways.
Some frameworks read the ignored values directly from the labels, while others expect a
separate `loss_mask` tensor (which can be provided via the `return_loss_mask=True`
setting) or a count of the labels that should count toward the loss so that they can do
normalization of the loss across microbatches (via `return_num_valid_tokens=True`).

**How tokens relate to each other.** When documents are packed together, the model sees
one long sequence, but the tokens that it contains belong to separate documents. The
`pack_flat` operator records this structure as a list of position indices that resets to
zero at the beginning of each document, and the downstream model can use that information
in two different ways.

The first is the positions: position embeddings like
[RoPE](https://arxiv.org/abs/2104.09864) tell the model how far apart two tokens are. By
default, `to_training` passes along the per-document positions from `pack_flat`, so that
every document starts at position zero. Some training recipes number the entire packed
sequence continuously instead, as if it was a single long document, which you can get from
`to_training` by passing in `position_mode="sequence"`.

The second use of this structure is controlling attention across document boundaries.
Resetting positions for position embeddings does not, by itself, stop tokens in document B
from attending to document A. The attention implementation must also use the document
boundaries to restrict which tokens can attend to each other.

Different training frameworks consume these boundaries differently. Variable-length
attention kernels use cumulative sequence lengths, such as `[0, 4, 7]` in our example.
Setting `return_cu_seqlens=True` provides this representation directly, as used by our
[Megatron integration](../training_integrations.md) when inter-document masking is
enabled.

Other integrations derive the boundaries from the per-document positions that
`to_training()` preserves by default. In our example:

```text
inputs:      BOS  a1  a2  EOS   BOS  b1  b2
positions:    0    1   2   3     0    1   2
documents:   └───── A ─────┘    └──── B ───┘
```

The reset to zero at the second `BOS` identifies the start of document `B`. Our TorchTitan
integration follows this approach: its FlexAttention path builds a causal attention mask
that restricts each token to its own document, while its variable-length attention path
converts the position resets into cumulative sequence lengths internally. Neither path
needs Zephon to return `cu_seqlens`.

For integrations that infer boundaries from position resets, keep
`position_mode="preserve"`; choosing `"sequence"` removes the resets that identify
documents within each packed record. Setting `return_padding_mask=True` additionally
identifies padding tokens so that the framework can distinguish them from document
content. The required fields depend on the attention implementation your model uses.

Which of these choices is right depends on your model and training recipe and not your
data loader, so the `to_training` method passes along the per-document positions by
default and leaves the rest of the options as opt-in requests if you need them. Again, we
emphasize that some of the options of `to_training()` stack, whereas others contradict,
and it's always a question of the recipe: you might want to disable cross-document
attention but not mask the loss between document boundaries, or vice versa, etc. Zephon
provides merely the knobs to enable such different recipes.

**Matching your framework.** Everything up to this point affects what the model learns.
The remaining options on `to_training` are purely mechanical to help you arrange the
output the way that your training loop expects to receive it. The `rename_fields` option
allows you to map from Zephon's field names onto the ones your framework uses (e.g.,
mapping from Zephon's `"input_ids"` to your framework's `"tokens"`), `exclude_fields`
allows you to drop any fields that your training loop doesn't need, and `flatten=True`
turns each `[microbatch_size, sequence_length]` tensor into a single flat sequence for
training loops that expect that shape.

Our two reference integrations show how differently two training frameworks can make the
same set of choices here:

```{literalinclude} ../../../examples/guide/pipelines/to_training_integrations.py
:language: python
:caption: examples/guide/pipelines/to_training_integrations.py
```

The [Training integrations](../training_integrations.md) page explains the context behind
each of these choices, and the API reference for `SampleBatch.to_training` describes every
option in detail.

**Other kinds of data.** The `to_training` method is currently designed for token
sequences. If you are training on other kinds of data, like images or audio, you can build
the tensors for your model directly from the payloads of the records in each
`SampleBatch`:

```{literalinclude} ../../../examples/guide/pipelines/non_token_batch.py
:language: python
:caption: examples/guide/pipelines/non_token_batch.py
```

Note that the `to_training` method cannot convert the output of the `pack_sequences`
operator, since it keeps each packed record as a list of individual documents. Use
`pack_flat` if you want `to_training` to do the conversion for you, and `pack_sequences`
if you need to work with the individual documents yourself.


**Padding unequal documents.** Use `pad_to_length=L + 1, pad_token_id=...` to
right-pad unshifted documents before next-token conversion. Longer documents raise an
error rather than being truncated. Existing padding metadata and newly added padding
are combined; `return_padding_mask=True` identifies input padding by position, even
when PAD and EOS use the same ID. Ordinary token-aligned extra fields and supervision
masks are padded with zero. Scalar `positions` give the padding tail its own segment.
Omitting `pad_to_length` preserves the existing stacking behavior.

**Media and multi-axis positions.** `field_collators` registers a callable per named
payload field; registration also includes that field in the output. The callable runs
once for each batch with token-layout metadata passed by keyword:

```python
def collate_field(values, *, length, shifted):
    ...
```

`values` contains the original unpadded values in document order. `length`
is the resolved padded token length (or the common token length without padding).
For example, a media callback concatenates image tensors along their media axis;
a model-position callback pads each `[3, N]` value to `length`, removes
its last position when `shifted=True`, and stacks to `[3, B, S]`. The callback
must not mutate its inputs or retain per-call state. Its result is not automatically
cast, sliced or flattened; output renaming and exclusion still apply. Empty batches
invoke the callback with `values=()` and `length=pad_to_length` (or zero when
padding is omitted). Return the correctly typed empty value or raise a clear
`ValueError` if the callback does not support empty batches. All named
fields must exist in every document, and reserved fields such as `input_ids`, `labels`,
`positions` and `loss_mask` cannot be overridden. Keep scalar boundary positions
separate from model-specific multi-axis positions.

These field callbacks only assemble already prepared values. They do not select bins,
shift input IDs/labels, drop records or select an attention policy.


**Structured packs.** `pack_sequences` groups complete payloads with its existing
`first_fit` or `best_fit` algorithm. A `pack_payloads` callback can assemble each
selected pack into one mapping of **unshifted** `input_ids`, token-aligned `loss_mask`,
scalar subdocument-reset `positions`, and ordered media/model-position fields.
The result remains inside `{"packed_samples": mapping}`; Zephon retains the original
record's contributor lineage. Explicitly unwrap it before batching:

```python
from operator import itemgetter

pipeline.pack_sequences(..., pack_payloads=merge_documents)
pipeline.map_transform(itemgetter("packed_samples"))
pipeline.batch(batch_size)
```

`map_transform` changes only the payload; the record metadata and contributor
lineage stay intact. `to_training` then performs its ordinary shift, padding and
field collation on the resulting mapping. The converter does not inspect packing
envelopes or select packing algorithms. The callback defines how the model fields merge.

Measure packing length after expanding image placeholders into the actual token
slots. For `L` model input tokens use capacity `L + 1`, and charge each complete
document its full expanded token count. The merge callback must preserve that
unshifted length and media order. It must not silently remove a terminal media
slot to manufacture the next-token shift.


**Closing a pipeline.** Call `pipeline.close()` in the loader's cleanup path when
training stops early or initialization fails. It closes paused/unstarted iterators
and any runtime created by `checkpoint()` before iteration (for example, while
preparing a distributed-checkpoint destination). Closing a fresh pipeline does not
initialize it, and repeated closes are safe. Call it between iteration steps from
the owning thread; it is not a concurrent cancellation API. Save checkpoints using
`checkpoint()` as usual, then use `restore()` before resuming a closed stream.
`close()` does not create an implicit checkpoint or discard a pending restore.
