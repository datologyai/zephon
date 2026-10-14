# Preparing Samples

Every `Pipeline` begins with an implicit `fetch` operation that pulls in record payloads
from the underlying shards based on sample identifiers in work chunks provided by the
`WorkSource`. For many training workloads, we will need to transform those raw samples
into a format that our model expects, and this page describes the tools that the
`Pipeline` API provides to process a single input record at a time; more complex
operations that involve operating on collections of records will be covered later.

You can think of a pipeline like a chain of transformations that change the representation
of each sample. At the start, a sample is a pointer to a sample in a shard; Zephon's
implicit fetch operation transforms this pointer to the actual payload data (e.g., a
string). Now all of the following downstream operators can further transform this sample
(e.g., tokenize the string), delete (filter out) the sample, spawn new samples based on a
sample (e.g., split a tokenized sequence every 100 tokens), merge samples (e.g., pack
together multiple short tokenized sequences), etc. The samples "flow" through the system
until they reach the training framework in their final representation — commonly, a batch
of tensors.

**Tokenizing Text.** By far the most common type of data preparation for model training is
converting a string of text in an input record into a sequence of tokens that can be fed
into the model. The `Pipeline` API provides a number of built-in and optimized operations
for these common data preparation steps. The `tokenize` operator on the `Pipeline` builds
on top of the HuggingFace `transformers` library to support efficient tokenization of
regular text strings:

```{literalinclude} ../../../examples/guide/pipelines/tokenize_text.py
:language: python
:caption: examples/guide/pipelines/tokenize_text.py
```

By default, the `tokenize` operator emits one record for each input sample, no matter how
long it is. If you need to cap the length of each output record, you can set the
`max_length` argument to the longest output you want to emit, and then either set
`truncation=True` to discard any text beyond this limit or `split_long_samples=True` to
split the input sample into any number of output records that will each be at most
`max_length`:

```{literalinclude} ../../../examples/guide/pipelines/tokenize_max_length.py
:language: python
:caption: examples/guide/pipelines/tokenize_max_length.py
```

If the underlying data format you are reading from returns text as bytes instead of as a
Python string, you can insert a `decode_text` operation in order to handle converting the
raw bytes into a string prior to tokenization:

```{literalinclude} ../../../examples/guide/pipelines/decode_text_then_tokenize.py
:language: python
:caption: examples/guide/pipelines/decode_text_then_tokenize.py
```

Finally, if your input record contains a conversation represented in some form of chat
message format, you can use the `tokenize_chat` method to apply a chat template and
tokenize the conversation. This operator comes with other useful options, such as
producing a `loss_mask` field that identifies which tokens should contribute to the
training loss, allowing the model to learn from the assistant's responses without also
training it to predict the user's messages:

```{literalinclude} ../../../examples/guide/pipelines/tokenize_chat.py
:language: python
:caption: examples/guide/pipelines/tokenize_chat.py
```

By default, both tokenization operators replace the original record payload with their
tokenized output. If later operations in your pipeline need access to the original fields,
set `preserve_upstream_payload=True` to retain those fields along with the tokenized
output. We recommend to check out the full docstring of all options.

**Selecting Chat Targets.** `tokenize_chat(loss_scope=...)` accepts:

| Scope | Direct loss targets |
|---|---|
| `all_assistant` (default) | All spans supplied by the span source |
| `last_span` | The final contiguous run of supervised tokens |
| `last_assistant` | All generated ranges of the final raw assistant message |
| `after_last_user` | All generated ranges of assistants after the final raw user |

For `U → A-call → Tool-result → A-answer`, `after_last_user` trains both assistant
messages; `last_assistant` trains the answer. Tool results remain context.
`loss_on_last_turn_only=True` is deprecated in favor of `loss_scope="last_span"`;
`False` maps to `all_assistant`. Supplying both arguments raises an error.

```python
pipeline.expand_chat().tokenize_chat(
    tokenizer=tokenizer,
    chat_template=training_template,
    loss_scope="after_last_user",
)
```

A scope selects rendered tokens. It cannot recover reasoning removed by a chat
template. Expanding prefixes lets a template render each query as final, which
matters for templates that strip historical reasoning. With `after_last_user`,
a conversation ending in an unanswered user has no targets and is dropped.
Truncation can remove a late query's targets; no earlier assistant is substituted.

The default and `last_span` tagged paths use HF's assistant mask directly.
`last_assistant` and `after_last_user` track the raw message owning each generation
block. Their tagged path requires:

- Transformers 4.52 or later and a fast tokenizer.
- Generation blocks directly inside loops over input assistant messages. Multiple
  blocks per assistant are supported. Put assistant EOS inside an owned block.
- Text-only content and JSON-schema tool dictionaries. Normalize dataset roles,
  reasoning and tool-call arguments before expansion and tokenization.

Verification runs automatically at setup and for each new input structure before
output. Every actual render checks owners, ranges and nonempty selected messages.
Successful probes use a bounded process-local LRU cache. `preflight_tokenizers()`
runs setup checks earlier; omitting it never disables runtime checks.

Profiles include message order, tool names and schema keys. Diverse agentic rows
can miss the cache even after warmup. In a Qwen3-0.6B CPU microbenchmark, 600 fresh
rows required 485 probes and took 1.98 ms/row, versus 0.77 ms/row for `all_assistant`;
3,000 fresh rows required 2,421 probes. Budget roughly 2–3× tokenization time for
similar data, and benchmark the actual dataset. Repeated rows benefit more from
the cache. These timings exclude model training and depend on hardware and row size.

Orphan EOS blocks, buffered or nested generation, loop-variable rebinding, and
external templates raise. Missing or incorrectly owned selected content also
raises. Empty selections and targets removed by truncation produce dropped rows.
Finite marker probes cannot prove every content-dependent branch; templates that
transform markers or restrict tool argument schemas may fail conservatively.
Explicit `span_source="prefix_diff"` uses guarded partial renders. Literal mode
records content spans directly.

Use `zephon.validation.verify_chat_template(..., loss_scope="after_last_user",
probes=..., stock_template=...)` for extra fixtures or training/inference template
comparisons in CI. It returns errors and warnings for unexercised generation
sites. `ChatTemplateProbe` describes fixture fields, including optional explicit
marker expectations. This helper supplements mandatory operator verification.
Keep the stock template in inference exports. Changing scopes or templates when
resuming a checkpoint is unsupported.

**Transforming and Filtering Records.** The built-in `Pipeline` operations cover common
data preparation tasks like tokenization, but many data pipelines will have their own
specific schema and requirements that will require some amount of custom processing. Use
the `map_transform` operator when you need to apply a custom Python function to each
record before it is handed off to the next operation. A map transformation receives a
record's payload and returns a new payload that should be propagated through the rest of
the pipeline. This can be used to rename fields, standardize schemas, or filter out some
subset of records that should be excluded from a particular training run by returning the
special value `None`. Let's look at a simple example of cleaning up some data prior to a
tokenization operation:

```{literalinclude} ../../../examples/guide/pipelines/map_transform_then_tokenize.py
:language: python
:caption: examples/guide/pipelines/map_transform_then_tokenize.py
```

To preserve Zephon's deterministic replay behavior, your transformation function should
produce the same output for the same input, including the decision to filter a record. Be
careful not to rely on mutable global state or any unseeded randomness inside of the
transformation function itself. For adding more complex transformations than the
`map_transform` can support, see the section on
[User-Defined Operators](user_defined_operators.md).
