# Mixing Datasets

**Defining A Static Dataset Mixture.** One of the most powerful levers that we have for
influencing the performance of models is the mixture of datasets that the model sees
during the training process. By combining datasets that contain samples from specific
domains (like math problems or code, or documents that are written in one or more target
languages, or that include detailed instructions or reasoning traces), we can often
significantly improve the performance of a model on the evaluations that we care about
most.

The `StaticMixtureWorkSource` allows you to specify a list of individual `Dataset` objects
and a set of mixture weights that determine how to combine the datasets together to form a
single training pass, or *epoch*. How long an epoch is is defined by the
{ref}`repetition strategy <repeating-samples>`.

```{literalinclude} ../../../examples/guide/worksources/mixture_worksource.py
:language: python
:caption: examples/guide/worksources/mixture_worksource.py
```

In this example, we are mixing samples from two datasets: one that is primarily composed
of code examples, and another that is primarily composed of documents crawled from the
web. The `mixture` specification identifies the desired mixture proportion for each
`Dataset` by its `name` property (`code` and `web`, respectively). In this example, the
weights sum to 1, but they do not have to; Zephon will normalize the relative fraction of
training samples by the sum of the weights, so that specifying the weights as
`{"code": 0.25, "web": 0.75}` will yield the same relative mixture as
`{"code": 1, "web": 3}`.

The mixture weights determine how frequently samples are drawn from each `Dataset`,
independently of how many samples it contains, which means that some datasets will run out
of samples before others. By default, Zephon repeats those datasets as needed in order to
maintain the requested mixture while the remaining datasets finish their first pass.
Settings for controlling how samples are repeated and randomized during training are
covered in [Shuffling and Repeating Samples](shuffling_and_repeating_samples.md).

Once the `StaticMixtureWorkSource` is defined, we can inspect it to estimate the number of
samples it will produce before stopping:

```{literalinclude} ../../../examples/guide/worksources/inspect_worksource.py
:language: python
:caption: examples/guide/worksources/inspect_worksource.py
```

**Token-Aware Mixtures.** By default, Zephon mixes based on the number of *samples* in
each `Dataset`, which is not necessarily the same thing as the number of *tokens* in each
`Dataset`. If your data is already pre-tokenized and pre-packed into samples that each
contain the same number of tokens, sample-aware mixing and token-aware mixing are
equivalent. But if your samples are not prepared in this way, it is entirely possible that
some of your `Dataset`s are made up of samples that contain a very large number of tokens
(e.g. long web documents or books), whereas others may contain only a small number of
tokens per sample (like short text messages, math problems, or code snippets). Since our
goal during model training is usually to optimize the mixture of tokens, we need a
mechanism to adjust a sample-based mixture specification so that it more accurately
reflects the number of tokens in samples from different `Dataset`s.

To solve this problem in Zephon, we use a `TokenEstimation` utility that reads a small but
representative subset of the samples from each of the `Dataset`s to mix, tokenizes each
record, and estimates how many tokens each sample will yield during the training run. The
`StaticMixtureWorkSource` then uses these estimates to adjust the mixture specification so
that the sample-based mixture plan that it generates will approximate the desired
token-based mixture of the combined `Dataset`s. Let's take a look at how this works in
practice:

```{literalinclude} ../../../examples/guide/worksources/token_aware_mixture.py
:language: python
:caption: examples/guide/worksources/token_aware_mixture.py
```

In this example, we are defining a 50/50 training mixture made up of long web documents
and short code samples. If we do not adjust the sample mixture in a token-aware way, the
long web documents will contribute more than 50% of the tokens that the model is trained
on. By configuring `TokenEstimation` when we create the `StaticMixtureWorkSource`, we can
use a small subset of tokenized records from each `Dataset` to adjust the mixture
specification so that the plan that the work source comes up with is more closely aligned
with our desired token mix.

**The Mixture That Reaches The Model.** It is important to remember that the
`StaticMixtureWorkSource` only defines which samples will get processed by the operator
pipeline, even when it is token-aware. Transformations and filters that are applied to the
samples in the `Pipeline` can meaningfully change the composition of the data that the
model is ultimately trained on in ways that need to be accounted for, since they may cause
deviations from the mixture that was defined by the work source. For example, imagine a
filter operator dropping samples with a high toxicity score. Even if you configure a 50:50
mixture from chat messages to books on the WorkSource level, the model will most likely
see more book tokens than chat tokens, as chat messages are more likely to have a higher
toxicity score.

The `Pipeline.ensure_mixture` operator can re-order records after filtering and
transformation to keep the mixture of its outputs closer to the target specified by the
work source; a discussion of the options and tradeoffs that are involved in using it is
provided in the page for the `Pipeline`'s
[Shuffling and Maintaining Mixtures](../pipelines/shuffling_and_maintaining_mixtures.md).
