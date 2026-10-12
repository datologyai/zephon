<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)"
            srcset="https://raw.githubusercontent.com/datologyai/zephon/main/docs/source/_static/zephon-wordmark-mint.png">
    <img src="https://raw.githubusercontent.com/datologyai/zephon/main/docs/source/_static/zephon-wordmark-teal.png"
         alt="Zephon" width="300">
  </picture>
</p>

<p align="center"><em>Fast, flexible, elastically deterministic data loading for foundation model training</em></p>

<p align="center">
  <a href="https://pypi.org/project/zephon/"><img src="https://img.shields.io/pypi/v/zephon.svg?color=096361" alt="PyPI" /></a>
  <a href="https://github.com/datologyai/zephon/actions/workflows/pytest.yaml"><img src="https://github.com/datologyai/zephon/actions/workflows/pytest.yaml/badge.svg" alt="Unit Tests" /></a>
  <a href="https://github.com/datologyai/zephon/actions/workflows/pytest-integration.yaml"><img src="https://github.com/datologyai/zephon/actions/workflows/pytest-integration.yaml/badge.svg" alt="Integration Tests" /></a>
  <a href="https://datologyai.github.io/zephon/"><img src="https://img.shields.io/badge/docs-user%20guide-096361.svg" alt="Docs" /></a>
  <a href="https://github.com/datologyai/zephon/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-096361.svg" alt="License" /></a>
</p>

<p align="center">
  <a href="https://zephon.io">zephon.io</a> ·
  <a href="https://datologyai.github.io/zephon/">User Guide</a> ·
  <a href="https://www.datologyai.com/blog/zephon">Launch blog post</a> ·
  <a href="https://arxiv.org/abs/2610.03087">Paper</a> ·
  <a href="https://pypi.org/project/zephon/">PyPI</a>
</p>

---

Zephon is a high-performance data loading library that separates **what** you train on
from **how** data is loaded and transformed. It supports online processing (tokenization,
sequence packing, filtering, dynamic mixtures) rather than requiring expensive offline
preprocessing, while guaranteeing deterministic, reproducible output
regardless of parallelism or hardware topology.

We built Zephon at [DatologyAI](https://www.datologyai.com) because running data
experiments is a lot of what we do, and an experiment that compares two datasets is only
useful if nothing else about the two training runs changes. In practice, the data loader
is often the part that doesn't hold still: if a job gets preempted and comes back on fewer
GPUs (which happens more often than any of us would like), many loaders will either
refuse to resume or quietly start feeding it different data. That's a manageable problem
for datasets you can index into directly. It's much harder for pipelines that pack
sequences, since what ends up in each packed sequence depends on every sample that came
before it. Zephon was designed from the start to keep that kind of pipeline deterministic,
so the number of GPUs you happen to have on a given day is one less thing to worry about.

## Installation

Zephon requires Python 3.10 or newer.

```bash
uv pip install zephon
```

Most of what Zephon can read is behind optional extras, so you only install what you use:

| Extra | What it adds |
|---|---|
| `zephon[cloud]` | Reading from S3, GCS, and Azure (`s3://`, `gs://`, `az://` paths) |
| `zephon[hf]` | Hugging Face datasets (`hf://` paths), plus Parquet |
| `zephon[parquet]` | Parquet shards |
| `zephon[streaming]` | MosaicML Streaming / MDS shards |
| `zephon[litdata]` | LitData shards |
| `zephon[vortex]` | Vortex shards (Python 3.11+) |
| `zephon[ray]` | The experimental Ray runner |

For example, to train on Parquet data in S3: `uv pip install "zephon[cloud,parquet]"`.
Tokenization uses Hugging Face tokenizers, so you'll also want `uv pip install transformers`
if you're following the example below.

## A First Pipeline

A Zephon pipeline has three parts: a `Dataset` that describes your data, a `WorkSource`
that decides what to train on and in what order, and a `Pipeline` of operators that turns
samples into training batches.

```python
from zephon import Pipeline
from zephon.io import Dataset
from zephon.work import MixtureSpec, StaticMixtureWorkSource

# 1. Describe the data: a directory of JSONL, Parquet, MDS, LitData, or Vortex shards
dataset = Dataset.from_path("train", "/data/train/")

# 2. Decide what to train on, and in what order
ws = StaticMixtureWorkSource(
    datasets=[dataset],
    mixture=MixtureSpec({"train": 1.0}),
    seed=42,
)

# 3. Turn samples into training batches
pipeline = (
    Pipeline(ws)
    .decode_text()
    .tokenize(tokenizer_id="gpt2", field="text", padding=True)
    .batch(microbatch_size=2)
)

for batch in pipeline:
    train_step(batch.to_training())
```

Mixing datasets is a matter of adding them to the `WorkSource` with the proportions you
want:

```python
ws = StaticMixtureWorkSource(
    datasets=[
        Dataset.from_path("fineweb", "s3://my-bucket/fineweb/"),
        Dataset.from_path("dclm", "s3://my-bucket/dclm/"),
    ],
    mixture=MixtureSpec({"fineweb": 0.7, "dclm": 0.3}),
    seed=42,
)
```

A `Pipeline` is a plain Python iterable, so it works with whatever training loop you
have. It will also run inside a PyTorch `DataLoader` or torchdata `StatefulDataLoader` if
your framework insists on one, although we recommend iterating over it directly.

## Checkpointing and Elastic Resume

You save and restore a pipeline's position alongside your model checkpoint:

```python
state = pipeline.checkpoint()  # store this with the model and optimizer state

pipeline.restore(state)  # on resume, before iterating
for batch in pipeline:
    ...
```

Resuming on a different number of GPUs comes down to one setting that you pick at the
start of the run: the number of *lanes* (`canonical_replicas`) that Zephon divides the
data into. Each data-parallel group owns an equal share of the lanes, so you can run or
resume on any data-parallel size that divides the lane count evenly, as long as each
optimizer step consumes a multiple of `canonical_replicas` batches. For example, a
lane count of 32 lets you run on any of 8, 16, or 32 data-parallel groups. Only the data-parallel degree
counts here; tensor, pipeline, and context parallelism don't affect it.

What Zephon guarantees is that every global batch contains the same data after a resume.
That's not quite the same as promising identical losses or weights, which also depend on
things like the order of reductions and which kernels get selected, and those are outside
of anything a data loader can control.

## Training Framework Integrations

We maintain reference integrations for two training frameworks, each in its own fork with
a README, launch commands, and smoke tests:

- **TorchTitan**: [datologyai/torchtitan-zephon](https://github.com/datologyai/torchtitan-zephon/tree/main/examples/zephon)
- **Megatron-LM**: [datologyai/Megatron-LM-zephon](https://github.com/datologyai/Megatron-LM-zephon/tree/main/examples/zephon)

The [Training Integrations](https://datologyai.github.io/zephon/training_integrations.html)
chapter of the User Guide explains the decisions behind them, which is the place to start
if you want to connect Zephon to a different framework.

## How It Works

Under the hood, Zephon is organized into three layers:

```
WorkSource          what to train on (datasets, mixtures, shuffling)
    │               emits lightweight pointers: (dataset, shard, sample)
    ▼
Pipeline            how to process it (decode, tokenize, pack, batch, ...)
    │               a chain of operators
    ▼
Engine              where and when to run it (threads, processes, queues)
                    deterministic scheduling with backpressure
```

The WorkSource produces a deterministic sequence of sample pointers, working mostly from
metadata like shard listings and sample counts rather than the samples themselves.
The Engine compiles the Pipeline into concurrent stages connected by bounded queues, so
fetching, processing, and your training step all overlap, and it does this without letting
the amount of parallelism change the order of the output. If you're curious what it came
up with, `pipeline.explain()` will show you the compiled plan.

## Documentation

The [Zephon User Guide](https://datologyai.github.io/zephon/) covers all of this in much
more detail:

- [Why Zephon?](https://datologyai.github.io/zephon/why_zephon.html): elastic determinism and why it matters for experiments
- [Basic Concepts](https://datologyai.github.io/zephon/basic_concepts.html): Datasets, WorkSources, and Pipelines
- [Working with Datasets](https://datologyai.github.io/zephon/datasets/index.html): shard formats, storage backends, and the shard cache
- [WorkSources](https://datologyai.github.io/zephon/worksources/index.html): mixing, shuffling, and repeating samples
- [Pipelines](https://datologyai.github.io/zephon/pipelines/index.html): operators, packing, distributed training, and checkpointing
- [Training Integrations](https://datologyai.github.io/zephon/training_integrations.html): the TorchTitan and Megatron-LM reference integrations
- [API Reference](https://datologyai.github.io/zephon/api/zephon_pipeline.html)

## Status

Zephon is beta software, and the Ray runner in particular is experimental. If something
doesn't work the way this README or the User Guide says it should, please
[open an issue](https://github.com/datologyai/zephon/issues).

## Citing Zephon

If you use Zephon in your research, please cite
[our paper](https://arxiv.org/abs/2610.03087):

```bibtex
@misc{boether2026zephon,
  title         = {Zephon: Elastic Determinism for Online, Stateful Foundation Model Data Loading Pipelines},
  author        = {B{\"o}ther, Maximilian and Wills, Josh and Robroek, Ties and Xu, Sonnet and
                   Burstein, Paul and Zayas, Daniel and Blakeney, Cody and Joshi, Siddharth and
                   Yin, Haoli and Adiga, Rishabh and Mongstad, Haakon and Merrick, Luke and
                   Maini, Pratyush and Morcos, Ari and Leavitt, Matthew and Klimovic, Ana and
                   Gaza, Bogdan},
  year          = {2026},
  eprint        = {2610.03087},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2610.03087}
}
```

## License

Zephon is released under the [Apache 2.0 License](https://github.com/datologyai/zephon/blob/main/LICENSE).
