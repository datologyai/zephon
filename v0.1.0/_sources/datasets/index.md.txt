# Working with Datasets

**What is a `Dataset`?** A `Dataset` is how Zephon refers to a collection of training
samples that are organized into individual files (which we refer to as *shards*)
underneath a common storage location (like a filesystem directory or an object storage
bucket). Typically, a `Dataset` is a semantic grouping of files based on some property —
for example, a `Dataset` might cover all "wikipedia" data, or all "Spanish" data. In many
setups, shards then are (often pre-shuffled) subsets of those samples without further
semantic grouping. We use shards because, when we train models using data in the cloud,
downloading individual samples has too much overhead. A collection of samples (e.g., in
64 MB shards) gives us a more tangible object to work with. We assume all shards within a
`Dataset` follow the same file format (e.g., all files in a directory are
jsonl/parquet/litdata/…).

To be precise, a `Dataset` is an abstraction in the context of Zephon's built-in
`StaticMixtureWorkSource`, as detailed in
[WorkSources: The Training Curriculum](../worksources/index.md). We will discuss
WorkSources later on, but note that if you use another `WorkSource` implementation, it
might use different concepts.

**How to define a `Dataset`.** For training runs, you normally define a `Dataset` using
the `Dataset.from_path` factory method, giving it a `name` for the dataset and a location
for where to find its shards:

```{literalinclude} ../../../examples/guide/datasets/dataset_from_path.py
:language: python
:caption: examples/guide/datasets/dataset_from_path.py
```

The `name` is how you refer to the `Dataset` later when you are constructing a mixture for
the run via a `WorkSource`. When the `Dataset` is constructed, Zephon collects the
metadata that the `WorkSource` will need to plan the training run, including the shards
and how many records each shard contains, without necessarily reading the individual shard
files themselves.

Zephon can read this metadata from an `index.json` file stored alongside the shards, for
every supported data format. Some formats, such as
[litData](binary_shard_formats/litdata.md), already include this index as part of their
format specification; for others, such as [Parquet](parquet.md), it can be generated
during data preparation. We recommend always including an index so Zephon can gather the
metadata needed for planning without inspecting each shard individually. Without one,
Zephon will have to scan the shards to construct it before training can start.

**Inspecting a `Dataset`.** A `Dataset` is a lightweight descriptor of the collection and
its metadata; it does not provide methods for reading samples directly. You can use the
`DatasetInspector` class if you need to examine the raw samples inside of a `Dataset`
definition interactively, without needing to explicitly construct a `WorkSource` or a
`Pipeline`. Zephon also provides an in-memory backend via the `Dataset.from_dict` factory
method, which allows you to construct a `Dataset` instance that can be used for testing
without needing to define an external data source.

```{literalinclude} ../../../examples/guide/datasets/dataset_inspector.py
:language: python
:caption: examples/guide/datasets/dataset_inspector.py
```

The next several pages give an overview of the various shard formats that have built-in
support in Zephon and outline the benefits and drawbacks of each format for different
model training use cases, along with any format-specific additional configuration options
you should be aware of. We close on how to effectively utilize Zephon's caching when you
are using object storage (S3, GCS, etc.) or Hugging Face as the storage backend.

```{toctree}
:maxdepth: 2

jsonl
parquet
binary_shard_formats/index
storage_backends_and_shard_cache
```
