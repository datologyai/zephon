# Mosaic (MDS)

[Mosaic Data Shards](https://docs.mosaicml.com/projects/streaming/en/latest/), or MDS, is
the binary format used by the MosaicML Streaming data loader. An MDS dataset consists of a
collection of shard files along with an `index.json` file that describes their contents.
Each MDS record is a dictionary of named fields whose encodings are specified when the
dataset is written. These encodings support different kinds of training data, including
text, numerical arrays, and images. If you already have data preparation tooling that
writes MDS shards for MosaicML Streaming, you can use those shards directly in Zephon
without needing to convert them to another file format.

**Preparing MDS Shards.** To use the MDS format with Zephon, install the project with the
`mds` extra, which includes the MosaicML `streaming` library and the dependencies needed
for reading both uncompressed shards and shards compressed using Zstandard:

```{literalinclude} ../../../../examples/guide/datasets/mds_install.sh
:language: bash
:caption: examples/guide/datasets/mds_install.sh
```

Let's look at an example of writing out a small MDS dataset using the `MDSWriter` class
from the `streaming` library:

```{literalinclude} ../../../../examples/guide/datasets/mds_write_shards.py
:language: python
:caption: examples/guide/datasets/mds_write_shards.py
```

The `columns` argument maps each field name to the encoding that should be used to write
its values. For example, a text field can use the `str` encoding while an array of tokens
can use the `ndarray` encoding. Each record that you write must provide values that are
compatible with its declared encoding. More details and examples are available in the
[MosaicML Streaming data preparation guide](https://docs.mosaicml.com/projects/streaming/en/latest/preparing_datasets/basic_dataset_conversion.html).

**Reading MDS Shards.** Once the MDS files have been written, you can define a `Dataset`
by passing in the dataset location to the `Dataset.from_path` function with the
`fmt="mds"` argument. Zephon will infer the format if `fmt` is not passed. You can then use
a `DatasetInspector` to examine the decoded sample payloads:

```{literalinclude} ../../../../examples/guide/datasets/mds_dataset.py
:language: python
:caption: examples/guide/datasets/mds_dataset.py
```

As is the case for other binary shard formats, keep the binary shard files and the
corresponding `index.json` file together when moving the dataset so that Zephon can read
them. Zstandard compression can make the shards cheaper to store and transport, but they
will be fully decompressed before any records are read, so account for their decompressed
sizes when planning your
[local cache settings](../storage_backends_and_shard_cache.md) when processing datasets in
remote storage.
