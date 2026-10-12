# Vortex

[Vortex](https://vortex.dev/) is a columnar file format designed to support both efficient
random access and efficient compressed storage. Like Parquet, it integrates into the Arrow
data ecosystem, which makes it worth considering if your existing data preparation tools
work well with Arrow tables. Unlike litData and MDS, Vortex files are self-describing and
do not ship with a separate `index.json` file to describe how their contents are structured
and stored. You can think of Vortex as a modern Parquet with fast random access. You can
read more about the Vortex format's structure in the
[project's documentation](https://docs.vortex.dev/).

**Preparing Vortex Shards.** To use the Vortex format with Zephon, install the project
with the `vortex` extra which includes the `vortex-data` dependency (note that Vortex
requires Python >= 3.11):

```{literalinclude} ../../../../examples/guide/datasets/vortex_install.sh
:language: bash
:caption: examples/guide/datasets/vortex_install.sh
```

Let's look at an example of writing a small collection of records into Vortex files:

```{literalinclude} ../../../../examples/guide/datasets/vortex_write_shards.py
:language: python
:caption: examples/guide/datasets/vortex_write_shards.py
```

The Vortex python library can also write data from Arrow tables, which provides a path for
converting existing tabular datasets into this format:

```{literalinclude} ../../../../examples/guide/datasets/vortex_from_arrow.py
:language: python
:caption: examples/guide/datasets/vortex_from_arrow.py
```

Although Vortex does not require a separate index file, Zephon provides an indexing tool
that collects and packages the shard names, file sizes, and record counts into an
`index.json` file that helps for efficiently planning the work to be done during training
without requiring Zephon to load each Vortex shard individually:

```{literalinclude} ../../../../examples/guide/datasets/vortex_index.sh
:language: bash
:caption: examples/guide/datasets/vortex_index.sh
```

**Reading Vortex Shards in Zephon.** Once the Vortex files have been written, you can
define a `Dataset` object by passing the dataset location and the `fmt="vortex"` argument
to the `Dataset.from_path` function. Zephon will infer the format if `fmt` is not passed.
You can then use the `DatasetInspector` to examine the decoded sample payloads:

```{literalinclude} ../../../../examples/guide/datasets/vortex_dataset.py
:language: python
:caption: examples/guide/datasets/vortex_dataset.py
```

Note that Vortex handles compression operations within the file format itself, so Zephon
reads the `.vortex` files directly without requiring an extra, decompressed copy of the
shard to be created.
