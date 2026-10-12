# Parquet

**What Is It?** Apache Parquet is an open-source and widely-adopted columnar file format
for data storage and retrieval. It is ubiquitous in the data warehouse community as a way
of efficiently storing and querying datasets, and it can be read and written by most
databases and analytical data processing tools in every major programming language. It
strikes a reasonable balance between the easy inspection and editing of JSONL and the
training efficiency of binary shard formats.

**What Is It Good For?** Parquet is the most common choice for training datasets that have
a natural tabular structure, or when the training is being done at a company that has
standardized on Parquet for other data workloads and has lots of tooling to support
working with it.

**What Is It Bad For?** Parquet's columnar layout is often less efficient when your
training curriculum requires frequent random access to individual records, which is
typically the case. Zephon needs to read and decode the row groups containing the
requested samples, which often requires processing more data than the samples themselves.
This makes the organization of your shards and your data access patterns substantially
more important for you to think about when you are training against Parquet datasets.
Unless there is an organizational reason to use Parquet, we prefer training-optimized
binary shard formats over it.

**How Does Zephon Collect Dataset Metadata?** Each `*.parquet` file under the Dataset path
is a shard. Parquet embeds useful metadata in each file, including its row count, so
Zephon can discover a Dataset without a separate index file and without having to parse
each record in the file, as it does for JSONL files. Still, the footer of each file needs
to be read, which at large scale can take a significant amount of time.

For larger Parquet datasets that are comprised of a large number of shards, Zephon
includes an indexing tool that can be used to create an `index.json` file that the Dataset
can use to quickly collect shard metadata information without needing to parse each of the
individual shards to collect their metadata for faster discovery at the cost of a one-time
index creation run:

```{literalinclude} ../../../examples/guide/datasets/parquet_index.sh
:language: bash
:caption: examples/guide/datasets/parquet_index.sh
```

If you have existing infrastructure for writing out Parquet files using distributed data
processing frameworks like Spark or Ray, you might want to write out an `index.json` file
while creating the dataset.

**Caching Metadata.** Zephon caches up to 256 Parquet footers per fetch replica to avoid
rereading metadata. You can change this with
`pipeline.options(io_options={"cache": {"metadata_cache_entries": 256}})`; set the limit
to zero to disable it. The same setting applies independently to Vortex. This cache is
separate from the shard and row-group caches.

**Reading Parquet in Zephon**

```{literalinclude} ../../../examples/guide/datasets/parquet_dataset.py
:language: python
:caption: examples/guide/datasets/parquet_dataset.py
```

## Sample payloads

Parquet and Vortex return each sample as a dictionary keyed by column name, even for
single-column files. Missing values are represented by `None`.

Numeric lists become NumPy arrays, preserving their dtype and fixed-size nested shapes.
These arrays may be read-only, so call `.copy()` if you need to modify them. Lists with
null elements or variable-length nesting remain Python lists, such as `[1, None]` or
`[[1, 2], [3]]`. Numeric scalars retain their NumPy dtype; other values use Python types,
such as `str`, `bytes`, `bool`, and `datetime.datetime`.
