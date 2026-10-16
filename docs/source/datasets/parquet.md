# Parquet

**What is it?** Apache Parquet is an open-source and widely-adopted columnar file format
for data storage and retrieval. It is ubiquitous in the data warehouse community as a way
of efficiently storing and querying datasets, and it can be read and written by most
databases and analytical data processing tools in every major programming language. It
strikes a reasonable balance between the easy inspection and editing of JSONL and the
training efficiency of binary shard formats.

**What is it good for?** Parquet is the most common choice for training datasets that have
a natural tabular structure, or when the training is being done at a company that has
standardized on Parquet for other data workloads and has lots of tooling to support
working with it.

**What is it bad for?** Parquet's columnar layout is often less efficient when your
training curriculum requires frequent random access to individual records, which is
typically the case. Zephon needs to read and decode the row groups containing the
requested samples, which often requires processing more data than the samples themselves.
This makes the organization of your shards and your data access patterns substantially
more important for you to think about when you are training against Parquet datasets.
Unless there is an organizational reason to use Parquet, we prefer training-optimized
binary shard formats over it.

**How does Zephon collect dataset metadata?** Each `*.parquet` file under the Dataset path
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

**Reading Parquet in Zephon**

```{literalinclude} ../../../examples/guide/datasets/parquet_dataset.py
:language: python
:caption: examples/guide/datasets/parquet_dataset.py
```

## Sample payloads

Each sample is a dictionary keyed by column name, even for single-column files.
Parquet and Vortex use the same conversion rules for top-level columns:

| Stored value | Sample value |
| --- | --- |
| Missing value | `None` |
| Integer or floating-point scalar | NumPy scalar preserving its dtype |
| Timestamp or date | `np.datetime64` |
| Duration | `np.timedelta64` |
| String, bytes, boolean, or decimal | Python `str`, `bytes`, `bool`, or `decimal.Decimal` |
| Numeric list without null elements | Read-only NumPy array preserving dtype and fixed-size nested shapes |
| List with null elements or variable-length nesting | Python list preserving `None` and nesting |
| Struct | Dictionary, with nested values converted to Python types |

Call `.copy()` if you need to modify a numeric array. Numeric lists inside structs
remain Python lists. Temporal scalars preserve their stored units and precision;
timezone-bearing timestamps represent the UTC instant without retaining the timezone name.

## Reader configuration

**Caching reads.** Zephon caches up to 256 Parquet footers in memory per fetch replica.
The `metadata_cache_entries` option controls this limit for both Parquet and Vortex;
zero disables it. Wide files with many row groups have larger footers and may need a
smaller limit.

The decoded row-group cache avoids repeatedly decoding the same data. It stores files
locally and shares them across fetch processes on the same machine. It is enabled with
the shard cache by default; use `parquet.rg_cache` to change its location or size, or
set `enabled=False` to disable it. For example:

```python
pipeline.options(
    io_options={
        "metadata_cache_entries": 256,
        "parquet": {"rg_cache": {"limit_bytes": "4gb"}},
    }
)
```

**Worker threads.** Arrow's CPU and I/O pools are shared within each process, including
by other formats that use Arrow. Zephon defaults both pools to one thread to avoid
multiplying thread counts across fetch processes and training ranks. Parquet also
reads columns sequentially by default; separate fetches can still run concurrently.
These are the defaults:

```python
pipeline.options(
    io_options={
        "arrow": {"cpu_threads": 1, "io_threads": 1},
        "parquet": {"use_threads": False},
    }
)
```

Set `parquet.use_threads=True` to decode columns through Arrow's CPU pool. Increasing
`arrow.cpu_threads` alone does not enable parallel column decoding.
