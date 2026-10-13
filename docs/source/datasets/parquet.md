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
`pipeline.options(io_options={"metadata_cache_entries": 256})`; set the limit
to zero to disable it. The same setting applies independently to Vortex. This cache is
separate from the shard and row-group caches, and is unaffected by `cache.enabled`.
Footer size grows with columns × row groups, so wide files with many row groups can
need a much smaller entry limit. Each fetch replica has its own limit.

**Reading Parquet in Zephon**

```{literalinclude} ../../../examples/guide/datasets/parquet_dataset.py
:language: python
:caption: examples/guide/datasets/parquet_dataset.py
```

## Sample payloads

Parquet and Vortex return each sample as a dictionary keyed by column name, even for
single-column files. Missing values are represented by `None`.

Top-level numeric list columns become read-only NumPy arrays, preserving their dtype
and fixed-size nested shapes. Call `.copy()` if you need to modify them. This applies
to the thread runner and Torch-backed process transport, which preserves the
read-only flag without an extra reader-side copy. Without Torch, the process runner
uses ordinary NumPy pickling and received arrays can be writable. Lists with null
elements or variable-length nesting remain
Python lists, such as `[1, None]` or `[[1, 2], [3]]`. Numeric lists inside structs also
remain Python lists: `struct{ids: list<int>}` becomes `{"ids": [1, 2]}`.

Top-level integer and floating-point scalars retain their NumPy dtype. Timestamp and
date scalars are `np.datetime64`, and duration scalars are `np.timedelta64`, preserving
the stored unit and nanosecond precision without requiring pandas. Timezone-bearing
timestamps represent the UTC instant; NumPy scalars do not carry the timezone name.
Missing values remain `None`. Other values use Arrow's Python conversion, including
`str`, `bytes`, Python `bool`, and `decimal.Decimal`. Dictionary-encoded strings return
strings. Values nested inside structs or variable-length lists use Python conversion.

`torch.from_numpy()` can warn about read-only arrays. Other tensor conversion helpers
may use that same path; a tensor sharing this storage must not be modified in place.

Compared with the earlier Parquet reader, the shared conversion changes these values:

| Value | Earlier Parquet output | Shared conversion |
| --- | --- | --- |
| Numeric list without inner nulls | Writable NumPy array | Read-only NumPy array |
| Numeric list containing null elements | NumPy array | Python list preserving `None` |
| Missing floating-point scalar | `NaN` | `None` |
| Boolean scalar | `np.bool_` | Python `bool` |
| Variable nested numeric list | Object array of arrays | Nested Python lists |
| Numeric list inside a struct | NumPy array inside a dictionary | Python list inside a dictionary |

Timestamp and date scalars retain their NumPy representation.

## Reader configuration

Parquet read behavior and the decoded row-group cache are
configured together. Arrow's CPU and I/O pools have separate settings because they
are also used by Vortex and Arrow-backed LitData chunks:

```python
from zephon.io import ArrowOptions, ParquetOptions, ParquetRGCacheOptions, StoreOptions

pipeline.options(
    io_options=StoreOptions(
        arrow=ArrowOptions(cpu_threads=1, io_threads=1),
        parquet=ParquetOptions(
            use_threads=False,
            rg_cache=ParquetRGCacheOptions(
                enabled=True,
                root="/mnt/local/zephon-parquet-cache",
                limit_bytes=4 * 1024**3,
            ),
        ),
    )
)
```

The thread settings shown are the defaults. With `use_threads=False`, each fetch
thread decodes its Parquet columns sequentially; separate fetches can still decode
concurrently. Set `use_threads=True` to let a read decode columns through Arrow's CPU
pool. Increasing `arrow.cpu_threads` alone does not enable this per-read parallelism.
The same choice applies to direct reads and decoded-cache misses.

Arrow pool sizes are process-wide, shared by all stores and formats in that process.
They are applied when readers are set up, including inside process workers; importing
a format or doing discovery alone does not resize them. The process runner currently
also sets up operator copies in the driver, so building that pipeline configures the
driver's pools as well. An explicit `None` leaves
that pool unchanged. Different stores cannot have independent pool sizes: the last
applied count wins, and closing a store does not restore an earlier count. Increasing
pool sizes can multiply CPU demand across fetch processes or training ranks, including
ranks that each use the thread runner.

Dictionary options use the same nesting, for example
`{"parquet": {"use_threads": False, "rg_cache": {"limit_bytes": "4gb"}}}`.
The previous top-level `parquet_rg_cache` setting has moved to `parquet.rg_cache`;
the `ParquetRGCacheOptions` class is unchanged. The older `cache.rg_cache_bytes` alias
still works with a deprecation warning. `ZEPHON_PARQUET_ARROW_THREADS` is deprecated
and only supplies pool counts omitted from `ArrowOptions`; explicit counts or `None`
take precedence. Setting an entire options section to `None` resets that section to
Zephon's defaults, rather than leaving its individual pools unchanged.
