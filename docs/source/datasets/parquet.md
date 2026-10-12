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

**Footer cache.** Each fetch replica's store retains up to 256 parsed Parquet footers
across reads. Configure this through
`StoreOptions(cache=CacheOptions(metadata_cache_entries=256))` or
`{"cache": {"metadata_cache_entries": 256}}`; zero disables retention. This is the
same option used by Vortex, applied independently to each format's LRU. The limit
counts entries, not bytes, and applies even when the on-disk shard cache is disabled.
Both formats identify immutable shards by their local path, preserving metadata reuse
when the same bytes are evicted and downloaded again. The decoded Parquet row-group
cache has its own configuration and budget.

**Reading Parquet in Zephon**

```{literalinclude} ../../../examples/guide/datasets/parquet_dataset.py
:language: python
:caption: examples/guide/datasets/parquet_dataset.py
```

## Sample payloads

Parquet and Vortex use the same Arrow-to-sample conversion. A file with one top-level
column returns that column's value directly: a `tokens` column returns an array,
and a `text` column returns a string. Two or more columns return a dictionary keyed
by column name. Adding a second column therefore changes the payload structure.
Code that previously read `sample["tokens"]` from a single-column file should use
`sample` directly. Operations requiring named fields can wrap the payload explicitly
with a transform, such as `lambda tokens: {"tokens": tokens}`.

Numeric lists without null elements return read-only NumPy arrays, preserving dtype
and fixed-size nested shapes. Conversion shares Arrow buffers where possible; boolean
arrays require unpacking. Arrays remain valid after the reader closes. Call `.copy()`
before modifying an array or exposing it to in-place tensor operations. Duplicate
samples may share read-only numeric buffers. Process transport can create writable SHM
copies, so consumers must not rely on writability after transport.

Present numeric scalars retain their NumPy dtype even alongside null values. Strings,
bytes, and scalar booleans are Python values. A missing value remains `None`.
Lists containing null elements or variable-length nesting return Python lists:
`[1, None]` preserves missingness and `[[1, 2], [3]]` preserves uneven nesting.
Mutable fallback containers are independent for duplicate requests.

The converter processes Arrow chunks separately instead of concatenating their numeric
buffers. Parquet still gathers requested rows from decoded row groups before conversion;
read-only NumPy views avoid the additional copy previously made to provide writable arrays.
Neither format generally records whether a column originated in NumPy or Torch, so
numeric arrays are returned as NumPy arrays rather than reconstructed Torch tensors.
