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
with the `vortex` extra, which requires `vortex-data>=0.88.0` and Python >= 3.11:

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

Without an index, Zephon reads each shard's footer through the configured storage
backend, including for remote paths, using the backend's existing credentials. Small
files may fit entirely in the footer read, but discovery never scans sample rows.

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

Vortex and Parquet share the same [sample payload contract](../parquet.md#sample-payloads).
A single top-level column returns its value directly; multiple columns return a
dictionary. Numeric arrays are read-only NumPy arrays with their original dtype and
fixed-size shape, and remain valid after the reader closes. Numeric lists containing
null elements use a Python-list fallback. Call `.copy()` before mutating an array.

**Read caches.** Each fetch replica has a footer cache (256 entries by default) and an
encoded-segment cache (64 MiB target), shared across its Vortex readers. These are
independent of Zephon's on-disk shard cache. The segment cache reuses encoded bytes;
it does not retain decoded samples.

Configure them through the pipeline's `io_options`:

```python
from zephon.io import CacheOptions, StoreOptions, VortexOptions

io_options = StoreOptions(
    cache=CacheOptions(metadata_cache_entries=256),
    vortex=VortexOptions(segment_cache_bytes=64 * 1024**2),
)
# Pass io_options to pipeline.options(io_options=io_options).
```

The equivalent dictionary accepts human-readable sizes, for example
`{"cache": {"metadata_cache_entries": 256}, "vortex": {"segment_cache_bytes": "64mb"}}`.
The footer limit applies independently to Parquet and Vortex, whether or not the on-disk
shard cache is enabled. Set either limit to zero to disable that cache. Replicas have
separate caches and budgets, including when they share a process.

Segment-cache memory can exceed its byte target because eviction is asynchronous and
views can retain larger buffers. Metadata, decoded output, in-flight reads and cache
overhead consume additional memory. The footer limit counts entries rather than bytes.

Cache keys use the local shard path, which must identify the same contents throughout
the store's lifetime. Eviction and re-download of identical bytes preserve cache reuse.

**Threads.** Zephon defaults to one Vortex background worker per process and respects an
explicit `VORTEX_MAX_THREADS`. Set `ZEPHON_VORTEX_THREADS` before using the format to
override either setting: a positive integer sets the worker count, and a non-positive
value leaves the runtime unchanged. This applies to discovery and sample reads.
