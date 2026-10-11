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

Numeric lists without null elements are returned as read-only NumPy arrays, preserving
their dtype and fixed-size nested shapes. They share Arrow buffers where possible;
boolean arrays require unpacking. Returned arrays remain valid after the reader closes.
Call `.copy()` on an array if you need to modify it or pass it to `torch.from_numpy`
as a writable tensor. Duplicate rows can share read-only buffers; process transport
may copy those buffers into writable shared memory, so writability after transport
is not guaranteed either way. Numeric scalar values preserve their NumPy dtype even
alongside null rows; strings, bytes and scalar booleans remain Python values.
Lists containing null elements or nested variable-length lists use Python lists:
`[1, None]` retains the missing element, and `[[1, 2], [3]]` retains the uneven nesting.
Neither fits a plain numeric NumPy array without changing its representation. A null
row remains `None`. Both conversion paths preserve values; the NumPy path also retains
the numeric dtype and avoids creating a Python object for each element.
Ordinary Vortex columns do not identify whether the writer originally used NumPy or
Torch, so Zephon does not automatically reconstruct Torch tensors.

**Read caches.** Each fetch replica's store retains up to 256 parsed file footers and shares a 64 MiB
in-memory cache of encoded segments across its Vortex shards. Both caches survive the
temporary readers used for separate fetch groups and are cleared when the store closes.
The segment cache saves repeat reads of encoded bytes; it does not cache decoded samples.
These caches are independent of Zephon's on-disk shard cache.

Configure them through the pipeline's `io_options`:

```python
from zephon.io import StoreOptions, VortexOptions

io_options = StoreOptions(
    vortex=VortexOptions(
        segment_cache_bytes=64 * 1024**2,
        metadata_cache_entries=256,
    )
)
# Pass io_options to pipeline.options(io_options=io_options).
```

The equivalent dictionary accepts human-readable sizes, for example
`{"vortex": {"segment_cache_bytes": "64mb", "metadata_cache_entries": 256}}`.
Set either limit to zero to disable that cache. Each `FetchOp` replica has its own store,
including replicas in the same process: four replicas can retain 256 MiB of encoded
segments and 1,024 footers with the defaults. Replicas do not share cache hits or budgets.
The segment limit accounts for retained segment bytes,
not total process memory: cache overhead, footers, decoded outputs and in-flight reads
are additional, and Vortex applies cache evictions asynchronously. The footer limit is
an entry count, not a byte budget.

Shard contents must remain immutable during reads. Cache keys detect changes completed
before a reader opens; they do not provide an atomic snapshot across `stat` and native
open. Eviction can atomically replace a shard with identical bytes. Replacing it with
different content while a store is reading it is unsupported.

Zephon limits Vortex's process-wide runtime to one background worker when the Vortex
format is first loaded unless `VORTEX_MAX_THREADS` is explicitly set. Set
`ZEPHON_VORTEX_THREADS` before using the format to override either default: a positive
integer changes the limit, and a non-positive value leaves the runtime untouched. The cap applies
to discovery and sample reads without being reapplied for each file open. General worker
initialization also applies the configured limit when Vortex is already loaded, without
importing this optional dependency into workers that do not use it.
