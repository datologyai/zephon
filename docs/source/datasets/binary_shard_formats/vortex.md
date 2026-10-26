# Vortex

[Vortex](https://vortex.dev/) is a columnar file format designed to support both efficient
random access and efficient compressed storage. Like Parquet, it integrates into the Arrow
data ecosystem, which makes it worth considering if your existing data preparation tools
work well with Arrow tables. Unlike litData and MDS, Vortex files are self-describing and
do not ship with a separate `index.json` file to describe how their contents are structured
and stored. You can think of Vortex as a modern Parquet with fast random access. You can
read more about the Vortex format's structure in the
[project's documentation](https://docs.vortex.dev/).

**Preparing Vortex shards.** To use the Vortex format with Zephon, install the project
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

**Reading Vortex shards in Zephon.** Once the Vortex files have been written, you can
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

Vortex structs return dictionaries keyed by field name, including single-field structs.
Files containing bare values, such as strings or numeric lists, return those values
directly. Both use the [same conversion rules as Parquet](../parquet.md#sample-payloads).

**Memory-mapped reads.** By default, Zephon maps local Vortex shards into memory and
reads segments as slices of the map, using the operating system's page cache instead
of the segment cache. Decoding and output conversion can still allocate and copy.
Mapped shards must not change while in use. Set `memory_map=False` to use file IO
and the segment cache instead.

By default (`safe_mmap=True`), Zephon copies array views into owned memory so retained
samples cannot keep evicted shard files on disk beyond the shard cache's budget.
Process and Ray fetch stages skip this extra copy because sending outputs already
copies mapped arrays into independent storage. Set `safe_mmap=False` to keep array
views on the thread or inline runner, accepting that retained samples may keep shard
files on disk after eviction.

**Caching reads.** Zephon caches Vortex metadata in memory to speed up repeated reads.
When memory maps are disabled, it also caches encoded file segments. Each fetch replica
has its own caches, separate from the on-disk shard cache. You can adjust their sizes
through the pipeline's `io_options`; these are the defaults:

```python
pipeline.options(
    io_options={
        "metadata_cache_entries": 256,
        "vortex": {
            "memory_map": True,
            "safe_mmap": True,
            "segment_cache_bytes": "64mb",
        },
    }
)
```

Set either limit to zero to disable that cache. The metadata limit counts footers; the
segment size is a memory target that Vortex can exceed. Decoded samples and other memory
use are additional, so allow some headroom when planning memory usage.

**Worker threads.** Vortex uses a background thread pool shared by readers in the same
process. Zephon defaults to one background worker to avoid multiplying thread counts
across fetch processes and training ranks. Fetch threads can still read concurrently.
You can change the pool size through the pipeline's `io_options`:

```python
pipeline.options(io_options={"vortex": {"worker_threads": 1}})
```
