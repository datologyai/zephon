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

Vortex returns samples in the [same form as Parquet](../parquet.md#sample-payloads).

**Caching Reads.** Zephon caches Vortex metadata and encoded file segments in memory to
speed up repeated reads. Each fetch replica has its own caches, separate from the on-disk
shard cache. You can adjust their sizes through the pipeline's `io_options`; these are
the defaults:

```python
pipeline.options(
    io_options={
        "cache": {"metadata_cache_entries": 256},
        "vortex": {"segment_cache_bytes": "64mb"},
    }
)
```

Set either limit to zero to disable that cache. The metadata limit counts footers; the
segment size is a memory target that Vortex can exceed. Decoded samples and other memory
use are additional, so allow some headroom when planning memory usage.

**Worker Threads.** Configure Vortex's background pool through `VortexOptions`:

```python
from zephon.io import ArrowOptions, StoreOptions, VortexOptions

pipeline.options(
    io_options=StoreOptions(
        vortex=VortexOptions(worker_threads=1),
        arrow=ArrowOptions(cpu_threads=1, io_threads=1),
    )
)
```

Vortex defaults to one background worker in Zephon. Calling fetch threads can also
execute Vortex work, so four fetch threads with this setting can still read
concurrently. The background pool is shared within a process; each fetch process or
training rank has its own pool. This count does not limit Vortex's separate blocking
I/O pool. Arrow's CPU and I/O pools are configured independently as described in the
[Parquet reader configuration](../parquet.md#reader-configuration).

An explicit positive count overrides `VORTEX_MAX_THREADS`; zero disables background
workers, while `None` leaves the current pool unchanged. When `worker_threads` is
omitted, an existing `VORTEX_MAX_THREADS` setting is respected. Without either override,
Zephon applies one worker during reader setup. Importing the format or performing
discovery alone leaves Vortex's own pool settings unchanged. Different stores in the
same process share the pool, so the last applied count wins; closing a store does not
restore an earlier count. `ZEPHON_VORTEX_THREADS` is no longer used.
