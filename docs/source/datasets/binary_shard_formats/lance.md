# Lance

[Lance](https://lancedb.github.io/lance/) is a columnar format with fast random access,
wrapped in a versioned table format. A Lance dataset is a directory, not a flat collection
of shards: it holds its data files under `data/`, and one immutable *manifest* per version
under `_versions/`. The manifest lists every **fragment** — a horizontal slice of the
dataset with its own data file and row count.

**How Lance maps onto shards.** One fragment is one Zephon shard. The mapping is forced in
both directions: a fragment cannot be split across shards, because two shards would then
resolve the same cached file, and two fragments cannot be merged into one shard, because
they are two separate files. You control shard size when you write the dataset, through
`max_rows_per_file`.

Because the manifest already records each fragment's file, byte size and row count, it
*is* the index: Zephon discovers a Lance dataset in two range reads, with no `index.json`
to build and without opening a single data file. `python -m zephon.build_index` therefore
has no Lance mode.

**Preparing Lance datasets.** Install Zephon with the `lance` extra, which adds the
`pylance` dependency:

```{literalinclude} ../../../../examples/guide/datasets/lance_install.sh
:language: bash
:caption: examples/guide/datasets/lance_install.sh
```

```{literalinclude} ../../../../examples/guide/datasets/lance_write_shards.py
:language: python
:caption: examples/guide/datasets/lance_write_shards.py
```

**Reading Lance datasets in Zephon.** Pass the dataset directory to `Dataset.from_path`,
with or without `fmt="lance"`:

```{literalinclude} ../../../../examples/guide/datasets/lance_dataset.py
:language: python
:caption: examples/guide/datasets/lance_dataset.py
```

Sample payloads are dictionaries keyed by column name, following the
[same conversion rules as Parquet](../parquet.md#sample-payloads).

**Versions are pinned at discovery.** A Lance dataset directory is mutable — an append
creates a new version — while each manifest is immutable. `Dataset.from_path` resolves the
directory to its newest manifest and makes that manifest the dataset's path, so every rank
and every worker in the run plans against exactly one version, however the directory
changes underneath. Keep `dataset.path` if you want a later run to resume against the same
shards; pass a manifest directly to read an older version:

```{literalinclude} ../../../../examples/guide/datasets/lance_pin_version.py
:language: python
:caption: examples/guide/datasets/lance_pin_version.py
```

**Deleted rows.** Lance deletes rows by recording tombstones rather than rewriting data
files. Zephon reports the surviving row count per shard and maps each row it reads past
the deleted ones, so a dataset you have called `delete()` on reads correctly and needs no
compaction.

**What Zephon refuses.** Every unsupported case is rejected during discovery, with the
reason and the remedy in the message, so a run never fails halfway through:

| Case | Why | Remedy |
|---|---|---|
| A fragment whose columns span several data files, which `add_columns` and `merge` produce | A Zephon shard resolves exactly one file through the shard cache | Rewrite the dataset: `lance.write_dataset(lance.dataset(src), dst)`; `compact_files()` leaves a fragment it has nothing to merge with untouched |
| A data file still holding a column that `drop_columns` removed | A drop rewrites only the manifest, which then no longer says where the remaining columns sit in the file | Rewrite the dataset |
| Blob columns (`lance-encoding:blob`) | Their values live outside the fragment's data files | Store the payloads as ordinary binary columns |
| Data overlay files | Reading only the base values would return rows a later update has already replaced | Compact the dataset |
| Legacy v1 data files | Zephon reads the v2 file format | Rewrite with `data_storage_version="2.0"` or newer |
| Files reached through another dataset's base path (shallow clones, imported files) | The file lies outside the dataset root Zephon resolves against | Rewrite the dataset in place |
| Fragments stored in an external fragment tree | The fragment list is not in the manifest | — |
| A manifest setting a reader feature flag this Zephon does not know | Lance's own compatibility rule: an unknown flag may change how the data decodes | Upgrade Zephon |

**Caching reads.** Zephon retains each fragment's decoded metadata — its column
projection and its tombstones — per fetch replica, separately from the on-disk shard
cache, and keeps no file open between reads. The retention limit is
`metadata_cache_entries`; set it to zero to disable retention. Each shard open still reads
the data file's footer, because pylance's file reader cannot reuse one it has already
decoded.

```python
pipeline.options(io_options={"metadata_cache_entries": 256})
```

**Worker threads.** pylance starts its native thread pools with one thread per core in
every process that reads Lance, which multiplies across fetch processes and training
ranks. Zephon defaults to two workers per pool. That is one more than Vortex's default
because, unlike Vortex, pylance runs all of its decoding on these workers rather than on
the calling fetch threads: a single worker halves the throughput of large contiguous
reads, while two match pylance's uncapped speed. You can change the count through the
pipeline's `io_options`:

```python
pipeline.options(io_options={"lance": {"worker_threads": 2}})
```

pylance has no setting for its pools and sizes them once per process, from the
`TOKIO_WORKER_THREADS` and `LANCE_CPU_THREADS` environment variables, when it is first
used. Zephon sets those only while pylance starts and then restores them, so other
libraries that read `TOKIO_WORKER_THREADS`, such as obstore, keep their own pools. As a
result the count takes effect only if no Lance code has run earlier in the process, and
one set in your environment is respected unless you pass `worker_threads` explicitly.
`worker_threads=None` leaves pylance's defaults alone.

Reads of local files also run on pylance's blocking IO pool, which this count does not
cap. It grows with the number of reads in flight and retires each thread after a few idle
seconds, so a busy reader shows more threads than a Vortex reader would.
