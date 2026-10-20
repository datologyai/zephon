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

**Caching reads.** Zephon keeps each fragment's decoded column mapping and tombstones per
fetch replica, up to `metadata_cache_entries` (zero disables this). No file stays open
between reads, so each read reopens the data file and rereads its footer.

```python
pipeline.options(io_options={"metadata_cache_entries": 256})
```

**Worker threads.** By default pylance starts one thread per core in every process. Zephon
caps its pools at two workers. pylance decodes on these workers rather than on the fetch
threads, and in our measurements one worker halved large reads while two kept them at full
speed:

```python
pipeline.options(io_options={"lance": {"worker_threads": 2}})
```

pylance sizes its pools once per process from `TOKIO_WORKER_THREADS` and
`LANCE_CPU_THREADS`, so the count applies only if nothing in the process has used pylance
yet, including writing the dataset. Zephon sets those variables only while pylance starts,
so libraries like obstore keep their own pools. Values you set yourself win unless you
pass `worker_threads`; `None` leaves the existing pool configuration unchanged. pylance's
blocking IO pool is not capped and grows briefly during reads.