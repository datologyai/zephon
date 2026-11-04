# ArrayRecord

[ArrayRecord](https://github.com/google/array_record) is a file format from Google built
on [Riegeli](https://github.com/google/riegeli) for fast random access to records. It is
the default storage format for [Grain](https://github.com/google/grain) and is widely
used in JAX and TensorFlow data pipelines. Unlike the other formats in this section,
ArrayRecord has no schema: each record is an opaque byte string, usually a serialized
`tf.train.Example`, a msgpack or pickle payload, or an encoded image.

**Preparing ArrayRecord shards.** To use ArrayRecord with Zephon, install the project with
the `array_record` extra, which includes the `array-record` dependency. Its wheels cover
Python 3.11 and newer on Linux and Apple-silicon macOS; free-threaded Python builds are
not supported.

```{literalinclude} ../../../../examples/guide/datasets/array_record_install.sh
:language: bash
:caption: examples/guide/datasets/array_record_install.sh
```

Each `.array_record` file is one shard. Here we serialize each record with msgpack:

```{literalinclude} ../../../../examples/guide/datasets/array_record_write_shards.py
:language: python
:caption: examples/guide/datasets/array_record_write_shards.py
```

The writer's `group_size` option sets how many records share a compressed chunk. Reading a
single record decompresses its whole chunk, so `group_size:1` gives the fastest random
access, and larger groups trade read speed for a smaller file.

ArrayRecord files are self-describing, so an index is optional. Without one, Zephon opens
every shard during discovery to count its records, and a remote shard has to be downloaded
in full to do that. Build an `index.json` once for any large or remote dataset:

```{literalinclude} ../../../../examples/guide/datasets/array_record_index.sh
:language: bash
:caption: examples/guide/datasets/array_record_index.sh
```

**Reading ArrayRecord shards in Zephon.** Pass the dataset location to `Dataset.from_path`.
Zephon detects the format from the `.array_record` suffix, or you can pass
`fmt="array_record"` explicitly. Every sample's payload is a dictionary holding the
record's bytes under the `data` key, which you decode with a `map_transform`:

```{literalinclude} ../../../../examples/guide/datasets/array_record_dataset.py
:language: python
:caption: examples/guide/datasets/array_record_dataset.py
```

Shards in remote storage are copied into Zephon's local shard cache before they are read,
the same as JSONL and MDS shards. See
[Storage backends and the shard cache](../storage_backends_and_shard_cache.md).
