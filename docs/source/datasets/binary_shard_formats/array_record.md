# ArrayRecord

[ArrayRecord](https://github.com/google/array_record) is a file format from Google built
on [Riegeli](https://github.com/google/riegeli) for fast random access to records. It is
the default storage format for [Grain](https://github.com/google/grain) and is widely
used in JAX and TensorFlow data pipelines. Each record is a byte string, which can hold
a serialized `tf.train.Example`, a MessagePack or pickle payload, or an encoded image.

**Preparing ArrayRecord shards.** To use ArrayRecord with Zephon, install the project with
the `array_record` extra:

```{literalinclude} ../../../../examples/guide/datasets/array_record_install.sh
:language: bash
:caption: examples/guide/datasets/array_record_install.sh
```

Each `.array_record` file is one shard. Here we serialize text documents for pretraining
with MessagePack:

```{literalinclude} ../../../../examples/guide/datasets/array_record_write_shards.py
:language: python
:caption: examples/guide/datasets/array_record_write_shards.py
```

The writer's `group_size` option sets how many records share a chunk. With compression
enabled, reading one record decompresses its whole chunk. `group_size:1` lets each record
be decoded independently; larger groups can improve compression but may increase the
work needed for isolated reads. Measure the tradeoff with your access pattern.

An index is optional but highly recommended for any file format in Zephon.
With an `index.json`, Zephon discovers the dataset from the index without reading the
shards. Without one, it reads metadata from every shard; remote discovery does not
download the complete files. For large or remote datasets, build an index to speed up
discovery:

```{literalinclude} ../../../../examples/guide/datasets/array_record_index.sh
:language: bash
:caption: examples/guide/datasets/array_record_index.sh
```

**Reading ArrayRecord shards in Zephon.** Pass the dataset location to `Dataset.from_path`.
Zephon detects the format from the `.array_record` suffix, or you can pass
`fmt="array_record"` explicitly. Each sample's payload is the record's bytes:

```{literalinclude} ../../../../examples/guide/datasets/array_record_dataset.py
:language: python
:caption: examples/guide/datasets/array_record_dataset.py
```

**Decoding records in a pipeline.** Unlike Parquet and Vortex, ArrayRecord does not carry
a schema describing the payloads or specify their encoding. Zephon therefore returns
`bytes`; it cannot infer the Python object that was serialized. Add a `map_transform`
that decodes records using the encoding chosen by the writer.

After running the writing example above, this complete pipeline reads the shards,
decodes their MessagePack records, and prints the original dictionaries:

```{literalinclude} ../../../../examples/guide/datasets/array_record_pipeline.py
:language: python
:caption: examples/guide/datasets/array_record_pipeline.py
```

```text
{'text': "The ocean covers most of Earth's surface."}
{'text': 'Plants use sunlight to turn water and carbon dioxide into sugars.'}
{'text': 'Written language allows knowledge to be shared across generations.'}
```

Shards in remote storage are copied into Zephon's local shard cache before they are read,
the same as JSONL and MDS shards. See
[Storage backends and the shard cache](../storage_backends_and_shard_cache.md).
