# JSONL

**What Is It?** [JSONL](https://jsonlines.org/) is a simple text-based format where each
line in a shard contains a [JSON](https://www.json.org/) payload. Its virtues are its
simplicity and interoperability; it's easy for both humans and computers to read, edit,
and write using widely available tools.

By convention, each line of a JSONL file uses the same schema for the JSON records, but
nothing in the JSONL format explicitly enforces this convention. It is possible to read
and process JSONL files where different lines in the same file have different schemas, but
any downstream processing code must be prepared to handle the different record structures
correctly.

**What Is It Good For?** In the context of model training, JSONL datasets are most often
used for small-to-medium sized text datasets, including richer structures like
conversations or reasoning traces. In these settings, the option to examine a sample
quickly and easily is more important than the raw performance of reading samples from
disk.

**What Is It Bad For?** Because each line is a variable-length JSON record, random access
to individual samples is not straightforward in JSONL. This is why, for large-scale
training, we generally prefer binary shard formats. They tend to require less overhead to
find and parse individual records.

**How Does Zephon Collect Dataset Metadata?** In Zephon terms, a JSONL Dataset is a
directory or object-storage prefix containing a collection of `*.jsonl` files. JSONL does
not have a standard index structure that can be used for doing fast shard counts. This is
usually acceptable for JSONL datasets that are used during training given their modest
size, but it does mean that doing metadata discovery for larger JSONL datasets is
comparatively expensive, if no `index.json` is supplied. In this case, when the Dataset is
constructed, Zephon scans the location, counts the `.jsonl` files, and reads each file to
build an index of byte offsets where each sample starts. This means all payloads of the
entire dataset have to be parsed once on start.

**Reading JSONL in Zephon**

```{literalinclude} ../../../examples/guide/datasets/jsonl_dataset.py
:language: python
:caption: examples/guide/datasets/jsonl_dataset.py
```
