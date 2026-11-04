# litData

[litData](https://github.com/Lightning-AI/litData) is a data processing and loading
library developed by [Lightning AI](https://lightning.ai/), the team behind
[PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/). Here, we are interested in
the binary format that `litData` reads and writes as a collection of records with a
consistent schema that are written into individual files called *chunks* along with an
`index.json` file that describes the contents of the chunks. (The chunks correspond to
what Zephon refers to as shards.)

The `litData` format is the one that we use for our large pretraining runs at DatologyAI
for both our text and multimodal models; we adopted the format because we liked the
flexibility that it provided for representing both tensors/numPy arrays and complex nested
Python dictionaries depending on the data schema that we needed for the file, and its
performance as it relies on a fast [PyTree serializer](https://pypi.org/project/optree/).

**Preparing litData Shards.** To use the litData format with Zephon, install the project
with the `litdata` extra, which includes all of the dependencies needed for reading litData
shards:

```{literalinclude} ../../../../examples/guide/datasets/litdata_install.sh
:language: bash
:caption: examples/guide/datasets/litdata_install.sh
```

Let's look at an example of writing out a small litData dataset using the `litData`
library:

```{literalinclude} ../../../../examples/guide/datasets/litdata_write_shards.py
:language: python
:caption: examples/guide/datasets/litdata_write_shards.py
```

Note that this code generates both the shards and the `index.json` file that describes the
shard contents and structure; be sure to keep the index and the shards together if you
move the dataset to or from object storage. More examples of tools for generating and
preparing samples for litData shards are available in
[litData's data preparation guide](https://github.com/Lightning-AI/litData#speed-up-model-training).

**Reading litData Shards.** Once the litData files have been written, you can define a
`Dataset` by passing the dataset location to the `Dataset.from_path` method with the
`fmt="litdata"` argument. Zephon will infer the format if `fmt` is not passed. You can then
use a `DatasetInspector` to examine the reconstructed sample payloads to confirm that the
samples are in the form that you expect:

```{literalinclude} ../../../../examples/guide/datasets/litdata_dataset.py
:language: python
:caption: examples/guide/datasets/litdata_dataset.py
```

Zephon also supports litData shards that are compressed using Zstandard. Depending on the
contents of the shards, this may significantly reduce the size of the shards and thus the
storage/networking cost of moving them around. Keep in mind that the compressed shards
will be fully decompressed before they can be read, so plan for this when
[configuring the size of your local cache](../storage_backends_and_shard_cache.md) if you
are reading the compressed shards from remote storage.
