# Binary Shard Formats

**What Is It?** We are using "binary shard formats" (BSFs) as a catch-all term for a
number of different file formats that are designed to support fast and efficient lookups
of individual data samples from a collection of shards. Where these formats differ is in
how they organize records, describe their schemas, and the data preparation tooling and
ecosystems that exist around each format. The detailed information on these small
differences is described in each of the pages.

**What Are They Good For?** BSFs have a couple of properties that make them appealing as a
format for large-scale model training: they provide very fast lookup and deserialization
of samples with complex structure (e.g. nested tensors, images, and video/audio blobs),
which minimizes the amount of processing that data loaders like Zephon need to do in order
to read individual samples. For very large scale training, or for situations where your
data loader needs to do CPU-intensive preprocessing of the samples before they are ready
for the GPU, the efficiency of these formats can simplify your data pipeline setup
significantly.

**What Are They Bad For?** The primary drawback of these formats is that inspecting and
preparing them requires format-specific tooling, and your existing data tools may not
support them as well as they support formats like JSONL and Parquet. It's also the case
that converting an existing dataset into one of these formats takes time and can mean that
you need to maintain another copy of your training data.

**How does Dataset discovery work for it?** litData and Mosaic Data Shards (MDS) normally
ship with an `index.json` file that describes the contents, serialization schema, and
compression settings for a collection of shards in that format. The third BSF that Zephon
supports, Vortex, does not have an `index.json` file that ships with the format itself,
but similar to Parquet and JSONL, Zephon still supports having such an index to speed up
the instantiation. We added an indexing tool in Zephon to construct these indexes offline
from a set of Vortex shards.

**Reading Binary Shards in Zephon.** Each subpage contains a runnable example that writes
a small dataset in that format, configures a `Dataset` to collect its metadata, and uses a
`DatasetInspector` to examine the sample payloads.

```{toctree}
:maxdepth: 1

litdata
mosaic
vortex
```
