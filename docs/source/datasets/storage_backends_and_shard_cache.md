# Storage Backends and Shard Cache

Zephon can read datasets from a variety of sources: local paths, distributed filesystems,
directly from object storage systems, from data services like
[Hugging Face](https://huggingface.co/), and via
[FUSE](https://en.wikipedia.org/wiki/Filesystem_in_Userspace)-backed filesystem mounts.
Distributed filesystems and FUSE mounts are infrastructure choices that are managed
outside of Zephon with their own setup, authentication, and performance optimization
settings that are specific to the environment you are running in and we cannot provide
guidance on the best way to configure them for model training here. Our own training stack
at DatologyAI makes heavy use of object storage for storing and managing our datasets, and
so Zephon ships with its own caching infrastructure that we have developed to efficiently
move shards from object storage onto compute nodes during model training. We acknowledge
the great work of
[Mosaic Streaming](https://docs.mosaicml.com/projects/streaming/en/latest/dataset_configuration/shard_retrieval.html#configure-shard-storage)
on their shard cache. The rest of this page describes how to configure and use that setup
effectively. Depending on your distributed filesystem/FUSE setup, you might also want to
rely on Zephon's cache.

**Reading from Object Storage.** The `path` argument that you provide in the
`Dataset.from_path` method can have the form of a URI like `s3://bucket/prefix`,
`gcs://bucket/prefix`, `az://container/prefix`, or `hf://org/name/split`. In the case of
`s3`, `gcs`, or `az`, Zephon will collect the shard metadata and handle downloads from the
object store directly using the excellent (and fast!)
[obstore](https://github.com/developmentseed/obstore) library. For
`hf://` URIs, Zephon reads the originally uploaded files when their format is supported,
falling back to Hugging Face's automatically converted Parquet shards when needed and
available. The path must identify the split you want to read such as `train` or
`validation`. If you also need to select a particular dataset configuration, include its
name before the split: for example, `hf://org/name/config/train`.

Whenever you are training from datasets in remote locations (e.g., object storage, Hugging
Face, or a DFS) we recommend that you enable Zephon's *shard cache*. The shard cache
functions — as implied by the name — as a node-local intermediate storage for the shards
we are currently training on, to avoid repeated download of the same files from the remote
location.

This requires there to be a local filesystem directory on each of the training nodes that
is used for storing copies of the shards that the workers on that node are currently
processing samples from. You need to configure this local cache directory and its size
limit before starting the run:

1. The name of the local filesystem directory that should be used for caching shards: the
   default directory if you do not specify one is `~/.cache/zephon`; we recommend that
   this directory is backed by very fast local storage like NVMe.
2. A size limit for the shard cache: before preparing another shard, Zephon will evict the
   least recently used shards from the cache as needed to make room within this limit. The
   size limit must be high enough to accommodate the shard being prepared, including
   compressed and decompressed copies if the upstream format is compressed. Although this
   limit is disabled by default, we strongly recommend setting a limit that is safely
   below the disk capacity available to the workers.

Note that this cache is another reason we have introduced the shard abstraction above
samples. Having one file per sample would, especially for cases with smaller samples such
as text, have too much overhead per download. The downside of this approach, however, is
that a truly global shuffle is extremely expensive — we might fetch an entire shard to
only access a single sample in it. Rather, as discussed in
[Shuffling and Repeating Samples](../worksources/shuffling_and_repeating_samples.md) we
can shuffle the shard order, within the shards, and maintain a tumbling shuffle window
across the shards. We recommend shards of 64–256 MB.

In setups with multiple GPUs per server, keep in mind that the cache is shared among all
of the GPU ranks using that same cache directory on the node: this means that if one rank
needs a sample from a shard, but the shard is already being downloaded by a second rank,
the first rank will wait for the download to finish and will not attempt to re-download
the same shard twice. This also means that the cache size limit needs to be large enough
to ensure that enough shards can be downloaded to the node so that all of the ranks on the
node have enough data to process. Setting the cache size limit too low means that the
cache may repeatedly evict shards that are needed again soon, which can cause thrashing
and low effective throughput due to the cost of repeatedly downloading shards. While
Zephon detects cache thrashing, we still recommend configuring your cache deliberately.

```{literalinclude} ../../../examples/guide/datasets/storage_configuration.py
:language: python
:caption: examples/guide/datasets/storage_configuration.py
```

:::{warning}
**Direct reads from S3.** Some object storage services like S3 introduced the option to
read byte ranges from files stored in the cloud, rather than having to download the entire
file first. Zephon does not currently support this, and for compressed formats such as
Parquet, we need to read the entire file. In the future, we plan to implement this
optimization for formats such as litData, as it enables truly global shuffling without
downloading all of the shards to the local node.
:::
