Zephon Documentation
====================

Zephon is a high-performance, modular, multimodal-first data loader for ML applications.

Key features:

- **Fluent builder-pattern API** for composing data pipelines
- **Deterministic checkpoint/restart** for fault-tolerant training
- **Cloud storage support** (S3, GCS, local filesystem)
- **Elastic deterministic training** (change rank count while preserving order)
- **Multiple data formats** (JSONL, LitData, MosaicML Streaming/MDS, Vortex)

Quick Example
-------------

.. code-block:: python

   from zephon import Pipeline
   from zephon.io import Dataset
   from zephon.work import MixtureSpec, StaticMixtureWorkSource

   # Discover a directory of JSONL shards
   ds = Dataset.from_path("train", "/data/train/")

   # Build a work source
   ws = StaticMixtureWorkSource(
       [ds],
       mixture=MixtureSpec({"train": 1.0}),
       chunk_size=16384,
       seed=42,
   )

   # Create and run a pipeline
   pipeline = (
       Pipeline(ws)
       .decode_text()
       .tokenize(
           tokenizer_id="gpt2",
           field="text",
           max_length=2048,
           truncation=True,
           padding="max_length",
       )
       .batch(microbatch_size=8)
   )

   for batch in pipeline:
       train_step(batch.to_training())


.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   🚀 Quick Start <quickstart>
   📖 Basic Concepts <basic_concepts>
   🧩 Transitioning from Streaming <transitioning>

.. toctree::
   :maxdepth: 2
   :caption: Understanding Zephon

   understanding/worksources
   understanding/sample_lifecycle
   understanding/checkpointing
   understanding/distributed_training
   understanding/determinism
   understanding/accumulators_operators

.. toctree::
   :maxdepth: 2
   :caption: Developer Resources

   guides/dev_guide
   guides/prefetch_op
   guides/sample_lifecycle

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/zephon_pipeline
   api/zephon_types
   api/zephon_io
   api/zephon_work
   api/zephon_ops
   api/zephon_observability
   api/zephon_build_index


Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
