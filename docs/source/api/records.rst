Records and batches
===================

What travels between operators, and what a batch hands the training loop. The
guide covers these in :doc:`../pipelines/building_training_batches`.

.. automodule:: zephon.types
   :members:
   :show-inheritance:

Type aliases
------------

These are type aliases, not classes; they name the shapes that appear in
operator and payload signatures.

.. py:data:: SampleId
   :type: tuple[DatasetId, ShardId, LocalSampleId]

   Where a sample sits in the data: which dataset, which shard, which row.

.. py:data:: LineagePath
   :type: tuple[LineageIndex, ...]

   One index per fan-out step, recording how a record descends from a sample.

.. py:data:: SampleCursorKey
   :type: tuple[ChunkId, ChunkOffset, LineagePath, SampleId]

   Cursor order: chunk, then offset, then lineage, with the sample id last so it
   only breaks ties.

.. py:data:: SampleNumeric
   :type: int | float | complex | _NumpyNumber

   The numeric types a payload may hold, including NumPy scalars.
   ``_NumpyNumber`` denotes ``numpy.number[Any]``.

.. py:data:: SamplePayloadAtom
   :type: bytes | memoryview | str | SampleNumeric | _NumpyTemporal | SamplePayloadArray | None

   A payload leaf: anything that is not a list or a dict of further payloads.
   ``_NumpyTemporal`` denotes ``numpy.datetime64 | numpy.timedelta64``;
   ``None`` represents a null value.

.. py:data:: SamplePayload
   :type: SamplePayloadAtom | list[SamplePayload] | dict[Any, SamplePayload]

   A payload of any shape: a leaf, or lists and dicts nesting further payloads.

.. py:data:: SamplePayloadDict
   :type: dict[Any, SamplePayload]

   The mapping form of a payload, which is what most operators receive.

.. py:data:: StreamItem
   :type: SampleRecord | SampleBatch

   What travels between operators: records before ``batch``, batches after it.
