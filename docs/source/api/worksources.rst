WorkSources
===========

Deciding what to train on: datasets, mixture weights, shuffling and repetition.
The guide covers these in :doc:`../worksources/index`.

.. automodule:: zephon.work
   :members:
   :show-inheritance:

Shuffle block size
------------------

.. py:currentmodule:: zephon.work.static_mixture

.. py:data:: ShuffleBlockSpec
   :type: int | Literal['auto', 'global'] | None

   Accepted values for ``StaticMixtureWorkSource(shuffle_block_size=...)``:
   ``None`` disables cross-shard shuffling, a positive ``int`` sets the block
   size in samples, ``"auto"`` uses eight times the largest shard, and
   ``"global"`` shuffles across the whole mixture. Sizes are clamped to the
   total sample count.

Work chunk types
----------------

The aliases :class:`~zephon.work.WorkChunk` uses for its components and the
sample ids it yields.

.. py:currentmodule:: zephon.work.base

.. py:data:: MixtureComponent
   :type: str

   Name of one component of a dataset mixture.

.. py:data:: SourcedSampleId
   :type: tuple[SampleId, MixtureComponent]

   A sample id paired with the mixture component it came from.

.. py:data:: SamplesPerComponent
   :type: MutableMapping[MixtureComponent, list[SampleId]]

   Sample ids grouped by mixture component, as in ``WorkChunk.components``.
