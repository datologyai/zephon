WorkSources
===========

Deciding what to train on: datasets, mixture weights, shuffling and repetition.
The guide covers these in :doc:`../worksources/index`.

.. automodule:: zephon.work
   :members:
   :show-inheritance:

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
