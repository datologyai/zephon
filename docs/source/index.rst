Zephon
======

Zephon is an elastically deterministic data loader for ML training. It keeps your data in
the same order even when you resume a run on a different number of GPUs, which turns out
to be both surprisingly useful and tricky to get right.

The best place to start is :doc:`basic_concepts`, which introduces the ``Dataset``,
``WorkSource``, and ``Pipeline`` abstractions that the rest of this guide walks through
in turn. If you want to know even more about why we built Zephon (and why you might care
about the order your data arrives in), :doc:`why_zephon` has you covered. The
Internals section opens Zephon up and describes the machinery behind the guarantees the
guide makes, and the API reference documents the public classes and functions that the
guide refers to.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   basic_concepts
   why_zephon
   datasets/index
   worksources/index
   pipelines/index
   training_integrations

.. toctree::
   :maxdepth: 2
   :caption: Internals

   internals/index

.. toctree::
   :maxdepth: 1
   :caption: API reference

   api/datasets
   api/worksources
   api/pipeline
   api/operators
   api/records
   api/runtime
   api/tools
