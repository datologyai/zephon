Tools
=====

Offline helpers that run beside a training job rather than inside one. The guide
builds an index in :doc:`../datasets/parquet` and
:doc:`../datasets/binary_shard_formats/vortex`.

.. automodule:: zephon.build_index
   :members:
   :show-inheritance:

Chat template verification
--------------------------

Message-based loss scopes run verification automatically.
:func:`zephon.validation.verify_chat_template` exposes additional fixtures and
stock-template comparisons for template development.
