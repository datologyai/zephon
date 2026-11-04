Runtime and tuning
==================

How a pipeline executes, what it reports while running, and what it checks
before it starts. The guide covers these in
:doc:`../pipelines/inspecting_and_tuning_pipelines` and
:doc:`../pipelines/distributed_training`.

Runtime options
---------------

.. automodule:: zephon.options
   :members:
   :show-inheritance:

.. py:data:: IpcTransport
   :type: Literal['socketpair', 'pipe']

   Transport for the process runner's and MTP's IPC queues.

Observability
-------------

.. automodule:: zephon.observability
   :members:
   :show-inheritance:

Validation
----------

.. automodule:: zephon.validation
   :members:
   :show-inheritance:

Debugging
---------

Diagnostics for a run that leaks or hangs on shutdown.

.. autofunction:: zephon.debug.install_debug_hooks
.. autofunction:: zephon.debug.dump_semaphore_registry
.. autofunction:: zephon.debug.dump_semaphore_leak_report
