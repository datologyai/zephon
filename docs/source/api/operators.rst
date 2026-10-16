Operators
=========

The authoring contract for operators you write yourself, and the grouping and
lineage helpers they use. The guide covers these in
:doc:`../pipelines/user_defined_operators`.

.. automodule:: zephon.ops
   :members:
   :show-inheritance:

Argument types
--------------

The string literals that operator-configuration arguments accept.

.. py:data:: PackingAlgorithm
   :type: Literal['first_fit', 'best_fit', 'wrap', 'best_fit_wrap']

   Which algorithm a packing operator fills its sequences with.

.. py:data:: SpecialTokensMode
   :type: Literal['bos_eos', 'bos', 'eos', 'none', 'tokenizer_default']

   Which special tokens ``tokenize`` adds around each sequence.

.. py:data:: MissingFieldMode
   :type: Literal['error', 'empty']

   What ``tokenize`` does when a record lacks the field it was given.

.. py:data:: SpanSource
   :type: Literal['auto', 'generation_tags', 'prefix_diff']

   How ``tokenize_chat`` works out which tokens belong to the assistant.

Accumulator types
-----------------

.. py:data:: ReadyBatch
   :type: tuple[list[T], int]

   What an accumulator emits for one worker invocation: the grouped elements,
   and the nanoseconds they waited, which is carried for metrics.
