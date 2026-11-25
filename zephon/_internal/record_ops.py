# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Record-level builder methods shared by `Pipeline` and `Branch`.

``Pipeline`` appends each op these methods build to its graph. ``Branch``
holds one of them until ``Pipeline.branch()`` wires it after the op before the
branches.
"""

import functools
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, Optional, TypeVar, Union, overload

from zephon._internal.functional_op import _FunctionalOp
from zephon._internal.ops.decode_text import DecodeText
from zephon._internal.ops.expand_chat import ExpandChat
from zephon._internal.ops.map_transform import MapTransform
from zephon._internal.ops.tokenize_chat import TokenizeChat, _resolve_loss_scope
from zephon._internal.ops.tokenize_text import TokenizeText
from zephon.ops.accumulators import Accumulator, PassthroughAccumulator
from zephon.ops.base import BaseOp
from zephon.ops.config import (
    ChatSpecialTokensMode,
    LossScope,
    MissingFieldMode,
    SpanSource,
    SpecialTokensMode,
)
from zephon.ops.traits import OpTraits
from zephon.types import SamplePayload, SampleRecord

# Self type for builder methods shared by Pipeline and Branch
_RecordOpsT = TypeVar("_RecordOpsT", bound="_RecordOps")

# TypeVar for Pipeline methods that mutate the graph or cached plan state
_PipelineMethod = TypeVar("_PipelineMethod", bound=Callable[..., Any])


def _mutates_graph(method: _PipelineMethod) -> _PipelineMethod:
    """Decorator for Pipeline methods that mutate the graph or cached plan state.

    Automatically calls ``_invalidate_plan()`` before the method body so that
    the plan/engine cache is always cleared when the graph changes.  Applying
    this decorator makes the intent explicit and is enforced by the structural
    test ``test_all_graph_mutating_methods_invalidate_plan``.
    """

    @functools.wraps(method)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        self._invalidate_plan()
        return method(self, *args, **kwargs)

    wrapper._mutates_graph = True  # type: ignore[attr-defined]
    return wrapper  # type: ignore[return-value]


class _RecordOps:
    """Record-level builder methods shared by `Pipeline` and `Branch`.

    Each method builds one stateless, record-local op and hands it to
    ``_attach``: `Pipeline` appends it to the graph, while `Branch` holds it
    until ``Pipeline.branch()`` wires it after U.
    """

    def _invalidate_plan(self) -> None:
        raise NotImplementedError

    def _attach(
        self,
        name: str,
        op: BaseOp,
        *,
        placement: str,
        parallelism: Optional[int],
    ) -> None:
        raise NotImplementedError

    @_mutates_graph
    def decode_text(
        self: _RecordOpsT, parallelism: Optional[int] = None, **kwargs: Any
    ) -> _RecordOpsT:
        self._attach(
            "decode_text",
            DecodeText(**kwargs),
            placement="local",
            parallelism=parallelism,
        )
        return self

    @_mutates_graph
    def map_transform(
        self: _RecordOpsT,
        transform_fn: Callable[[SamplePayload], SamplePayload | None],
        *,
        drop_none: bool = True,
        placement: str = "auto",
        parallelism: Optional[int] = None,
    ) -> _RecordOpsT:
        """Add a map-style transformation operator.

        Applies a user-provided transformation function to each sample's payload.
        Supports lambda functions, closures, and nested functions seamlessly with
        multiprocessing-based runners (uses cloudpickle for serialization).

        Args:
            transform_fn: Callable that transforms the payload.
                If it returns None and drop_none=True, the sample is filtered out.
                Supports lambdas, closures, and nested functions.
            drop_none: If True, drop samples where transform_fn returns None.
            placement: Placement hint for this operator.
            parallelism: Override default parallelism for this operator.

        Returns:
            Self for method chaining.

        Example:
            >>> pipeline.map_transform(lambda x: {"value": x["value"] * 2})
        """
        op = MapTransform(transform_fn, drop_none=drop_none)
        self._attach(
            "map_transform",
            op,
            placement=placement,
            parallelism=parallelism,
        )
        return self

    @overload
    def add_op(
        self: _RecordOpsT,
        op: BaseOp,
        /,
        *,
        name: Optional[str] = None,
        placement: str = "auto",
    ) -> _RecordOpsT: ...

    @overload
    def add_op(
        self: _RecordOpsT,
        name: str,
        /,
        *,
        process_many: Callable[[list[Any]], list[Any]],
        preserves_cursor_order: bool,
        calibration_replayable: bool = False,
        accumulator: Optional[Callable[..., "Accumulator[Any]"]] = None,
        process_one: Optional[Callable[[Any], list[Any]]] = None,
        validation_samples: Optional[Callable[[], list[SampleRecord]]] = None,
        parallelism: int = 1,
        placement: str = "auto",
        indexable: bool = False,
        batch_shape_sensitive: bool = False,
        requires_serial_state: bool = False,
    ) -> _RecordOpsT: ...

    @_mutates_graph
    def add_op(
        self: _RecordOpsT,
        name_or_op: Union[str, BaseOp],
        /,
        *,
        name: Optional[str] = None,
        process_many: Optional[Callable[[list[Any]], list[Any]]] = None,
        preserves_cursor_order: Optional[bool] = None,
        calibration_replayable: Optional[bool] = None,
        accumulator: Optional[Callable[..., "Accumulator[Any]"]] = None,
        process_one: Optional[Callable[[Any], list[Any]]] = None,
        validation_samples: Optional[Callable[[], list[SampleRecord]]] = None,
        parallelism: Optional[int] = None,
        placement: str = "auto",
        indexable: Optional[bool] = None,
        batch_shape_sensitive: Optional[bool] = None,
        requires_serial_state: Optional[bool] = None,
    ) -> _RecordOpsT:
        """Append a custom operator — two forms.

        **Instance form**: ``pipe.add_op(op, name=..., placement=...)``.
        Pass a prebuilt :class:`BaseOp` subclass instance.  Use this when
        the op needs full lifecycle control — ``__init__`` for
        configuration, ``setup`` for per-worker resource construction
        (tokenizers, model handles) and :class:`OpContext` services,
        ``traits`` for any combination of `OpTraits` fields.  The
        framework deep-copies the instance per parallel worker, so
        ``self.*`` attributes set in ``setup`` are isolated per worker
        on every runner (including the thread runner).  This matches
        the lifecycle built-in operators use.  Traits come from
        ``op.traits()`` rather than kwargs.

        **Kwargs form**: ``pipe.add_op(name, *, process_many=..., ...)``.
        ``preserves_cursor_order`` is required.
        Convenience for stateless transforms.  The framework builds an
        internal `BaseOp` subclass from the supplied callables +
        accumulator factory + traits.  ``process_many`` must not carry
        state between calls; any per-lane or cross-invocation state
        belongs in the accumulator.  Read-only shared resources (codec
        tables, thresholds) can be captured in a closure over the
        callables.  See the "Accumulators and Operators" page in the
        documentation for the operator/accumulator split.

        Args:
            op: Instance form. The :class:`BaseOp` instance to attach.
            name: Operator name used in plan graphs, metrics, and logs.
                The instance form defaults to the class name of ``op``.
            placement: Placement hint passed to the planner (``"auto"``,
                ``"local"``, or a runner-specific tag).
            process_many: Kwargs form, required. Callable applied to each ready batch
                from the accumulator. Receives a list of upstream items,
                returns the list of downstream items. Runs in parallel
                workers when ``parallelism > 1``.
            accumulator: Kwargs form. Optional factory returning a fresh `Accumulator`
                instance. The framework calls the factory once at runner
                setup and again on ``reset_buffers`` between runs, so a
                factory (rather than an instance) is required for
                correctness. Two signatures are supported and auto-detected:

                - ``Callable[[], Accumulator]`` — simplest form. Use
                  ``lambda: CountingAccumulator(max_batch=N)`` for
                  size-based per-lane batching (it always groups by lane).
                - ``Callable[*, deterministic, ctx], Accumulator]`` —
                  for accumulators whose construction depends on the
                  deterministic mode or runtime context. Example::

                      accumulator=lambda *, deterministic, ctx: (
                          MyAccumulator(timed_flush=not deterministic)
                      )

                In deterministic mode the runner drops ``max_latency_ms``
                from any `CountingAccumulator` either form returns, so
                ``CountingAccumulator(max_batch=N, max_latency_ms=3)`` is
                safe to return unconditionally.

                Defaults to a `PassthroughAccumulator` factory — each
                upstream micro-batch becomes one ready batch as-is.
            process_one: Kwargs form. Optional fast path for single-element
                processing.
                If omitted, the framework wraps each element in a list and
                routes it through ``process_many``.
            validation_samples: Kwargs form. Optional factory returning a list of
                :class:`~zephon.types.SampleRecord` instances for the
                validation harness.  Override when ``process_many``
                requires payload fields beyond the validator's synthetic
                ``{'text': str, 'value': int}`` — letting the harness
                run the full op-level + state-diff suite instead of
                degrading to ``OP_REJECTS_GENERIC_PAYLOAD``.  Buggy
                factories surface ``OP_VALIDATION_SAMPLES_FACTORY_FAILED``
                and the validator falls back to synthetic records.
                Include at least two distinct ``lane_id`` values so the
                cross-call-state probe stays meaningful.
            parallelism: Kwargs form. Number of worker invocations to run in
                parallel for this op.  Default 1 (serial).  Increase when
                ``process_many`` is CPU/GPU-bound.
            preserves_cursor_order: Kwargs form, required. True when ``process_many``
                emits records whose ``chunk_id`` order matches their
                inputs (1:1 maps, payload transforms, non-reordering
                filters). False when the op reorders, shuffles, or
                packs. The planner picks a different eviction strategy
                based on this trait — getting it wrong corrupts
                checkpoint semantics silently. See the Checkpointing
                page in the documentation.
            indexable: Kwargs form. Whether the op preserves indexability through the
                plan.  Default False; set True only if the transform is
                1:1 and deterministic.
            calibration_replayable: Kwargs form. Whether token calibration may replay
                the op before tokenization. Default False; see :class:`OpTraits`.
            batch_shape_sensitive: Kwargs form. Set True when ``process_many``'s output
                can depend on how inputs are grouped into micro-batches
                (per-batch RNG, statistics, etc.). Descriptive only: batch
                boundaries come from the accumulator, and in deterministic
                mode the runner already drops latency-based flushing from
                `CountingAccumulator`. A custom `Accumulator` that flushes
                on time must disable that itself when ``deterministic`` is
                True.
            requires_serial_state: Kwargs form. Set True when the op's accumulator
                holds cross-invocation state that cannot be sharded
                across parallel worker instances. In deterministic mode
                the planner pins ``parallelism=1`` for ops with this
                trait.

        Returns:
            Self for method chaining.

        Examples::

            # Instance form — class with setup/traits/accumulator overrides.
            class Tokenize(BaseOp):
                def __init__(self, name):
                    super().__init__()
                    self._name = name
                    self._tokenizer = None
                def traits(self):
                    return OpTraits(
                        preserves_cursor_order=True,
                        calibration_replayable=True,
                        parallelism=4,
                    )
                def setup(self, ctx):
                    super().setup(ctx)
                    self._tokenizer = load_tokenizer(self._name)
                def process_many(self, elems):
                    for e in elems:
                        e.payload = {"ids": self._tokenizer.encode(e.payload["text"])}
                    return elems

            pipeline.add_op(Tokenize("gpt2"))

            # Kwargs form — stateless windowed transform.
            from zephon.ops import CountingAccumulator
            pipeline.add_op(
                "windowed_transform",
                process_many=lambda elems: [...],
                accumulator=lambda: CountingAccumulator(max_batch=64),
                parallelism=4,
                preserves_cursor_order=True,
            )
        """
        if isinstance(name_or_op, BaseOp):
            mismatched = [
                kw
                for kw, val in (
                    ("process_many", process_many),
                    ("preserves_cursor_order", preserves_cursor_order),
                    ("calibration_replayable", calibration_replayable),
                    ("accumulator", accumulator),
                    ("process_one", process_one),
                    ("validation_samples", validation_samples),
                    ("parallelism", parallelism),
                    ("indexable", indexable),
                    ("batch_shape_sensitive", batch_shape_sensitive),
                    ("requires_serial_state", requires_serial_state),
                )
                if val is not None
            ]
            if mismatched:
                raise ValueError(
                    "add_op(op) with a BaseOp instance does not accept "
                    f"{', '.join(mismatched)}; traits, callables, and "
                    "validation_samples come from the op itself "
                    "(override `traits()` / `validation_samples()` on "
                    "your subclass).  Use the kwargs form add_op(name, "
                    "...) instead if you want to supply callables."
                )
            instance_op = name_or_op
            node_name = name if name is not None else type(instance_op).__name__
            if not node_name:
                raise ValueError("add_op() requires a non-empty name")
            self._attach(node_name, instance_op, placement=placement, parallelism=None)
            return self

        op_name = name_or_op
        if name is not None:
            raise ValueError(
                "add_op(name, ...) takes the op name positionally; the "
                f"`name=` keyword is only for the instance form. Got positional "
                f"{op_name!r} and name={name!r} — drop the `name=` kwarg."
            )
        if not op_name:
            raise ValueError("add_op() requires a non-empty name")
        if process_many is None:
            raise ValueError("add_op(name, ...) requires `process_many`")
        if preserves_cursor_order is None:
            raise ValueError(
                "add_op(name, ...) requires `preserves_cursor_order` "
                "(True for 1:1 maps and non-reordering filters; False for "
                "reorder/shuffle/pack ops)"
            )

        functional_op = _FunctionalOp(
            process_many_fn=process_many,
            accumulator_factory=(
                accumulator if accumulator is not None else PassthroughAccumulator
            ),
            process_one_fn=process_one,
            validation_samples_factory=validation_samples,
            op_traits=OpTraits(
                preserves_cursor_order=preserves_cursor_order,
                calibration_replayable=(
                    False if calibration_replayable is None else calibration_replayable
                ),
                parallelism=1 if parallelism is None else parallelism,
                indexable=False if indexable is None else indexable,
                batch_shape_sensitive=(
                    False if batch_shape_sensitive is None else batch_shape_sensitive
                ),
                requires_serial_state=(
                    False if requires_serial_state is None else requires_serial_state
                ),
            ),
        )
        self._attach(
            op_name,
            functional_op,
            placement=placement,
            parallelism=parallelism,
        )
        return self

    @_mutates_graph
    def tokenize(
        self: _RecordOpsT,
        tokenizer: Any | None = None,
        tokenizer_id: str | None = None,
        *,
        eos_token: str | None = None,
        field: str,
        missing_field: MissingFieldMode = "error",
        add_attention_mask: bool = True,
        max_length: int | None = None,
        padding: bool | str = False,
        truncation: bool = False,
        return_tensors: str | None = None,
        split_long_samples: bool = False,
        use_fast: bool | None = True,
        preserve_upstream_payload: bool = False,
        special_tokens: SpecialTokensMode = "bos_eos",
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        placement: str = "auto",
        parallelism: Optional[int] = None,
    ) -> _RecordOpsT:
        """Tokenize a text field.

        Array outputs preserve the dtype supplied by the tokenizer, which may be
        unsigned; Python lists use the requested framework's dtype inference.
        Dtype conversion for training should normally happen in ``to_training()``
        or a custom map op.
        """
        op = TokenizeText(
            tokenizer,
            tokenizer_id,
            eos_token=eos_token,
            field=field,
            missing_field=missing_field,
            add_attention_mask=add_attention_mask,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
            split_long_samples=split_long_samples,
            use_fast=use_fast,
            preserve_upstream_payload=preserve_upstream_payload,
            special_tokens=special_tokens,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
        )
        self._attach("tokenize", op, placement=placement, parallelism=parallelism)
        return self

    @_mutates_graph
    def expand_chat(
        self: _RecordOpsT,
        *,
        field: str = "messages",
        placement: str = "auto",
        parallelism: int | None = None,
    ) -> _RecordOpsT:
        """Emit one conversation prefix per user query containing an assistant.

        Each prefix ends before the next user, or at the end of the input.
        Earlier history and other payload fields are retained. Empty assistant
        content is allowed. Queries without an assistant emit nothing.
        Prefixes own independent nested payloads; the sole or final prefix may
        reuse input objects.

        Expansion repeats context and does not select loss targets. Normalize
        roles before expansion; do not expand data already stored as prefixes.
        Packed prefixes require document-isolated attention in the model.
        Keep expansion and tokenization in one stage to avoid serializing repeated
        history between processes. Changing expansion settings before checkpoint
        resume is unsupported and is not detected.

        Args:
            field: Dot-separated payload path of the messages list.
            placement: Execution placement, usually shared with tokenization.
            parallelism: Worker count override.
        """
        self._attach(
            "expand_chat",
            ExpandChat(field=field),
            placement=placement,
            parallelism=parallelism,
        )
        return self

    @_mutates_graph
    def tokenize_chat(
        self: _RecordOpsT,
        tokenizer: Any | None = None,
        tokenizer_id: str | None = None,
        *,
        eos_token: str | None = None,
        special_tokens: ChatSpecialTokensMode | None = None,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        field: str = "messages",
        max_length: int | None = None,
        chat_template: str | os.PathLike[str] | None = None,
        apply_chat_template: bool = True,
        span_source: SpanSource = "auto",
        supervise_assistant_header: bool = False,
        loss_scope: LossScope | None = None,
        loss_on_last_turn_only: bool | None = None,
        chat_template_kwargs: Mapping[str, Any] | None = None,
        tools_field: str = "tools",
        enable_thinking_field: str = "enable_thinking",
        mask_field_out: str = "loss_mask",
        preserve_upstream_payload: bool = False,
        placement: str = "auto",
        parallelism: Optional[int] = None,
    ) -> _RecordOpsT:
        """Tokenize each conversation into token IDs and a loss mask.

        Reads role/content messages from ``field`` (default: ``messages``).
        Provide a fast Hugging Face ``tokenizer`` or ``tokenizer_id``.
        Uses the tokenizer's own chat template unless ``chat_template`` supplies
        a template name, Jinja text or a file path. Without an override, a
        non-None value at ``tools_field`` selects ``tool_use`` when that template
        exists, otherwise ``default``. ``chat_template_kwargs`` supplies static arguments;
        ``tools_field`` and ``enable_thinking_field`` select per-record values.

        Output replaces the payload with aligned, unshifted NumPy arrays:
        ``input_ids`` (int64) and ``loss_mask`` (uint8; rename via ``mask_field_out``).
        Ones mark target tokens. ``preserve_upstream_payload=True`` retains other
        fields; ``max_length`` truncates the output.
        Samples with fewer than two tokens or no supervised tokens are dropped.

        Targets come from ``{% generation %}`` blocks. Untagged templates infer
        assistant ranges by rendering successive conversation prefixes. Set
        ``supervise_assistant_header=True`` to include assistant headers in those ranges.
        ``loss_scope`` selects all assistant spans (``all_assistant``, default),
        the final contiguous supervised run (``last_span``), all assistant messages
        after the final user (``after_last_user``), or every token except the first
        (``all_tokens``). Untagged templates require explicit
        ``span_source="prefix_diff"`` with ``after_last_user``.

        ``special_tokens="bos"``, ``"eos"`` or ``"bos_eos"`` adds the requested
        tokens around the complete conversation, reserving room under ``max_length``.
        ``"none"`` adds no tokens beyond those in the input or template.
        ``bos_token_id`` and ``eos_token_id`` choose IDs for explicit outer insertion.
        ``eos_token`` overrides the EOS string when loading ``tokenizer_id``.

        ``apply_chat_template=False`` concatenates message contents without role
        headers. In this mode, ``special_tokens="tokenizer_default"`` uses the
        tokenizer's configured special-token processing, and ``"assistant_eos"``
        appends EOS after each assistant message. With ``special_tokens`` omitted,
        plain conversations use per-assistant EOS; templated conversations get a
        final EOS if missing and space remains.

        With generation tags, ``after_last_user`` verifies templates automatically.
        Each selected assistant's generation blocks must cover its reasoning,
        calls, answer and end-of-turn token within its message loop, excluding
        user/tool-result content. Verification adds rendering and tokenization at
        setup and for new message/tool formats or non-object tool argument values;
        successful checks are reused. ``preflight_tokenizers()`` runs setup checks
        before iteration. Samples with omitted selected content are dropped and
        logged; other verification failures raise. Inspect outputs and masks:
        checks cannot cover every content-dependent template branch.

        ``loss_on_last_turn_only`` is deprecated: True selects ``last_span`` and
        False selects ``all_assistant``. Cannot accompany ``loss_scope``.
        """
        # warn -> resolver -> tokenize_chat -> @_mutates_graph -> caller
        loss_scope = _resolve_loss_scope(
            loss_scope, loss_on_last_turn_only, stacklevel=4
        )
        op = TokenizeChat(
            tokenizer,
            tokenizer_id,
            eos_token=eos_token,
            special_tokens=special_tokens,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            field=field,
            max_length=max_length,
            chat_template=chat_template,
            apply_chat_template=apply_chat_template,
            span_source=span_source,
            supervise_assistant_header=supervise_assistant_header,
            loss_scope=loss_scope,
            chat_template_kwargs=chat_template_kwargs,
            tools_field=tools_field,
            enable_thinking_field=enable_thinking_field,
            mask_field_out=mask_field_out,
            preserve_upstream_payload=preserve_upstream_payload,
        )
        self._attach(
            "tokenize_chat",
            op,
            placement=placement,
            parallelism=parallelism,
        )
        return self


@dataclass(frozen=True)
class _BranchOp:
    """The op a `Branch` holds until ``Pipeline.branch()`` adds it."""

    name: str
    op: BaseOp
    placement: str
    parallelism: Optional[int]


class Branch(_RecordOps):
    """Builds one branch of ``Pipeline.branch()``.

    A branch function receives an empty ``Branch`` and returns it. Call one
    record-level method on it to add that branch's op, or none to pass records
    through. A branch holds at most one op for now; steps that follow go on
    the pipeline after the ``branch()`` call.
    """

    def __init__(self, name: str) -> None:
        """Start an empty branch called ``name``."""
        self._name = name
        self._held: _BranchOp | None = None

    def __repr__(self) -> str:
        """Describe the branch and the op it holds."""
        held = self._held.name if self._held is not None else "passthrough"
        return f"Branch({self._name!r}, {held})"

    def _invalidate_plan(self) -> None:
        """Do nothing: a branch caches no plan; ``Pipeline.branch()`` does."""

    def _attach(
        self,
        name: str,
        op: BaseOp,
        *,
        placement: str,
        parallelism: Optional[int],
    ) -> None:
        if self._held is not None:
            raise ValueError(
                f"branch {self._name!r} already holds {self._held.name!r}; a "
                "branch holds at most one op for now, so add later steps after "
                "the branch() call"
            )
        self._held = _BranchOp(name, op, placement, parallelism)
