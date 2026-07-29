# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Tokenizer operators that prepare text fields for model consumption."""

from __future__ import annotations

import logging
import typing
from types import ModuleType
from typing import (
    TYPE_CHECKING,
    Any,
    Mapping,
    Optional,
    Sequence,
    TypeAlias,
    TypeVar,
    Union,
    cast,
)

from zephon._internal.op_base import DefaultSetup
from zephon._internal.ops.tokenize_base import _MISSING, TokenizeBase
from zephon._internal.token_counting import TextTokenCountingSpec
from zephon._internal.utils.tokenizer import (
    TokenBatch,
    TokenizerLike,
    TokenizerOutput,
)
from zephon._internal.utils.torch_compat import (
    _TENSOR_ITER_LOCK,
    _should_use_tensor_lock,
    _tensor_lock_ctx,
)
from zephon.ops.base import OpContext
from zephon.ops.children import spawn_child
from zephon.ops.config import SpecialTokensMode
from zephon.ops.traits import OpTraits
from zephon.types import SampleMeta, SamplePayload, SamplePayloadDict, SampleRecord

if TYPE_CHECKING:
    import numpy as np
    import tensorflow as tf
    import torch

log = logging.getLogger(__name__)

_VALID_SPECIAL_TOKENS_MODES: tuple[str, ...] = typing.get_args(SpecialTokensMode)

TokenSeq: TypeAlias = Union[
    "np.ndarray", "torch.Tensor", "tf.Tensor", Sequence[int], list[int]
]
# Bound TypeVar ensures "Tensor in -> Tensor out" relationship
T_TokenSeq = TypeVar("T_TokenSeq", bound=TokenSeq)


class TokenizeText(TokenizeBase):
    """Tokenize text fields using a provided or auto-resolved tokenizer."""

    def __init__(
        self,
        tokenizer: TokenizerLike | None = None,
        tokenizer_id: str | None = None,
        *,
        field: str,
        add_attention_mask: bool = True,
        max_length: int | None = None,
        padding: bool | str = False,
        truncation: bool = False,
        return_tensors: str | None = None,
        split_long_samples: bool = False,
        use_fast: bool | None = True,
        max_batch: int = 64,
        max_latency_ms: Optional[int] = 3,
        preserve_upstream_payload: bool = False,
        special_tokens: SpecialTokensMode = "bos_eos",
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
    ) -> None:
        """Tokenize text fields using a provided or auto-resolved tokenizer.

        Args:
            tokenizer: Pre-instantiated HF-compatible tokenizer; takes precedence over
                ``tokenizer_id``.
            tokenizer_id: HF model id passed to ``AutoTokenizer.from_pretrained``.
                ``"__fallback__"`` selects a small in-process stub for tests.
            field: Dot-separated payload path of the text to tokenize.
            add_attention_mask: Emit ``attention_mask`` alongside ``input_ids``.
            max_length: Per-record cap on output length, in tokens *including* any
                BOS/EOS added by ``special_tokens``. Only enforced as a hard
                cap when ``truncation=True`` or ``split_long_samples=True``;
                on its own ``max_length`` is just a target for
                ``padding="max_length"`` and a no-op otherwise — rows already
                longer than ``max_length`` pass through unchanged (matches
                HF). When ``truncation=True`` and the operator is adding
                BOS/EOS, the HF call receives ``max_length - num_specials``
                so the bracketed output lands at exactly ``max_length``.
                Required when ``truncation=True`` in any bracket mode (the
                implicit HF fallback to ``tokenizer.model_max_length`` cannot
                account for BOS/EOS).
            padding: When ``special_tokens="tokenizer_default"`` this is passed
                straight to the HF tokenizer. In every other mode HF is called
                with ``padding=False`` (so BOS/EOS can be placed adjacent to
                real content) and the operator pads the resulting sequences
                itself: ``True`` / ``"longest"`` pads to the longest sequence
                in the batch, ``"max_length"`` pads to ``max_length`` (which
                must then be set). ``special_tokens="none"`` uses the same
                operator-owned padding path even though no specials are added.
                Note: ``padding="max_length"`` does not imply truncation. Rows
                whose bracketed length exceeds ``max_length`` pass through
                unchanged; only shorter rows are padded up. Matches HF.
            truncation: Truncate to ``max_length`` (which must be set explicitly
                in bracket modes). When the operator is adding BOS/EOS itself,
                the HF call receives ``max_length - num_specials`` so the final
                bracketed output comes out at exactly ``max_length`` tokens.
            return_tensors: ``"pt"`` / ``"np"`` / ``"tf"`` backend for emitted fields.
            split_long_samples: Slice each tokenized sequence into ``max_length``
                chunks. BOS/EOS, if added, land only on the first/last chunk.
            use_fast: Forwarded to ``AutoTokenizer.from_pretrained``.
            preserve_upstream_payload: Keep all upstream payload keys; otherwise the
                output dict contains only ``input_ids`` and ``attention_mask``.
            special_tokens: How BOS/EOS are added.

                * ``"bos_eos"`` (default): the operator brackets each input
                  document as ``[BOS, ...tokens, EOS]`` explicitly. The HF
                  tokenizer is called with ``add_special_tokens=False`` so the
                  template never contributes; BOS/EOS come from the tokenizer's
                  ``bos_token_id`` / ``eos_token_id`` attributes (or the
                  overrides below). Suitable for autoregressive pretraining
                  where every document should be explicitly delimited and the
                  output should not depend on the tokenizer family's template.
                * ``"bos"`` / ``"eos"``: as above but with only one side added.
                * ``"none"``: emits raw content tokens. Useful when an upstream
                  step already added the desired specials.
                * ``"tokenizer_default"``: the HF tokenizer is called with
                  ``add_special_tokens=True`` and its ``build_inputs_with_
                  special_tokens`` template decides what to add. The result is
                  tokenizer-family-dependent (Llama: BOS only; T5: EOS only;
                  GPT-2: nothing; chat templates: varies). Choose this when
                  you specifically want the template's behavior — e.g. when
                  feeding chat-formatted prompts into an instruct tokenizer.
            bos_token_id: BOS id to splice in under the bracket modes that
                add a BOS (``bos_eos`` / ``bos``), replacing whatever the
                tokenizer exposes via its own ``bos_token_id`` attribute.
                Required when the chosen mode needs a BOS and the tokenizer
                has none. The tokenizer itself is never mutated — this only
                affects what the operator splices into bracketed sequences.
                Passing this under ``tokenizer_default`` (where the
                tokenizer's template owns special-token placement) or
                ``none`` (where no specials are spliced) is a hard error at
                ``__init__`` — silently dropping the override would risk
                training under the wrong BOS assumption. To change which BOS
                HF's template emits under ``tokenizer_default``, mutate the
                tokenizer's ``bos_token`` before passing it in.
            eos_token_id: EOS id to splice in under the bracket modes that
                add an EOS (``bos_eos`` / ``eos``), replacing whatever the
                tokenizer exposes via its own ``eos_token_id`` attribute.
                Required when the chosen mode needs an EOS and the tokenizer
                has none. Tokenizer not mutated; raises at ``__init__``
                under ``tokenizer_default`` / ``none`` — see ``bos_token_id``.
        """
        if special_tokens not in _VALID_SPECIAL_TOKENS_MODES:
            raise ValueError(
                f"special_tokens must be one of {_VALID_SPECIAL_TOKENS_MODES}, "
                + f"got {special_tokens!r}"
            )

        # HF accepts ``"do_not_pad"`` as a no-pad string; collapse to ``False``
        # so downstream ``if self.padding`` checks don't have to handle both.
        if padding == "do_not_pad":
            padding = False

        self.special_tokens: SpecialTokensMode = special_tokens
        self._zephon_owns_brackets: bool = special_tokens != "tokenizer_default"

        if split_long_samples and truncation:
            raise ValueError("split_long_samples is mutually exclusive with truncation")
        if split_long_samples and not max_length:
            raise ValueError("split_long_samples requires max_length")

        if padding == "max_length" and max_length is None:
            raise ValueError("padding='max_length' requires max_length")

        if self._zephon_owns_brackets and truncation and max_length is None:
            raise ValueError(
                "truncation=True requires an explicit max_length when "
                + f"special_tokens={special_tokens!r}; the operator needs it "
                + "to reserve room for BOS/EOS. Set max_length, or use "
                + "special_tokens='tokenizer_default' to let HF's template "
                + "manage specials within its own truncation."
            )

        num_specials = self._num_specials_total()
        if (
            self._zephon_owns_brackets
            and truncation
            and max_length is not None
            and num_specials > 0
            and max_length - num_specials <= 0
        ):
            raise ValueError(
                f"max_length={max_length} cannot fit {num_specials} special "
                + f"token(s) under special_tokens={special_tokens!r}"
            )

        if special_tokens in ("tokenizer_default", "none") and (
            bos_token_id is not None or eos_token_id is not None
        ):
            raise ValueError(
                "bos_token_id/eos_token_id has no effect under "
                + f"special_tokens={special_tokens!r}; overrides apply only "
                + "to bracket modes ('bos_eos' / 'bos' / 'eos'). To splice a "
                + "specific BOS/EOS into the output, switch to a bracket "
                + "mode; to change which BOS/EOS HF's template emits under "
                + "'tokenizer_default', mutate the tokenizer's "
                + "``bos_token`` / ``eos_token`` before passing it in."
            )

        TokenizeBase.__init__(
            self,
            tokenizer,
            tokenizer_id,
            use_fast=use_fast,
            max_batch=max_batch,
            max_latency_ms=max_latency_ms,
        )
        self.field = field
        self._field_path: tuple[str, ...] = tuple(field.split(".")) if field else ()
        self.add_attention_mask = add_attention_mask
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.return_tensors = return_tensors
        self.split_long_samples = split_long_samples
        self.preserve_upstream_payload = preserve_upstream_payload
        self._bos_id_override = bos_token_id
        self._eos_id_override = eos_token_id
        self._bos_id_resolved: int | None = None
        self._eos_id_resolved: int | None = None
        # Cached after _setup_tokenizer so the padding hot path doesn't
        # getattr() the tokenizer per batch.
        self._pad_token_id_cached: int = 0
        self._warned_non_mapping = False
        self._warned_preserve_non_mapping = False
        # Cache kwargs to avoid building dict per batch
        self._cached_kwargs: dict[str, Any] = {}
        # On free-threaded Python + old PyTorch, we intercept return_tensors='pt'
        # to avoid HF tokenizer creating tensors (which races with our code).
        self._convert_np_to_pt = False

    def setup(
        self,
        ctx: OpContext,
        stage_index: int,
        stage_name: str,
        op_index: int,
        collect_stats: bool,
    ) -> None:
        DefaultSetup.setup(self, ctx, stage_index, stage_name, op_index, collect_stats)
        # We do NOT set up the tokenizer here to avoid problems in multiprocessing:
        # hf tokenizers don't like if we fork after creating the object

        # In bracket mode we ask HF for ragged lists (no specials, no padding,
        # no return_tensors) and own bracketing/padding ourselves. Forwarding
        # ``return_tensors`` here would force HF to stack a ragged batch and
        # raise.
        add_specials = not self._zephon_owns_brackets

        # ``split_long_samples`` also owns padding (per-chunk, see
        # ``_split_long_samples``). Padding the full sequence on the HF side
        # before we slice it would inflate every chunk to the original
        # padded length — force HF padding off whenever splitting, regardless
        # of bracket-vs-default mode.
        hf_padding: bool | str = (
            self.padding if (add_specials and not self.split_long_samples) else False
        )
        hf_truncation = False if self.split_long_samples else self.truncation
        hf_max_length: int | None = None if self.split_long_samples else self.max_length

        if (
            not add_specials
            and self.truncation
            and self.max_length is not None
            and not self.split_long_samples
        ):
            # Reserve room for the BOS/EOS we splice in after HF returns.
            hf_max_length = self.max_length - self._num_specials_total()

        self._cached_kwargs = {
            "add_special_tokens": add_specials,
            "padding": hf_padding,
            "truncation": hf_truncation,
        }
        if hf_max_length is not None:
            self._cached_kwargs["max_length"] = hf_max_length

        # On free-threaded Python + old PyTorch, HF tokenizer's tensor creation
        # races with our code. Request numpy from HF and convert ourselves
        # under lock.
        # See: https://github.com/pytorch/pytorch/issues/171992
        forward_return_tensors = add_specials and not self.split_long_samples
        if forward_return_tensors and self.return_tensors is not None:
            if self.return_tensors == "pt" and _should_use_tensor_lock():
                self._convert_np_to_pt = True
                self._cached_kwargs["return_tensors"] = "np"
            else:
                self._cached_kwargs["return_tensors"] = self.return_tensors

    def _finalize_setup(self) -> None:
        # _ensure_padding_token may set pad_token=eos_token (the HF pattern
        # for Llama/GPT-2 et al.), so it must run before we snapshot pad id.
        self._resolve_special_token_ids()
        self._ensure_padding_token()
        self._pad_token_id_cached = self._compute_pad_token_id()

    def _resolve_special_token_ids(self) -> None:
        """Bind the BOS/EOS ids we will splice in. No-op in tokenizer_default mode.

        Raises on first batch (when _setup_tokenizer runs) if the configured
        mode needs a token the tokenizer cannot provide and no override was
        passed — better to fail loudly than silently emit unbracketed streams.
        """
        if not self._zephon_owns_brackets:
            return
        if self.tok is None:
            return

        if self._num_specials_to_prepend() > 0:
            bid = self._bos_id_override
            if bid is None:
                bid = getattr(self.tok, "bos_token_id", None)
            if bid is None:
                raise ValueError(
                    f"special_tokens={self.special_tokens!r} requires a BOS token, "
                    + "but the tokenizer has no bos_token_id and no bos_token_id "
                    + "override was provided. Pass bos_token_id=... or switch "
                    + "special_tokens to 'eos'/'none'/'tokenizer_default'."
                )
            self._bos_id_resolved = int(bid)

        if self._num_specials_to_append() > 0:
            eid = self._eos_id_override
            if eid is None:
                eid = getattr(self.tok, "eos_token_id", None)
            if eid is None:
                raise ValueError(
                    f"special_tokens={self.special_tokens!r} requires an EOS token, "
                    + "but the tokenizer has no eos_token_id and no eos_token_id "
                    + "override was provided. Pass eos_token_id=... or switch "
                    + "special_tokens to 'bos'/'none'/'tokenizer_default'."
                )
            self._eos_id_resolved = int(eid)

    def _num_specials_to_prepend(self) -> int:
        return 1 if self.special_tokens in ("bos_eos", "bos") else 0

    def _num_specials_to_append(self) -> int:
        return 1 if self.special_tokens in ("bos_eos", "eos") else 0

    def _num_specials_total(self) -> int:
        return self._num_specials_to_prepend() + self._num_specials_to_append()

    def traits(self) -> OpTraits:
        return OpTraits(indexable=True, preserves_cursor_order=True, parallelism=4)

    def token_counting_spec(self) -> TextTokenCountingSpec:
        return TextTokenCountingSpec.from_op(self)

    def _lookup_field(self, payload: Mapping[str, Any]) -> Any:
        """Resolve ``self._field_path`` against a (possibly nested) mapping payload."""
        value = self._lookup_path(payload, self._field_path)
        return "" if value is _MISSING else value

    def _extract_text(self, payload: SamplePayload) -> tuple[str, SamplePayloadDict]:
        if isinstance(payload, dict):
            # Safe cast: we expect the user to provide string fields as configured
            text_value = cast(str, self._lookup_field(payload))
            if self.preserve_upstream_payload:
                return text_value, payload
            # Fresh dict: drop upstream keys entirely
            return text_value, {}
        if self.preserve_upstream_payload and not self._warned_preserve_non_mapping:
            log.warning(
                "preserve_upstream_payload=True cannot be honored for non-mapping payloads "
                + "(got %s); tokenizing the value directly",
                type(payload).__name__,
            )
            self._warned_preserve_non_mapping = True
        elif self.field and not self._warned_non_mapping:
            log.warning(
                "TokenizeText expected mapping payloads for field '%s'; "
                + "got %s, tokenizing the value directly",
                self.field,
                type(payload).__name__,
            )
            self._warned_non_mapping = True
        text = str(payload)
        return text, {}

    def _set_payload_tensors(
        self, payload: SamplePayloadDict, ids: TokenSeq, mask: TokenSeq | None
    ) -> None:
        """Helper to centralize casting logic and keep _process clean."""
        # We must cast because SamplePayload definition is strict (Union),
        # but Tensors are handled as 'Any' or specific array types at runtime.
        payload["input_ids"] = cast(SamplePayload, ids)
        if self.add_attention_mask and mask is not None:
            payload["attention_mask"] = cast(SamplePayload, mask)

    def _bracket_one(
        self, ids: TokenSeq, mask: TokenSeq | None
    ) -> tuple[TokenSeq, TokenSeq | None]:
        """Prepend BOS / append EOS to one (ids, mask) pair per the configured mode.

        No-op when neither side adds a token (``tokenizer_default``, ``none``, or
        a mode that resolved no id). Backend-dispatched so list / numpy / torch /
        tensorflow values stay in their original container type.
        """
        bos = self._bos_id_resolved
        eos = self._eos_id_resolved
        if bos is None and eos is None:
            return ids, mask

        module = type(ids).__module__
        if "torch" in module:
            ids_t = cast("torch.Tensor", ids)
            mask_t = cast("torch.Tensor | None", mask)
            new_ids_t, new_mask_t = self._bracket_torch(ids_t, mask_t, bos, eos)
            return (
                cast(TokenSeq, new_ids_t),
                cast(TokenSeq | None, new_mask_t),
            )
        if "numpy" in module:
            ids_np = cast("np.ndarray", ids)
            mask_np = cast("np.ndarray | None", mask)
            new_ids_np, new_mask_np = self._bracket_numpy(ids_np, mask_np, bos, eos)
            return (
                cast(TokenSeq, new_ids_np),
                cast(TokenSeq | None, new_mask_np),
            )
        if "tensorflow" in module:
            ids_tf = cast("tf.Tensor", ids)
            mask_tf = cast("tf.Tensor | None", mask)
            new_ids_tf, new_mask_tf = self._bracket_tf(ids_tf, mask_tf, bos, eos)
            return (
                cast(TokenSeq, new_ids_tf),
                cast(TokenSeq | None, new_mask_tf),
            )
        ids_std = cast(Sequence[int], ids)
        mask_std = cast(Sequence[int] | None, mask)
        new_ids_std, new_mask_std = self._bracket_std(ids_std, mask_std, bos, eos)
        return (
            cast(TokenSeq, new_ids_std),
            cast(TokenSeq | None, new_mask_std),
        )

    def _bracket_std(
        self,
        ids: Sequence[int],
        mask: Sequence[int] | None,
        bos: int | None,
        eos: int | None,
    ) -> tuple[list[int], list[int] | None]:
        # Always emit a fresh list so the upstream batch container is not mutated.
        new_ids: list[int] = []
        if bos is not None:
            new_ids.append(bos)
        new_ids.extend(ids)
        if eos is not None:
            new_ids.append(eos)

        new_mask: list[int] | None = None
        if mask is not None:
            new_mask = []
            if bos is not None:
                new_mask.append(1)
            new_mask.extend(mask)
            if eos is not None:
                new_mask.append(1)
        elif self.add_attention_mask:
            new_mask = [1] * len(new_ids)
        return new_ids, new_mask

    def _bracket_numpy(
        self,
        ids: "np.ndarray",
        mask: "np.ndarray | None",
        bos: int | None,
        eos: int | None,
    ) -> tuple["np.ndarray", "np.ndarray | None"]:
        import numpy as np

        id_parts: list[np.ndarray] = []
        if bos is not None:
            id_parts.append(np.array([bos], dtype=ids.dtype))
        id_parts.append(ids)
        if eos is not None:
            id_parts.append(np.array([eos], dtype=ids.dtype))
        new_ids = np.concatenate(id_parts) if len(id_parts) > 1 else id_parts[0]

        new_mask: np.ndarray | None = None
        if mask is not None:
            m_parts: list[np.ndarray] = []
            if bos is not None:
                m_parts.append(np.array([1], dtype=mask.dtype))
            m_parts.append(mask)
            if eos is not None:
                m_parts.append(np.array([1], dtype=mask.dtype))
            new_mask = np.concatenate(m_parts) if len(m_parts) > 1 else m_parts[0]
        elif self.add_attention_mask:
            new_mask = np.ones_like(new_ids)
        return new_ids, new_mask

    def _bracket_torch(
        self,
        ids: "torch.Tensor",
        mask: "torch.Tensor | None",
        bos: int | None,
        eos: int | None,
    ) -> tuple["torch.Tensor", "torch.Tensor | None"]:
        import torch

        with _tensor_lock_ctx():
            id_parts: list[torch.Tensor] = []
            if bos is not None:
                id_parts.append(torch.tensor([bos], dtype=ids.dtype, device=ids.device))
            id_parts.append(ids)
            if eos is not None:
                id_parts.append(torch.tensor([eos], dtype=ids.dtype, device=ids.device))
            new_ids = torch.cat(id_parts, dim=-1) if len(id_parts) > 1 else id_parts[0]

            new_mask: torch.Tensor | None = None
            if mask is not None:
                m_parts: list[torch.Tensor] = []
                if bos is not None:
                    m_parts.append(
                        torch.tensor([1], dtype=mask.dtype, device=mask.device)
                    )
                m_parts.append(mask)
                if eos is not None:
                    m_parts.append(
                        torch.tensor([1], dtype=mask.dtype, device=mask.device)
                    )
                new_mask = (
                    torch.cat(m_parts, dim=-1) if len(m_parts) > 1 else m_parts[0]
                )
            elif self.add_attention_mask:
                new_mask = torch.ones_like(new_ids)
            return new_ids, new_mask

    def _bracket_tf(
        self,
        ids: "tf.Tensor",
        mask: "tf.Tensor | None",
        bos: int | None,
        eos: int | None,
    ) -> tuple["tf.Tensor", "tf.Tensor | None"]:
        import tensorflow as tf

        id_parts: list[tf.Tensor] = []
        if bos is not None:
            id_parts.append(tf.constant([bos], dtype=ids.dtype))
        id_parts.append(ids)
        if eos is not None:
            id_parts.append(tf.constant([eos], dtype=ids.dtype))
        new_ids = tf.concat(id_parts, axis=-1) if len(id_parts) > 1 else id_parts[0]

        new_mask: tf.Tensor | None = None
        if mask is not None:
            m_parts: list[tf.Tensor] = []
            if bos is not None:
                m_parts.append(tf.constant([1], dtype=mask.dtype))
            m_parts.append(mask)
            if eos is not None:
                m_parts.append(tf.constant([1], dtype=mask.dtype))
            new_mask = tf.concat(m_parts, axis=-1) if len(m_parts) > 1 else m_parts[0]
        elif self.add_attention_mask:
            new_mask = tf.ones_like(new_ids)
        return new_ids, new_mask

    def _pad_after_bracket(
        self,
        input_ids: Sequence[TokenSeq],
        attention_mask: Sequence[TokenSeq] | None,
    ) -> tuple[list[TokenSeq], list[TokenSeq] | None]:
        """Pad bracketed sequences to a uniform length.

        Matches HF: ``"max_length"`` pads every row to ``self.max_length``;
        ``True``/``"longest"`` pads to the batch longest regardless of any
        max_length cap (HF only honors the cap when truncation is also
        enabled).
        """
        pad_id = self._pad_token_id()
        n = len(input_ids)

        if self.padding == "max_length":
            assert self.max_length is not None
            target = self.max_length
        else:
            target = max((self._len_of(ids) for ids in input_ids), default=0)

        new_ids: list[TokenSeq] = []
        new_masks: list[TokenSeq] | None = [] if attention_mask is not None else None
        for i in range(n):
            ids = input_ids[i]
            mask = attention_mask[i] if attention_mask is not None else None
            cur_len = self._len_of(ids)
            if cur_len >= target:
                new_ids.append(ids)
                if new_masks is not None and mask is not None:
                    new_masks.append(mask)
                continue
            pad_len = target - cur_len
            new_ids.append(self._pad_one(ids, pad_len, pad_id))
            if new_masks is not None and mask is not None:
                new_masks.append(self._pad_one(mask, pad_len, 0))
        return new_ids, new_masks

    @staticmethod
    def _len_of(seq: TokenSeq) -> int:
        shape = getattr(seq, "shape", None)
        if shape is not None:
            try:
                return int(shape[0])
            except (TypeError, IndexError):
                pass
        return len(cast(Sequence[int], seq))

    def _pad_one(self, seq: TokenSeq, pad_len: int, pad_val: int) -> TokenSeq:
        module = type(seq).__module__
        if "torch" in module:
            import torch

            t = cast("torch.Tensor", seq)
            with _tensor_lock_ctx():
                pad = torch.full((pad_len,), pad_val, dtype=t.dtype, device=t.device)
                return cast(TokenSeq, torch.cat((t, pad), dim=-1))
        if "numpy" in module:
            import numpy as np

            arr = cast("np.ndarray", seq)
            return cast(
                TokenSeq,
                np.concatenate((arr, np.full((pad_len,), pad_val, dtype=arr.dtype))),
            )
        if "tensorflow" in module:
            import tensorflow as tf

            t2 = cast("tf.Tensor", seq)
            pad = tf.fill((pad_len,), tf.cast(pad_val, t2.dtype))
            return cast(TokenSeq, tf.concat((t2, pad), axis=-1))
        # Lists here are unaliased (fresh from _bracket_std or owned by the
        # current HF encode result), so extend in place.
        if isinstance(seq, list):
            seq.extend([pad_val] * pad_len)
            return cast(TokenSeq, seq)
        seq_list = list(cast(Sequence[int], seq))
        seq_list.extend([pad_val] * pad_len)
        return cast(TokenSeq, seq_list)

    def _process(self, elems: list[SampleRecord]) -> list[SampleRecord]:
        if not self._tokenizer_instantiated:
            self._setup_tokenizer()

        if self.tok is None:
            msg = "Tokenizer not initialised"
            raise RuntimeError(msg)

        texts: list[str] = []
        metas: list[SampleMeta] = []
        payloads: list[SamplePayloadDict] = []

        for elem in elems:
            text, payload = self._extract_text(elem.payload)
            texts.append(text)
            metas.append(elem.meta)
            payloads.append(payload)

        if not texts:
            return []

        encoded = self._tokenize_texts(texts)

        # On free-threaded Python + old PyTorch, we requested numpy from HF tokenizer
        # and convert to torch ourselves under lock to avoid allocator race condition.
        if self._convert_np_to_pt:
            encoded = self._convert_batch_np_to_torch(encoded)

        # Use type ignores here as TokenBatch is complex;
        # _normalize_batch handles the runtime safety.
        raw_input_ids: TokenBatch = encoded.get("input_ids", [])  # type: ignore[assignment]
        raw_attention_mask: TokenBatch | None = encoded.get("attention_mask")  # type: ignore[assignment]

        input_ids = self._normalize_batch(raw_input_ids, len(metas))

        attention_mask: Sequence[TokenSeq] | None = None
        if raw_attention_mask is not None:
            attention_mask = self._normalize_batch(raw_attention_mask, len(metas))

        # Bracket before any split: BOS lands on the first chunk, EOS on the
        # last; neither appears at slice boundaries.
        if self._zephon_owns_brackets and self._num_specials_total() > 0:
            bracketed_ids: list[TokenSeq] = []
            bracketed_mask: list[TokenSeq] | None = (
                [] if (attention_mask is not None or self.add_attention_mask) else None
            )
            for i in range(len(metas)):
                ids_i = input_ids[i]
                mask_i = attention_mask[i] if attention_mask is not None else None
                new_ids, new_mask = self._bracket_one(ids_i, mask_i)
                bracketed_ids.append(new_ids)
                if bracketed_mask is not None and new_mask is not None:
                    bracketed_mask.append(new_mask)
            input_ids = bracketed_ids
            attention_mask = bracketed_mask if bracketed_mask else None

        # Skip in split mode — the split path pads its own last chunk.
        if self._zephon_owns_brackets and self.padding and not self.split_long_samples:
            input_ids, attention_mask = self._pad_after_bracket(
                input_ids, attention_mask
            )

        # Bracket mode pulled lists from HF; lift to the user's backend now.
        if self._zephon_owns_brackets and self.return_tensors is not None and input_ids:
            input_ids = [self._convert_tensor(seq) for seq in input_ids]
            if attention_mask is not None:
                attention_mask = [self._convert_tensor(seq) for seq in attention_mask]

        # 3. FAST PATH: In-Place Reuse (Zero Allocation)
        if not self.split_long_samples:
            if attention_mask is None:
                # Zip input elements directly to modify them
                for elem, payload, ids in zip(elems, payloads, input_ids):
                    self._set_payload_tensors(payload, ids, None)
                    elem.payload = payload
            else:
                for elem, payload, ids, mask in zip(
                    elems, payloads, input_ids, attention_mask
                ):
                    self._set_payload_tensors(payload, ids, mask)
                    elem.payload = payload

            # Return the original list structure
            return elems

        # 4. Slower path: Splitting logic required (Must allocate new records)
        results: list[SampleRecord] = []
        for idx, meta in enumerate(metas):
            payload = payloads[idx]
            ids = input_ids[idx]
            mask = attention_mask[idx] if attention_mask is not None else None
            results.extend(self._maybe_split_outputs(meta, payload, ids, mask))
        return results

    def process_one(self, elem: SampleRecord) -> list[SampleRecord]:
        return self._process([elem])

    def process_many(self, elems: list[SampleRecord]) -> list[SampleRecord]:
        return self._process(elems)

    def _ensure_padding_token(self) -> None:
        if not self.padding:
            return
        tokenizer = self.tok
        if tokenizer is None:
            return
        if getattr(tokenizer, "pad_token", None) is not None:
            return
        eos_token = getattr(tokenizer, "eos_token", None)
        if eos_token is None:
            return
        tokenizer.pad_token = eos_token

    def _tokenize_texts(self, texts: list[str]) -> TokenizerOutput:
        tokenizer = self.tok
        if tokenizer is None:
            msg = "Tokenizer not initialised"
            raise RuntimeError(msg)
        self._ensure_padding_token()
        return tokenizer(texts, **self._cached_kwargs)

    def _convert_batch_np_to_torch(
        self, encoded: TokenizerOutput
    ) -> dict[str, TokenBatch]:
        """Convert numpy arrays to torch tensors under the lock.

        Used on free-threaded Python + old PyTorch to avoid allocator race
        condition when HF tokenizer creates tensors internally.
        """
        import torch

        with _tensor_lock_ctx():
            return {
                k: torch.from_numpy(v) if "numpy" in type(v).__module__ else v
                for k, v in encoded.items()
            }

    def _convert_tensor(self, value: TokenSeq) -> TokenSeq:
        backend = self.return_tensors
        if backend is None:
            return value
        src_mod = type(value).__module__
        if backend == "pt":
            torch = self._lazy_import("torch")
            if hasattr(value, "shape") and "torch" in src_mod:
                return value
            # On free-threaded Python + PyTorch < 2.10, detour list inputs
            # through numpy so ``from_numpy`` (under lock) is the only torch
            # allocation we do — numpy is unaffected by the allocator race.
            # See: https://github.com/pytorch/pytorch/issues/171992
            if hasattr(value, "shape") and "numpy" in src_mod:
                with _tensor_lock_ctx():
                    return torch.from_numpy(value)
            if _should_use_tensor_lock():
                np = self._lazy_import("numpy")
                arr = np.asarray(value)
                with _tensor_lock_ctx():
                    return torch.from_numpy(arr)
            with _tensor_lock_ctx():
                return torch.tensor(value)
        if backend in ("np", "numpy"):
            np = self._lazy_import("numpy")
            return (
                value
                if hasattr(value, "shape") and "numpy" in src_mod
                else np.asarray(value)
            )
        if backend == "tf":
            tf = self._lazy_import("tensorflow")
            return (
                value
                if hasattr(value, "shape") and "tensorflow" in src_mod
                else tf.convert_to_tensor(value)
            )
        raise ValueError(f"Unsupported return_tensors backend: {backend}")

    def _lazy_import(self, module: str) -> ModuleType:
        try:
            return __import__(module)
        except Exception as exc:
            raise RuntimeError(
                f"return_tensors='{self.return_tensors}' requested but {module} is unavailable"
            ) from exc

    def _normalize_batch(
        self, batch: TokenBatch, batch_size: int
    ) -> Sequence[TokenSeq]:
        """Ensure tokenizer outputs align with requested batch size.

        This handles edge cases where batch_size=1 causes some backends
        to squeeze dimensions, or where we need to ensure iterability.
        """
        # 1. Handle Lists (Fastest check for common HF output)
        if isinstance(batch, list):
            if len(batch) == batch_size:
                return cast(Sequence[TokenSeq], batch)
            # Mismatch implies it's a single unbatched sequence needing wrapping
            return [cast(TokenSeq, batch)]

        # 2. Handle Tuples
        if isinstance(batch, tuple):
            if len(batch) == batch_size:
                return cast(Sequence[TokenSeq], batch)
            return [cast(TokenSeq, batch)]

        # 3. Handle Tensors (Attributes check)
        shape = getattr(batch, "shape", None)

        if shape is not None:
            # Check dimension 0. Safety: len(shape) check handles 0-d scalars.
            if len(shape) > 0 and shape[0] == batch_size:
                # On free-threaded Python, convert PyTorch tensors to tuples to avoid
                # a race condition in the allocator during iteration.
                # See: https://github.com/pytorch/pytorch/issues/171992
                if _TENSOR_ITER_LOCK is not None and "torch" in type(batch).__module__:
                    with _TENSOR_ITER_LOCK:
                        return cast(Sequence[TokenSeq], batch.unbind(0))  # type: ignore[union-attr]
                return cast(Sequence[TokenSeq], batch)

            # If shape mismatch or scalar, wrap it in a list.
            # This is backend-agnostic (works for torch/tf/np) and functionally
            # equivalent to unsqueeze(0) when iterated in zip().
            return [cast(TokenSeq, batch)]

        # 4. Fallback (Unknown type, possibly just a Sequence of ints)
        if len(batch) == batch_size:  # type: ignore
            return cast(Sequence[TokenSeq], batch)
        return [cast(TokenSeq, batch)]

    def _pad_token_id(self) -> int:
        return self._pad_token_id_cached

    def _compute_pad_token_id(self) -> int:
        tok = self.tok
        if tok is None:
            return 0
        pad_id = getattr(tok, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(tok, "pad_token", None)
        try:
            return int(pad_id) if pad_id is not None else 0
        except Exception:
            return 0

    def _split_segments(
        self, ids: T_TokenSeq, mask: T_TokenSeq | None
    ) -> list[tuple[T_TokenSeq, T_TokenSeq | None]]:
        max_len = self.max_length
        assert max_len is not None
        pad_id = self._pad_token_id()

        module = type(ids).__module__

        # Dispatch based on module string to avoid strict isinstance imports
        if "torch" in module:
            ids_t = cast("torch.Tensor", ids)
            mask_t = cast("torch.Tensor | None", mask)

            result_torch = self._split_segments_torch(ids_t, mask_t, max_len, pad_id)
            return cast(list[tuple[T_TokenSeq, T_TokenSeq | None]], result_torch)

        if "numpy" in module:
            ids_np = cast("np.ndarray", ids)
            mask_np = cast("np.ndarray | None", mask)

            result_np = self._split_segments_numpy(ids_np, mask_np, max_len, pad_id)
            return cast(list[tuple[T_TokenSeq, T_TokenSeq | None]], result_np)

        if "tensorflow" in module:
            ids_tf = cast("tf.Tensor", ids)
            mask_tf = cast("tf.Tensor | None", mask)

            result_tf = self._split_segments_tf(ids_tf, mask_tf, max_len, pad_id)
            return cast(list[tuple[T_TokenSeq, T_TokenSeq | None]], result_tf)

        # Standard Python List/Sequence path
        # Explicitly cast to Sequence to satisfy helper
        ids_std = cast(Sequence[int], ids)
        mask_std = cast(Sequence[int] | None, mask)

        result_std = self._split_segments_std(ids_std, mask_std, max_len, pad_id)
        return cast(list[tuple[T_TokenSeq, T_TokenSeq | None]], result_std)

    def _split_segments_std(
        self,
        ids: Sequence[int],
        mask: Sequence[int] | None,
        max_len: int,
        pad_id: int,
    ) -> list[tuple[list[int], list[int] | None]]:
        # Performance: Avoid copy if already list
        ids_list = ids if isinstance(ids, list) else list(ids)

        mask_list: list[int] | None = None
        if mask is not None:
            mask_list = mask if isinstance(mask, list) else list(mask)

        if mask_list is None and self.add_attention_mask:
            mask_list = [1] * len(ids_list)

        segments: list[tuple[list[int], list[int] | None]] = []
        if not ids_list:
            empty_ids: list[int] = []
            empty_mask: list[int] | None = [] if mask_list is not None else None
            if self.padding:
                empty_ids = [pad_id] * max_len
                if self.add_attention_mask:
                    empty_mask = [0] * max_len
            segments.append((empty_ids, empty_mask))
            return segments

        for start in range(0, len(ids_list), max_len):
            # Slicing creates a copy, which is efficient for ints.
            seg_ids = ids_list[start : start + max_len]
            seg_mask = (
                mask_list[start : start + max_len] if mask_list is not None else None
            )
            if self.padding and len(seg_ids) < max_len:
                pad_len = max_len - len(seg_ids)
                # Performance: Use extend instead of '+' to modify the slice list in-place
                seg_ids.extend([pad_id] * pad_len)
                if seg_mask is not None:
                    seg_mask.extend([0] * pad_len)
            segments.append((seg_ids, seg_mask))
        return segments

    def _split_segments_torch(
        self,
        ids: "torch.Tensor",
        mask: "torch.Tensor | None",
        max_len: int,
        pad_id: int,
    ) -> list[tuple["torch.Tensor", "torch.Tensor | None"]]:
        import torch

        with _tensor_lock_ctx():
            segments: list[tuple[torch.Tensor, torch.Tensor | None]] = []
            if mask is None and self.add_attention_mask:
                mask = torch.ones_like(ids, dtype=ids.dtype)

            total = ids.shape[0]
            if total == 0:
                seg_ids = ids
                seg_mask = mask
                if self.padding:
                    seg_ids = torch.full(
                        (max_len,), pad_id, dtype=ids.dtype, device=ids.device
                    )
                    if self.add_attention_mask:
                        seg_mask = torch.zeros_like(seg_ids)
                segments.append((seg_ids, seg_mask))
                return segments

            for start in range(0, total, max_len):
                end = min(total, start + max_len)
                seg_ids = ids[start:end]
                seg_mask = mask[start:end] if mask is not None else None
                seg_len = end - start
                if self.padding and seg_len < max_len:
                    pad_len = max_len - seg_len
                    pad = torch.full(
                        (pad_len,), pad_id, dtype=ids.dtype, device=ids.device
                    )
                    seg_ids = torch.cat((seg_ids, pad), dim=-1)
                    if seg_mask is not None:
                        pad_mask = torch.zeros(
                            (pad_len,), dtype=seg_mask.dtype, device=seg_ids.device
                        )
                        seg_mask = torch.cat((seg_mask, pad_mask), dim=-1)
                segments.append((seg_ids, seg_mask))
            return segments

    def _split_segments_numpy(
        self,
        ids: "np.ndarray",
        mask: "np.ndarray | None",
        max_len: int,
        pad_id: int,
    ) -> list[tuple["np.ndarray", "np.ndarray | None"]]:
        import numpy as np

        segments: list[tuple[np.ndarray, np.ndarray | None]] = []
        if mask is None and self.add_attention_mask:
            mask = np.ones_like(ids, dtype=ids.dtype)

        total = ids.shape[0]
        if total == 0:
            seg_ids = ids
            seg_mask = mask
            if self.padding:
                seg_ids = np.full((max_len,), pad_id, dtype=ids.dtype)
                if self.add_attention_mask:
                    seg_mask = np.zeros((max_len,), dtype=ids.dtype)
            segments.append((seg_ids, seg_mask))
            return segments

        for start in range(0, total, max_len):
            end = min(total, start + max_len)
            seg_ids = ids[start:end]
            seg_mask = mask[start:end] if mask is not None else None
            seg_len = end - start
            if self.padding and seg_len < max_len:
                pad_len = max_len - seg_len
                seg_ids = np.concatenate(
                    (seg_ids, np.full((pad_len,), pad_id, dtype=ids.dtype))
                )
                if seg_mask is not None:
                    seg_mask = np.concatenate(
                        (seg_mask, np.zeros((pad_len,), dtype=seg_mask.dtype))
                    )
            segments.append((seg_ids, seg_mask))
        return segments

    def _split_segments_tf(
        self,
        ids: "tf.Tensor",
        mask: "tf.Tensor | None",
        max_len: int,
        pad_id: int,
    ) -> list[tuple["tf.Tensor", "tf.Tensor | None"]]:
        import tensorflow as tf

        segments: list[tuple[tf.Tensor, tf.Tensor | None]] = []
        total_dim = ids.shape[0]
        total = (
            int(total_dim) if total_dim is not None else int(tf.shape(ids)[0].numpy())
        )
        if mask is None and self.add_attention_mask:
            mask = tf.ones_like(ids)

        if total == 0:
            seg_ids = ids
            seg_mask = mask
            if self.padding:
                seg_ids = tf.fill((max_len,), tf.cast(pad_id, ids.dtype))
                if self.add_attention_mask:
                    seg_mask = tf.zeros((max_len,), dtype=ids.dtype)
            segments.append((seg_ids, seg_mask))
            return segments

        for start in range(0, total, max_len):
            end = min(total, start + max_len)
            seg_ids = ids[start:end]
            seg_mask = mask[start:end] if mask is not None else None
            seg_len = end - start
            if self.padding and seg_len < max_len:
                pad_len = max_len - seg_len
                pad = tf.fill((pad_len,), tf.cast(pad_id, ids.dtype))
                seg_ids = tf.concat((seg_ids, pad), axis=-1)
                if seg_mask is not None:
                    pad_mask = tf.zeros((pad_len,), dtype=seg_mask.dtype)
                    seg_mask = tf.concat((seg_mask, pad_mask), axis=-1)
            segments.append((seg_ids, seg_mask))
        return segments

    def _maybe_split_outputs(
        self,
        meta: SampleMeta,
        payload: SamplePayloadDict,
        ids: T_TokenSeq,
        mask: T_TokenSeq | None,
    ) -> list[SampleRecord]:
        if not self.split_long_samples or self.max_length is None:
            self._set_payload_tensors(payload, ids, mask)
            return [SampleRecord(meta=meta, payload=payload)]

        segments = self._split_segments(ids, mask)

        if len(segments) == 1:
            seg_ids, seg_mask = segments[0]
            # Convert backend if needed (e.g. return_tensors='pt')
            conv_ids = self._convert_tensor(seg_ids)
            conv_mask = self._convert_tensor(seg_mask) if seg_mask is not None else None
            self._set_payload_tensors(payload, conv_ids, conv_mask)
            return [SampleRecord(meta=meta, payload=payload)]

        records: list[SampleRecord] = []
        last_idx = len(segments) - 1
        for idx, (seg_ids, seg_mask) in enumerate(segments):
            child_meta = spawn_child(meta, idx, is_last_child=(idx == last_idx))
            child_payload = dict(payload)

            conv_ids = self._convert_tensor(seg_ids)
            conv_mask = self._convert_tensor(seg_mask) if seg_mask is not None else None

            self._set_payload_tensors(child_payload, conv_ids, conv_mask)
            records.append(SampleRecord(meta=child_meta, payload=child_payload))
        return records
