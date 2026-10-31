# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Shared scaffolding for the tokenize-op family."""

from __future__ import annotations

import contextlib
import threading
from typing import Any, Literal, Mapping, Optional

from zephon._internal.token_counting import TokenCountingSpec
from zephon._internal.utils.tokenizer import TokenizerLike, load_hf_tokenizer
from zephon._internal.utils.torch_compat import _gil_disabled
from zephon.ops.accumulators import Accumulator, CountingAccumulator
from zephon.ops.base import BaseOp
from zephon.types import SampleRecord

_MISSING: object = object()

# Free-threaded Python only: serializes _setup_tokenizer's check-then-set so
# concurrent _process threads can't race into double-loading the tokenizer.
_SETUP_LOCK: threading.Lock | None = threading.Lock() if _gil_disabled() else None


class TokenizeBase(BaseOp):
    """Lazy-tokenizer scaffolding shared by the tokenize-op family.

    Owns the deferred tokenizer load (never in ``setup()`` — HF tokenizers
    must not be constructed before workers fork) and the failure-caching
    envelope around ``_finalize_setup``, where subclasses run their post-load
    configuration: a finalize failure is deterministic (bad config), so the
    same error re-raises on every subsequent batch instead of retrying
    half-initialised state. Loader failures are *not* cached — they are
    transient by nature (network) and a later batch may succeed.
    """

    def __init__(
        self,
        tokenizer: TokenizerLike | None,
        tokenizer_id: str | None,
        *,
        use_fast: bool | None,
        eos_token: str | None = None,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        max_batch: int,
        max_latency_ms: Optional[int],
    ) -> None:
        super().__init__()
        if eos_token is not None and (tokenizer is not None or tokenizer_id is None):
            raise ValueError(
                "eos_token only applies when the op loads tokenizer_id; pass "
                "tokenizer_id without a pre-instantiated tokenizer, or set "
                "the EOS on your tokenizer instead"
            )
        self.tok = tokenizer
        self.tokenizer_id = tokenizer_id
        self.use_fast = use_fast
        self.eos_token = eos_token
        self._bos_id_override = bos_token_id
        self._eos_id_override = eos_token_id
        self._eos_token: int | str | None = None
        self._max_batch = max_batch
        self._max_latency_ms = max_latency_ms
        self._tokenizer_instantiated = False
        self._setup_error: Exception | None = None

    def _setup_tokenizer(self) -> None:
        lock_ctx = _SETUP_LOCK if _SETUP_LOCK is not None else contextlib.nullcontext()
        with lock_ctx:
            if self._setup_error is not None:
                raise self._setup_error
            if self._tokenizer_instantiated:
                return

            if self.tok is None:
                self.tok = load_hf_tokenizer(
                    self.tokenizer_id,
                    use_fast=self.use_fast,
                    eos_token=self.eos_token,
                )

            try:
                self._eos_token = getattr(self.tok, "eos_token", None)
                self._finalize_setup()
            except Exception as exc:
                self._setup_error = exc
                raise
            self._tokenizer_instantiated = True

    def _resolve_special_token_id(
        self, name: Literal["bos", "eos"], *, required: bool = False
    ) -> int | None:
        """Use an explicit ID override, falling back to the loaded tokenizer."""
        value = self._bos_id_override if name == "bos" else self._eos_id_override
        if value is None:
            value = getattr(self.tok, f"{name}_token_id", None)
        if value is None and required:
            article = "a" if name == "bos" else "an"
            raise ValueError(
                f"{type(self).__name__} requires {article} {name.upper()} token; "
                + f"set {name}_token_id or configure the tokenizer"
            )
        return int(value) if value is not None else None

    def _finalize_setup(self) -> None:
        """Post-load, per-op configuration; failures are cached and re-raised."""

    def _cache_preflight_result(self, probe: TokenizeBase) -> None:
        """Retain reusable checks without copying the probe's initialized tokenizer."""

    def _lookup_path(self, payload: Any, path: tuple[str, ...]) -> Any:
        """Resolve a dot-path against a (possibly nested) mapping payload.

        Returns the ``_MISSING`` sentinel when any step is absent or
        non-mapping; subclasses decide whether that is lenient or fatal.
        """
        value: Any = payload
        for key in path:
            if not isinstance(value, Mapping):
                return _MISSING
            value = value.get(key, _MISSING)
            if value is _MISSING:
                return _MISSING
        return value

    def token_counting_spec(self) -> TokenCountingSpec:
        """Return the settings needed to reproduce delivered-token counts."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement token_counting_spec(), "
            f"so token-aware mixture priming cannot calibrate through it. "
            f"Implement the hook or pin ratios via TokenEstimation(primer=...)."
        )

    def configured_tokenizer_id(self) -> str | None:
        """Return the *configured* tokenizer id for observability.

        For ``tokenizer_id``-driven ops, this is the original string passed at
        construction time (HF Hub id, local path, or cloud URI) — not the
        local cache path that cloud URIs are resolved to. For pre-built
        tokenizers, this is ``tok.name_or_path``.
        """
        if self.tokenizer_id is not None:
            return self.tokenizer_id
        name = getattr(self.tok, "name_or_path", None)
        return "__fallback__" if name is None else str(name)

    def accumulator(
        self, *, deterministic: bool, ctx: dict[str, Any]
    ) -> Accumulator[SampleRecord]:
        return CountingAccumulator[SampleRecord](
            max_batch=self._max_batch,
            max_latency_ms=None if deterministic else self._max_latency_ms,
        )
