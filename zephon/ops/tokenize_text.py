# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Tokenizer operators that prepare text fields for model consumption."""

import logging
from typing import Any, Optional

from zephon.core.constants import SamplePayload, SamplePayloadDict, SampleRecord
from zephon.core.op_base import DefaultFinalize, DefaultSetup, OpContext
from zephon.core.traits import Buffering, OpTraits

log = logging.getLogger(__name__)


class TokenizeText(DefaultSetup, DefaultFinalize[SampleRecord]):
    """Tokenize text fields using a provided or auto-resolved tokenizer."""

    def __init__(
        self,
        tokenizer: Any | None = None,
        tokenizer_id: str | None = None,
        *,
        field: str = "text",
        add_attention_mask: bool = True,
        buffering: Optional[Buffering] = None,
    ) -> None:
        DefaultSetup.__init__(self)

        self.tok = tokenizer
        self.tokenizer_id = tokenizer_id
        self.field = field
        self.add_attention_mask = add_attention_mask
        self._buffering = buffering or Buffering(max_batch=64, max_latency_ms=3)

    def setup(
        self,
        ctx: OpContext,
        stage_index: int,
        stage_name: str,
        op_index: int,
        collect_stats: bool,
    ) -> None:
        DefaultSetup.setup(self, ctx, stage_index, stage_name, op_index, collect_stats)
        if self.tok is None:
            if self.tokenizer_id in (None, "__fallback__"):
                self.tok = _fallback_tokenizer()
            else:
                try:
                    from transformers import AutoTokenizer

                    self.tok = AutoTokenizer.from_pretrained(self.tokenizer_id)
                except Exception as exc:
                    log.warning(
                        "Falling back to toy tokenizer after HF load error: %s", exc
                    )
                    self.tok = _fallback_tokenizer()

    def traits(self) -> OpTraits:
        return OpTraits(indexable=True, parallelism=4)

    def buffering(self) -> Optional[Buffering]:
        return self._buffering

    def _expect_mapping(self, payload: SamplePayload) -> SamplePayloadDict:
        if not isinstance(payload, dict):
            raise TypeError("TokenizeText expects dict payloads")
        return payload

    def process_one(self, elem: SampleRecord) -> list[SampleRecord]:
        tokenizer = self.tok
        if tokenizer is None:
            msg = "Tokenizer not initialised"
            raise RuntimeError(msg)
        payload = self._expect_mapping(elem.payload)
        text = payload.get(self.field, "")
        encoded = tokenizer(
            text, add_special_tokens=True, padding=False, truncation=False
        )
        payload = dict(payload)
        payload["input_ids"] = encoded["input_ids"]
        if self.add_attention_mask and "attention_mask" in encoded:
            payload["attention_mask"] = encoded["attention_mask"]
        return [SampleRecord(meta=elem.meta, payload=payload)]

    def process_many(self, elems: list[SampleRecord]) -> list[SampleRecord]:
        tokenizer = self.tok
        if tokenizer is None:
            msg = "Tokenizer not initialised"
            raise RuntimeError(msg)
        texts: list[str] = []
        metas: list[Any] = []
        payloads: list[SamplePayloadDict] = []
        for elem in elems:
            payload = self._expect_mapping(elem.payload)
            texts.append(str(payload.get(self.field, "")))
            metas.append(elem.meta)
            payloads.append(payload)
        encoded = tokenizer(
            texts, add_special_tokens=True, padding=False, truncation=False
        )
        input_ids = encoded.get("input_ids", [])
        attention_mask = encoded.get("attention_mask")
        results: list[SampleRecord] = []
        for idx, meta in enumerate(metas):
            payload = dict(payloads[idx])
            payload["input_ids"] = input_ids[idx]
            if self.add_attention_mask and attention_mask is not None:
                payload["attention_mask"] = attention_mask[idx]
            results.append(SampleRecord(meta=meta, payload=payload))
        return results

    def resolved_tokenizer_id(self) -> str | None:
        if self.tokenizer_id is not None:
            return self.tokenizer_id
        tokeniser_name = getattr(self.tok, "name_or_path", None)
        if tokeniser_name is None:
            return "__fallback__"
        return str(tokeniser_name)


def _fallback_tokenizer() -> Any:
    class _Tokenizer:
        def __call__(
            self,
            texts: Any,
            add_special_tokens: bool = True,
            padding: bool = False,
            truncation: bool = False,
        ) -> dict[str, Any]:
            def encode_single(value: Any) -> dict[str, list[int]]:
                tokens = [abs(hash(word)) % 10000 + 1 for word in str(value).split()]
                return {"input_ids": tokens, "attention_mask": [1] * len(tokens)}

            if isinstance(texts, str):
                return encode_single(texts)

            encoded = [encode_single(item) for item in texts]
            return {
                "input_ids": [item["input_ids"] for item in encoded],
                "attention_mask": [item["attention_mask"] for item in encoded],
            }

    return _Tokenizer()
