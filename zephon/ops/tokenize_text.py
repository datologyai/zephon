# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Tokenizer operators that prepare text fields for model consumption."""

import logging
from typing import Any, Optional

from zephon.core.constants import Element, SampleRecord
from zephon.core.op_base import DefaultFinalize, OpContext
from zephon.core.traits import Buffering, OpTraits

log = logging.getLogger(__name__)


class TokenizeText(DefaultFinalize):
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
        self.tok = tokenizer
        self.tokenizer_id = tokenizer_id
        self.field = field
        self.add_attention_mask = add_attention_mask
        self._buffering = buffering or Buffering(max_batch=64, max_latency_ms=3)

    def setup(self, ctx: OpContext) -> None:
        if self.tok is None:
            if self.tokenizer_id in (None, "__fallback__"):
                self.tok = _fallback_tokenizer()
            else:
                try:
                    from transformers import AutoTokenizer

                    self.tok = AutoTokenizer.from_pretrained(self.tokenizer_id)
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "Falling back to toy tokenizer after HF load error: %s", exc
                    )
                    self.tok = _fallback_tokenizer()

    def traits(self) -> OpTraits:
        return OpTraits(indexable=True, parallelism=4)

    def buffering(self) -> Optional[Buffering]:
        return self._buffering

    def process_one(self, elem: Element) -> list[Element]:
        assert isinstance(elem, SampleRecord)
        tokenizer = self.tok
        if tokenizer is None:
            msg = "Tokenizer not initialised"
            raise RuntimeError(msg)
        text = elem.payload.get(self.field, "")
        encoded = tokenizer(
            text, add_special_tokens=True, padding=False, truncation=False
        )
        payload = dict(elem.payload)
        payload["input_ids"] = encoded["input_ids"]
        if self.add_attention_mask and "attention_mask" in encoded:
            payload["attention_mask"] = encoded["attention_mask"]
        return [SampleRecord(meta=elem.meta, payload=payload)]

    def process_many(self, elems: list[Element]) -> list[Element]:
        tokenizer = self.tok
        if tokenizer is None:
            msg = "Tokenizer not initialised"
            raise RuntimeError(msg)
        texts: list[str] = []
        metas: list[Any] = []
        payloads: list[dict[str, Any]] = []
        for elem in elems:
            assert isinstance(elem, SampleRecord), f"elem is {type(elem)} = {elem}"
            texts.append(elem.payload.get(self.field, ""))
            metas.append(elem.meta)
            payloads.append(elem.payload)
        encoded = tokenizer(
            texts, add_special_tokens=True, padding=False, truncation=False
        )
        input_ids = encoded.get("input_ids", [])
        attention_mask = encoded.get("attention_mask")
        results: list[Element] = []
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
