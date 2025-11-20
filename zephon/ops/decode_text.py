# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Operators that decode text payloads from raw bytes or mixed content."""

from typing import Any, Optional, Sequence

from zephon.core.constants import SamplePayload, SamplePayloadDict, SampleRecord
from zephon.core.op_base import DefaultFinalize, DefaultSetup
from zephon.core.traits import Buffering, OpTraits


class DecodeText(DefaultSetup, DefaultFinalize[SampleRecord]):
    """Decode configured payload fields into normalised text."""

    def __init__(
        self,
        fields: Sequence[str] = ("text",),
        *,
        encoding: str = "utf-8",
        errors: str = "strict",
        normalize_newlines: bool = True,
        lowercase: bool = False,
        buffering: Optional[Buffering] = None,
    ) -> None:
        DefaultSetup.__init__(self)
        self.fields = tuple(fields)
        self.encoding = encoding
        self.errors = errors
        self.normalize_newlines = normalize_newlines
        self.lowercase = lowercase
        self._buffering = buffering or Buffering(max_batch=128, max_latency_ms=2)

    def traits(self) -> OpTraits:
        return OpTraits(indexable=True, parallelism=2)

    def buffering(self) -> Optional[Buffering]:
        return self._buffering

    def _decode(self, value: Any) -> str:
        if isinstance(value, bytes):
            result = value.decode(self.encoding, errors=self.errors)
        else:
            result = str(value)
        if self.normalize_newlines:
            result = result.replace("\r\n", "\n").replace("\r", "\n")
        if self.lowercase:
            result = result.lower()
        return result

    def _expect_mapping(self, payload: SamplePayload) -> SamplePayloadDict:
        if not isinstance(payload, dict):
            raise TypeError("DecodeText expects dict payloads")
        return payload

    def process_one(self, elem: SampleRecord) -> list[SampleRecord]:
        payload = dict(self._expect_mapping(elem.payload))
        for field in self.fields:
            if field in payload:
                payload[field] = self._decode(payload[field])
        return [SampleRecord(meta=elem.meta, payload=payload)]

    def process_many(self, elems: list[SampleRecord]) -> list[SampleRecord]:
        return [self.process_one(elem)[0] for elem in elems]
