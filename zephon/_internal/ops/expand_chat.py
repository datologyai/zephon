# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Conversation prefixes ending at answered user-query boundaries."""

import copy
import logging
from collections.abc import Mapping, Sequence
from typing import Any, cast

from zephon.ops.base import BaseOp
from zephon.ops.children import spawn_child, tombstones_for_record
from zephon.ops.traits import OpTraits
from zephon.types import ContributorRef, SampleMeta, SampleRecord

log = logging.getLogger(__name__)


class ExpandChat(BaseOp):
    """Emit one prefix per answered user query, retaining its complete tool cycle.

    Each A below includes every assistant/tool exchange before the next user::

        input:     U1 -> A1 -> U2 -> A2 -> U3 (unanswered)
        prefixes:  U1 -> A1
                   U1 -> A1 -> U2 -> A2

    Empty assistant content counts as an answer. Normalize roles and filter
    accidental empty answers upstream. Expansion does not select loss targets;
    configure tokenize_chat separately and isolate packed prefixes in attention.

    Multiple prefixes append original user-query ordinals to cursor lineage;
    unanswered queries leave gaps. A sole prefix retains the parent's lineage.
    Siblings own independent nested payloads; the final prefix consumes the source.

    Calibration counts all prefix tokens against original input bytes: four
    100-token cycles emit 100 + 200 + 300 + 400 = 1,000 tokens. Custom estimators
    must include this expansion. Keep expansion and tokenization in one stage
    to avoid serializing repeated history between processes.
    """

    def __init__(self, *, field: str = "messages") -> None:
        super().__init__()
        if any(not part for part in field.split(".")):
            raise ValueError("field must name a non-empty messages payload path")
        self.field = field
        self._field_path = tuple(field.split("."))
        self._warned_no_user = False

    def traits(self) -> OpTraits:
        return OpTraits(indexable=False, preserves_cursor_order=True, parallelism=4)

    def _messages(self, record: SampleRecord) -> Sequence[Mapping[str, Any]]:
        """Resolve the field path and validate each message's role.

        ``field="chat.messages"`` reads ``payload -> chat -> messages``.
        """
        value = record.payload
        for key in self._field_path:
            if not isinstance(value, Mapping) or key not in value:
                raise ValueError(f"expand_chat: missing field {self.field!r}")
            value = value[key]
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise TypeError(
                f"expand_chat: {self.field!r} must be a sequence of messages"
            )
        for index, message in enumerate(value):
            if (
                not isinstance(message, Mapping)
                or not isinstance(message.get("role"), str)
                or not message["role"]
            ):
                raise ValueError(
                    f"expand_chat: message {index} must have a non-empty string role"
                )
        return cast(Sequence[Mapping[str, Any]], value)

    def _prefix_payload(
        self, payload: Any, messages: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        """Copy containers along the field path and replace the message sequence.

        ``chat.messages: [U1, A1, U2, A2] -> [U1, A1]``; other fields are shared
        until process_many deep-copies the non-final siblings.
        """
        result = current = dict(payload)
        source = payload
        for key in self._field_path[:-1]:
            source = source[key]
            current[key] = dict(source)
            current = current[key]
        current[self._field_path[-1]] = messages
        return result

    @staticmethod
    def _child_meta(meta: SampleMeta, ordinal: int, last: bool) -> SampleMeta:
        """Append the query ordinal; only the final prefix inherits closing refs.

        Lineage ``(4,) -> (4, ordinal)``. Inherited flags ``[open, closed]``
        become ``[open, open]`` before the last prefix, then ``[open, closed]``.
        """
        if not meta.contributors:
            return spawn_child(meta, ordinal, is_last_child=last)
        # Only the final prefix inherits its parent's closing contributors.
        opened = tuple(ContributorRef(ref.cursor, False) for ref in meta.contributors)
        child = spawn_child(meta.with_contributors(opened), ordinal)
        return child.with_contributors(meta.contributors if last else opened)

    def process_many(self, elems: list[SampleRecord]) -> list[SampleRecord]:
        """Scan user boundaries, then emit prefixes or closing tombstones."""
        out: list[SampleRecord] = []
        for record in elems:
            if record.meta.tombstone:
                out.append(record)
                continue
            messages = self._messages(record)
            # Each entry is (user-query ordinal, exclusive prefix end).
            prefixes: list[tuple[int, int]] = []
            ordinal, answered = -1, False
            for index, message in enumerate(messages):
                if message["role"] == "user":
                    # Unanswered queries have no prefix of their own.
                    if answered:
                        prefixes.append((ordinal, index))
                    ordinal, answered = ordinal + 1, False
                elif message["role"] == "assistant" and ordinal >= 0:
                    answered = True
            if answered:
                prefixes.append((ordinal, len(messages)))
            if ordinal < 0:
                # No user message was encountered; warn once per operator.
                if not self._warned_no_user:
                    log.warning(
                        "expand_chat: dropping conversations with no user message"
                    )
                    self._warned_no_user = True
            if not prefixes:
                # No query has an assistant message; emit only completion markers.
                out.extend(tombstones_for_record(record))
                continue
            if len(prefixes) == 1:
                # One answered query yields one output with the input's metadata.
                end = prefixes[0][1]
                if end == len(messages):
                    # The prefix covers the entire conversation.
                    out.append(record)
                else:
                    # Remove trailing unanswered queries from the sole prefix.
                    out.append(
                        SampleRecord(
                            meta=record.meta,
                            payload=self._prefix_payload(
                                record.payload, messages[:end]
                            ),
                        )
                    )
                continue
            for index, (ordinal, end) in enumerate(prefixes):
                last = index == len(prefixes) - 1
                meta = self._child_meta(record.meta, ordinal, last)
                payload = self._prefix_payload(record.payload, messages[:end])
                # Siblings own independent nested data; the final prefix consumes
                # the source objects after all earlier prefixes have been copied.
                if not last:
                    payload = copy.deepcopy(payload)
                out.append(
                    SampleRecord(
                        meta=meta,
                        payload=payload,
                    )
                )
        return out
