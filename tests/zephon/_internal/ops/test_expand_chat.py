# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

import copy
from typing import Any

import pytest

from zephon._internal.ops.expand_chat import ExpandChat
from zephon.ops.children import spawn_child, tombstones_for_record
from zephon.types import SampleMeta, SampleRecord
from zephon.validation import (
    _check_op_determinism,
    _check_op_no_cross_call_state,
    _check_op_sample_identity,
)


def record(*roles: str) -> SampleRecord:
    return SampleRecord(
        meta=SampleMeta(sample_id=(0, 0, 0), lane_id=0, chunk_id=7, chunk_offset=2),
        payload={
            "messages": [
                {"role": role, "content": str(i)} for i, role in enumerate(roles)
            ],
            "tools": [{"name": "lookup"}],
        },
    )


def test_query_prefixes_include_tool_cycles_and_original_ordinals() -> None:
    parent = record(
        "system",
        "assistant",
        "user",
        "user",
        "assistant",
        "user",
        "assistant",
        "tool",
        "assistant",
        "user",
    )
    original = copy.deepcopy(parent)
    outputs = ExpandChat().process_one(parent)
    assert [len(child.payload["messages"]) for child in outputs] == [5, 9]
    assert [child.meta.lineage for child in outputs] == [(1,), (2,)]
    assert [child.meta.contribution_refs()[0].is_last_child for child in outputs] == [
        False,
        True,
    ]
    assert all(child.meta.sample_id == parent.meta.sample_id for child in outputs)
    assert all(child.payload["tools"] == parent.payload["tools"] for child in outputs)
    assert parent == original
    assert len(tombstones_for_record(outputs[-1])) == 1
    assert tombstones_for_record(outputs[0]) == []


@pytest.mark.parametrize(
    "roles", [(), ("assistant",), ("user",), ("user", "tool", "user")]
)
def test_dropped_rows_close_and_tombstones_pass_through(roles: tuple[str, ...]) -> None:
    parent = record(*roles)
    op = ExpandChat()
    outputs = op.process_one(parent)
    assert len(outputs) == 1 and outputs[0].meta.tombstone
    assert outputs[0].meta.cursor == parent.meta.cursor
    assert op.process_one(outputs[0])[0] is outputs[0]
    parent.meta = spawn_child(parent.meta, 3, is_last_child=False)
    assert op.process_one(parent) == []


def test_single_prefix_keeps_metadata_and_nested_payload() -> None:
    parent = record("user", "assistant")
    parent.payload["messages"][-1]["content"] = None
    parent.payload["chat"] = {
        "messages": parent.payload.pop("messages"),
        "extra": "value",
    }
    op = ExpandChat(field="chat.messages")
    assert op.process_one(parent)[0] is parent
    parent.payload["chat"]["messages"].append({"role": "user", "content": "unanswered"})
    original_payload = parent.payload
    output = op.process_one(parent)[0]
    assert output is parent
    assert output.meta is parent.meta
    assert len(output.payload["chat"]["messages"]) == 2
    assert len(original_payload["chat"]["messages"]) == 3
    assert output.payload["chat"]["extra"] == "value"


@pytest.mark.parametrize(
    ("messages", "error", "reason"),
    [
        (None, TypeError, "must be a sequence of messages"),
        ("text", TypeError, "must be a sequence of messages"),
        ({}, TypeError, "must be a sequence of messages"),
        ([None], ValueError, "message 0"),
        ([{}], ValueError, "message 0"),
        ([{"role": None}], ValueError, "message 0"),
        ([{"role": ""}], ValueError, "message 0"),
        ([{"role": "user"}, {}], ValueError, "message 1"),
    ],
)
def test_bad_messages_raise(messages: Any, error: type[Exception], reason: str) -> None:
    parent = record()
    parent.payload["messages"] = messages
    with pytest.raises(error, match=reason) as exc_info:
        ExpandChat().process_one(parent)
    assert repr(parent.meta.sample_id) in str(exc_info.value)
    assert "'messages'" in str(exc_info.value)


@pytest.mark.parametrize(
    ("field", "payload"),
    [
        ("messages", {}),
        ("chat.messages", {}),
        ("chat.messages", {"chat": {}}),
        ("chat.messages", {"chat": None}),
    ],
)
def test_missing_field_reports_sample(field: str, payload: Any) -> None:
    parent = record()
    parent.payload = payload
    with pytest.raises(ValueError, match="expand_chat") as exc_info:
        ExpandChat(field=field).process_one(parent)
    assert repr(parent.meta.sample_id) in str(exc_info.value)
    assert repr(field) in str(exc_info.value)


def test_builtin_identity_determinism_and_cross_call_harness() -> None:
    samples = [record("user", "assistant", "user", "assistant", "tool")]
    op = ExpandChat()
    assert not _check_op_determinism(op, "expand_chat", samples)
    assert not _check_op_no_cross_call_state(op, "expand_chat", samples)
    assert not _check_op_sample_identity(op, "expand_chat", samples)


def test_prefix_payloads_do_not_alias() -> None:
    parent = record("user", "assistant", "user", "assistant")
    parent.payload["messages"][1]["tool_calls"] = [
        {"function": {"arguments": {"a": 1}}}
    ]
    first, last = ExpandChat().process_one(parent)
    first.payload["messages"][1]["tool_calls"][0]["function"]["arguments"]["a"] = 2
    first.payload["tools"][0]["name"] = "changed"
    assert last.payload == parent.payload


def test_tuple_messages() -> None:
    parent = record("user", "assistant", "user", "assistant")
    parent.payload["messages"] = tuple(parent.payload["messages"])
    assert [len(r.payload["messages"]) for r in ExpandChat().process_one(parent)] == [
        2,
        4,
    ]
