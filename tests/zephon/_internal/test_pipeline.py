"""Tests for pipeline option validation."""

from typing import Any, Literal

import pytest

from tests.helpers.work import FakeIndexableWorkSource, make_inmem_dataset
from zephon._internal.pipeline import Pipeline


def test_removed_payload_memory_options_fail_before_updating_pipeline() -> None:
    pipe = Pipeline(FakeIndexableWorkSource(make_inmem_dataset("tiny", [{"x": 1}])))
    pipe.options(shm_min_item_bytes=456, shm_coalesce=True)
    for old, new, value in (
        ("shm_min_size", "shm_min_item_bytes", 123),
        ("coalesce_tensors", "shm_coalesce", False),
    ):
        with pytest.raises(TypeError, match=f"{old}.*Use '{new}'"):
            pipe.options(max_workers=99, **{old: value, new: value})
    assert pipe._options.max_workers is None
    assert pipe._options.shm_min_item_bytes == 456
    assert pipe._options.shm_coalesce


@pytest.mark.parametrize("rename_messages", [False, True])
@pytest.mark.parametrize("copy_mode", ["deep", "shallow"])
def test_pipeline_replays_chat_prefixes(
    rename_messages: bool, copy_mode: Literal["deep", "shallow"]
) -> None:
    from zephon.work.token_estimation import _PreTokenizeReplay

    payload = {
        "messages": [
            {"role": role, "content": str(index)}
            for index, role in enumerate(("user", "assistant") * 2)
        ]
    }
    work = FakeIndexableWorkSource(make_inmem_dataset("chat", [payload]), chunk_size=1)
    pipe = Pipeline(work).expand_chat(copy_mode=copy_mode)
    field = "messages"
    if rename_messages:
        pipe = pipe.map_transform(lambda p: {"chat": p["messages"]})
        field = "chat"
    pipe = pipe.tokenize_chat(tokenizer_id="unused", field=field)
    replay = pipe._build_pre_tokenize_replay(len(pipe._graph.nodes) - 1)
    assert isinstance(replay, _PreTokenizeReplay)
    replayed = replay.apply(payload)
    assert replayed == [
        {field: payload["messages"][:2]},
        {field: payload["messages"]},
    ]
    assert (replayed[0][field][0] is replayed[1][field][0]) == (copy_mode == "shallow")
    assert not pipe.is_indexable


@pytest.mark.parametrize("copy_mode", ["invalid", "", None])
def test_expand_chat_rejects_invalid_copy_mode(copy_mode: Any) -> None:
    pipe = Pipeline(FakeIndexableWorkSource(make_inmem_dataset("chat", [])))
    tail = pipe._tail
    with pytest.raises(ValueError, match="copy_mode must be 'deep' or 'shallow'"):
        pipe.expand_chat(copy_mode=copy_mode)
    assert pipe._tail is tail


@pytest.fixture
def chat_scope_tokenizer():
    from tests.helpers.chat import make_fast_tokenizer

    return make_fast_tokenizer()


def test_preflight_reports_failure_without_disabling_runtime_checks(
    chat_scope_tokenizer,
) -> None:
    from tests.helpers.chat import CONVERSATION, TAGGED_TEMPLATE
    from tests.helpers.work import FakeIndexableWorkSource, make_inmem_dataset
    from zephon import Pipeline

    work = FakeIndexableWorkSource(
        make_inmem_dataset("chat", [{"messages": CONVERSATION}]), chunk_size=1
    )
    pipe = Pipeline(work).tokenize_chat(
        chat_scope_tokenizer,
        chat_template=TAGGED_TEMPLATE + "{% generation %}eos{% endgeneration %}",
        loss_scope="after_last_user",
    )
    assert not pipe.preflight_tokenizers(strict=False).ok
    assert not pipe._tail.op._tokenizer_instantiated
    with pytest.raises(ValueError, match="no enclosing loop"):
        pipe._tail.op._ids_and_mask({"messages": CONVERSATION}, "fixture")


def test_deprecation_warning_points_to_public_caller(chat_scope_tokenizer) -> None:
    from tests.helpers.chat import CONVERSATION
    from tests.helpers.work import FakeIndexableWorkSource, make_inmem_dataset
    from zephon import Pipeline

    work = FakeIndexableWorkSource(
        make_inmem_dataset("chat", [{"messages": CONVERSATION}]), chunk_size=1
    )
    with pytest.warns(FutureWarning) as caught:
        Pipeline(work).tokenize_chat(chat_scope_tokenizer, loss_on_last_turn_only=True)
    assert caught[0].filename == __file__
