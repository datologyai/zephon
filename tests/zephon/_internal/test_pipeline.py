"""Tests for pipeline option validation."""

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


def test_pipeline_replay_and_field_validation() -> None:
    from zephon.work.token_estimation import _PreTokenizeReplay

    payload = {
        "messages": [
            {"role": role, "content": "text"} for role in ("user", "assistant") * 2
        ]
    }
    work = FakeIndexableWorkSource(make_inmem_dataset("chat", [payload]), chunk_size=1)
    pipe = Pipeline(work).expand_chat().tokenize_chat(tokenizer_id="unused")
    replay = pipe._build_pre_tokenize_replay(len(pipe._graph.nodes) - 1)
    assert isinstance(replay, _PreTokenizeReplay)
    assert [len(p["messages"]) for p in replay.apply(payload)] == [2, 4]
    assert not pipe.is_indexable
    Pipeline(work).expand_chat().map_transform(
        lambda p: {"chat": p["messages"]}
    ).tokenize_chat(tokenizer_id="unused", field="chat")
