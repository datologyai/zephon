"""Unit tests for pipeline options and output notification."""

import gc
import pickle
from collections.abc import Generator, Iterator
from unittest.mock import Mock, patch
from weakref import ref

import pytest

from tests.helpers.work import FakeIndexableWorkSource, make_inmem_dataset
from zephon._internal.graph import Plan
from zephon._internal.pipeline import Pipeline
from zephon.types import SampleMeta, SampleRecord


def test_checkpoint_encoding_wraps_the_collected_state() -> None:
    pipe = object.__new__(Pipeline)
    state = {"inflight": {0: {1: {"seed": 1 << 128}}}}
    pipe._checkpoint_state = Mock(return_value=state)
    with patch(
        "zephon._internal.checkpoint.encode_checkpoint", return_value=b"blob"
    ) as encode:
        assert pipe.checkpoint() is state
        encode.assert_not_called()
        assert pipe.checkpoint(as_bytes=True) == b"blob"
        encode.assert_called_once_with(state)
    assert pipe._checkpoint_state.call_count == 2


def test_invalid_checkpoint_bytes_do_not_start_a_runtime() -> None:
    pipe = object.__new__(Pipeline)
    pipe._engine = None
    pipe._pending_restore = None
    with patch(
        "zephon._internal.checkpoint.decode_checkpoint",
        side_effect=ValueError("invalid"),
    ) as decode:
        with pytest.raises(ValueError, match="invalid"):
            pipe.restore(b"invalid")
        decode.assert_called_once_with(b"invalid")
    assert pipe._engine is None
    assert pipe._pending_restore is None


def test_close_unstarted_iterator_preserves_pending_restore() -> None:
    pipe = Pipeline(FakeIndexableWorkSource(make_inmem_dataset("tiny", [{"x": 1}])))
    checkpoint = {"pending": "restore"}
    pipe._pending_restore = checkpoint

    iterators = [iter(pipe), iter(pipe)]
    pipe.close()
    pipe.close()
    assert pipe._engine is None
    assert pipe._pending_restore is checkpoint
    for iterator in iterators:
        with pytest.raises(StopIteration):
            next(iterator)
    restored = pickle.loads(pickle.dumps(pipe))
    assert not restored._iterators
    assert restored._pending_restore == checkpoint


def test_iterator_tracking_does_not_keep_an_unstarted_iterator_alive() -> None:
    pipe = Pipeline(FakeIndexableWorkSource(make_inmem_dataset("tiny", [{"x": 1}])))
    iterator = iter(pipe)
    reference = ref(iterator)
    assert len(pipe._iterators) == 1
    del iterator
    gc.collect()
    assert reference() is None
    assert not pipe._iterators


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
@pytest.mark.parametrize("failing_resource", ["mtp", "iterator"])
def test_close_attempts_all_cleanup_before_reraising(
    error_type: type[BaseException], failing_resource: str
) -> None:
    pipe = Pipeline(FakeIndexableWorkSource(make_inmem_dataset("tiny", [{"x": 1}])))
    engine = Mock()
    pipe._engine = engine
    subprocess_runtime = Mock()
    pipe._sp = subprocess_runtime
    if failing_resource == "mtp":
        subprocess_runtime.close.side_effect = error_type("mtp cleanup failed")
    closed: list[int] = []

    def paused(index: int) -> Generator[None, None, None]:
        try:
            yield
        finally:
            closed.append(index)
            if failing_resource == "iterator":
                raise error_type("iterator cleanup failed")

    iterators = [paused(index) for index in range(2)]
    for iterator in iterators:
        next(iterator)
        pipe._iterators.add(iterator)
    with pytest.raises(error_type, match=f"{failing_resource} cleanup failed"):
        pipe.close()
    assert sorted(closed) == [0, 1]
    subprocess_runtime.close.assert_called_once_with()
    engine.close.assert_called_once_with()


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
def test_pipeline_replays_chat_prefixes(rename_messages: bool) -> None:
    from zephon.work.token_estimation import _PreTokenizeReplay

    payload = {
        "messages": [
            {"role": role, "content": str(index)}
            for index, role in enumerate(("user", "assistant") * 2)
        ]
    }
    work = FakeIndexableWorkSource(make_inmem_dataset("chat", [payload]), chunk_size=1)
    pipe = Pipeline(work).expand_chat()
    field = "messages"
    if rename_messages:
        pipe = pipe.map_transform(lambda p: {"chat": p["messages"]})
        field = "chat"
    pipe = pipe.tokenize_chat(tokenizer_id="unused", field=field)
    replay = pipe._build_pre_tokenize_replay(len(pipe._graph.nodes) - 1)
    assert isinstance(replay, _PreTokenizeReplay)
    replayed = replay.apply(payload, (0, 0, 0))
    assert replayed == [
        {field: payload["messages"][:2]},
        {field: payload["messages"]},
    ]
    assert replayed[0][field][0] is not replayed[1][field][0]
    assert not pipe.is_indexable


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


@pytest.mark.parametrize("use_monotone", [True, False])
def test_controls_do_not_notify_delivery_or_escape_the_tail(use_monotone: bool) -> None:
    # Exercise only the tail wrapper: no source, graph compilation or engine.
    pipe = object.__new__(Pipeline)
    engine = Mock()
    pipe._engine = engine
    pipe._plan = Plan(
        stages=[], explain="", indexable=False, preserves_cursor_order=use_monotone
    )
    controls = [
        SampleRecord(
            meta=SampleMeta(sample_id=(0, 0, 0), lane_id=0, chunk_id=0, tags=tags),
            payload=None,
        )
        for tags in ({"_source_exhausted": 0}, {"_flush_sentinel": True})
    ]
    dropped = SampleRecord(
        meta=SampleMeta(sample_id=(0, 0, 3), lane_id=0, chunk_id=3).with_tombstone(),
        payload=None,
    )
    record = SampleRecord(
        meta=SampleMeta(sample_id=(0, 0, 0), lane_id=0, chunk_id=0),
        payload={"value": "still buffered when the controls passed"},
    )

    def source() -> Iterator[SampleRecord]:
        yield from controls
        assert engine.mock_calls == []  # Dummy cursors cannot complete offsets.
        yield dropped
        engine.record_delivery.assert_not_called()
        engine.notify.assert_not_called()
        engine.notify_monotone.assert_not_called()
        if use_monotone:
            engine.notify_completion.assert_not_called()
        else:
            engine.notify_completion.assert_called_once_with(
                0, dropped.meta.contribution_refs()
            )
        yield record

    iterator = pipe._yield_while_notifying(source())
    try:
        assert next(iterator) is record
        engine.record_delivery.assert_called_once_with(0)
        if use_monotone:
            engine.notify_monotone.assert_called_once_with(0, 0, 1, record.meta.cursor)
            engine.notify.assert_not_called()
        else:
            engine.notify.assert_called_once_with(
                0, record.meta.contribution_refs(), record_cursor=record.meta.cursor
            )
            engine.notify_monotone.assert_not_called()
    finally:
        # Leave end-of-stream cleanup to integration tests.
        iterator.close()
