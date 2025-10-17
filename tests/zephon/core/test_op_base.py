from zephon.core.op_base import DefaultFinalize, OpContext


def test_op_context_get_returns_default_and_value() -> None:
    ctx = OpContext({"a": 1})
    assert ctx.get("a") == 1
    assert ctx.get("missing") is None
    assert ctx.get("missing", 42) == 42


class _Sink(DefaultFinalize[int]):
    pass


def test_default_finalize_returns_empty_list() -> None:
    sink = _Sink()
    assert sink.finalize() == []
