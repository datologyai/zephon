from zephon.ops.base import OpContext


def test_op_context_get_returns_default_and_value() -> None:
    ctx = OpContext({"a": 1})
    assert ctx.get("a") == 1
    assert ctx.get("missing") is None
    assert ctx.get("missing", 42) == 42
