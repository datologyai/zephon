from zephon.ops.base import BaseOp, OpContext, StageInfo
from zephon.ops.traits import OpTraits


def test_op_context_get_returns_default_and_value() -> None:
    ctx = OpContext({"a": 1})
    assert ctx.get("a") == 1
    assert ctx.get("missing") is None
    assert ctx.get("missing", 42) == 42


def test_op_context_default_stage_info_is_sentinel() -> None:
    ctx = OpContext({})
    assert ctx.stage_info.stage_index == -1
    assert ctx.stage_info.stage_name == ""
    assert ctx.stage_info.op_index == -1
    assert ctx.stage_info.collect_stats is False


class _NoopOp(BaseOp):
    def traits(self) -> OpTraits:
        return OpTraits(preserves_cursor_order=True)

    def process_many(self, elems: list) -> list:
        return elems


def test_base_op_setup_records_stage_info() -> None:
    op = _NoopOp()
    assert op.stage_info == StageInfo()

    info = StageInfo(stage_index=2, stage_name="s2", op_index=1, collect_stats=True)
    op.setup(OpContext({}, info))
    assert op.stage_info is info


def test_base_op_plan_identity_defaults_to_qualified_class_name() -> None:
    assert _NoopOp.plan_identity() == f"{__name__}._NoopOp"
