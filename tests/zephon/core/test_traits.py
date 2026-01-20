from zephon.core.traits import OpTraits


def test_optraits_defaults() -> None:
    t = OpTraits()
    assert t.indexable is True
    assert t.preserves_cursor_order is None
    assert t.parallelism == 1
    assert t.batch_shape_sensitive is False
