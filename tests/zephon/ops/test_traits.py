import pytest

from zephon.ops.traits import OpTraits


def test_optraits_defaults() -> None:
    t = OpTraits(preserves_cursor_order=True)
    assert t.indexable is True
    assert t.preserves_cursor_order is True
    assert t.parallelism == 1
    assert t.batch_shape_sensitive is False


def test_optraits_requires_preserves_cursor_order() -> None:
    """`preserves_cursor_order` has no default — construction must fail."""
    with pytest.raises(TypeError):
        OpTraits()  # type: ignore[call-arg]
