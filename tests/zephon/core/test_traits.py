from zephon.core.traits import Buffering, OpTraits


def test_optraits_defaults() -> None:
    t = OpTraits()
    assert t.indexable is True
    assert t.parallelism == 1
    assert t.batch_shape_sensitive is False


def test_buffering_defaults_and_overrides() -> None:
    b = Buffering()
    assert b.max_batch == 32
    assert b.max_latency_ms == 5

    b2 = Buffering(max_batch=8, max_latency_ms=None)
    assert b2.max_batch == 8
    assert b2.max_latency_ms is None
