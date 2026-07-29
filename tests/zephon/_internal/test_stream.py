from zephon._internal.stream import LanePtr


def test_lane_ptr_defaults() -> None:
    ptr = LanePtr()
    assert ptr.chunk_id == -1
    assert ptr.offset == 0
