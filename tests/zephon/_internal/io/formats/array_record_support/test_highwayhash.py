# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Independent reference vectors for the Riegeli HighwayHash implementation."""

import pytest

from zephon._internal.io.formats.array_record_support.highwayhash import highwayhash64


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        (0, 8269648958290370529),
        (1, 3230624391482529181),
        (3, 4304189606288950041),
        (15, 3747781208208142691),
        (16, 4473128012321191079),
        (17, 12035953984724789962),
        (31, 5559177453180857981),
        (32, 13204393532866551012),
        (33, 17018982045904971585),
        (63, 13736695169678185619),
        (64, 16378217524184162991),
        (65, 12975527374359423089),
        (127, 9814354802786556198),
        (128, 15500900860333712566),
    ],
)
def test_highwayhash_reference_vectors(size: int, expected: int) -> None:
    # Generated with Google's independent HHStatePortable C++ implementation,
    # using Riegeli's key and inputs bytes(range(size)).
    assert highwayhash64(bytes(range(size))) == expected


def test_long_input_with_remainder_matches_reference() -> None:
    # Independent HHStatePortable C++ result: many packets plus a 31-byte tail.
    data = bytes((37 * i + 11) & 255 for i in range(65567))
    assert highwayhash64(data) == 3832203111179982436
