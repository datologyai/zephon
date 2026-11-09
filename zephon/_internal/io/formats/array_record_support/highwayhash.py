# Copyright 2015-2017 Google Inc.
# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Portable HighwayHash64 for Riegeli metadata integrity checks.

Follows Google's Apache-2.0 HighwayHash portable implementation:
https://github.com/google/highwayhash/blob/master/highwayhash/hh_portable.h
This runs only during discovery, never on sample payloads.
"""

import struct
from operator import itemgetter

_MASK = (1 << 64) - 1
_KEY = (0x2F696C6567656952, 0x0A7364726F636572) * 2
_INIT0 = (
    0xDBE6D5D5FE4CCE2F,
    0xA4093822299F31D0,
    0x13198A2E03707344,
    0x243F6A8885A308D3,
)
_INIT1 = (
    0x3BD39E10CB0EF593,
    0xC0ACF169B5F18A8C,
    0xBE5466CF34E90C6C,
    0x452821E638D01377,
)
# Both eight-byte permutations from ZipperMergeAndAdd, as low/high lanes.
# itemgetter performs the byte selection in C without a Python generator loop.
_ZIPPER = itemgetter(3, 12, 2, 5, 14, 1, 15, 0, 11, 4, 10, 13, 9, 6, 8, 7)


def _rotate(value: int) -> int:
    return ((value >> 32) | (value << 32)) & _MASK


def _zipper(a: int, b: int) -> tuple[int, int]:
    low, high = struct.unpack("<2Q", bytes(_ZIPPER(struct.pack("<2Q", b, a))))
    return low, high


def highwayhash64(data: bytes) -> int:
    """Hash bytes with the fixed Riegeli/records HighwayHash key."""
    mul0: list[int] = list(_INIT0)
    mul1: list[int] = list(_INIT1)
    v0 = [a ^ b for a, b in zip(_INIT0, _KEY)]
    v1 = [a ^ _rotate(b) for a, b in zip(_INIT1, _KEY)]

    def update(packet: tuple[int, ...]) -> None:
        for i in range(4):
            v1[i] = (v1[i] + packet[i] + mul0[i]) & _MASK
            mul0[i] ^= (v1[i] & 0xFFFFFFFF) * (v0[i] >> 32)
            v0[i] = (v0[i] + mul1[i]) & _MASK
            mul1[i] ^= (v0[i] & 0xFFFFFFFF) * (v1[i] >> 32)
        for source, target in ((v1, v0), (v0, v1)):
            for i in (0, 2):
                low, high = _zipper(source[i + 1], source[i])
                target[i] = (target[i] + low) & _MASK
                target[i + 1] = (target[i + 1] + high) & _MASK

    complete = len(data) & ~31
    for packet in struct.iter_unpack("<4Q", memoryview(data)[:complete]):
        update(packet)
    remainder = data[complete:]
    n = len(remainder)
    if n:
        for i in range(4):
            v0[i] = (v0[i] + (n << 32) + n) & _MASK
            low, high = v1[i] & 0xFFFFFFFF, v1[i] >> 32
            low = ((low << n) | (low >> (32 - n))) & 0xFFFFFFFF
            high = ((high << n) | (high >> (32 - n))) & 0xFFFFFFFF
            v1[i] = low | (high << 32)
        packet = bytearray(32)
        aligned = n & ~3
        packet[:aligned] = remainder[:aligned]
        if n & 16:
            packet[28:32] = remainder[-4:]
        elif n & 3:
            tail = remainder[aligned:]
            packet[16:19] = bytes((tail[0], tail[len(tail) // 2], tail[-1]))
        update(struct.unpack("<4Q", packet))
    for _ in range(4):
        update(tuple(_rotate(v0[i]) for i in (2, 3, 0, 1)))
    return (v0[0] + v1[0] + mul0[0] + mul1[0]) & _MASK
