# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the extra-codec framework: canonicalization, msgpack, registry."""

import numpy as np
import pytest

from zephon._internal.io.catalog import extra_codec
from zephon._internal.io.catalog.extra_codec import (
    EncodedExtra,
    _DefaultExtraCodec,
    canonicalize,
    get_extra_codec,
    packb,
    register_extra_codec,
    unpackb,
)


def test_canonicalize_sorts_dict_keys_recursively() -> None:
    out = canonicalize({"b": 1, "a": {"d": 2, "c": 3}})
    assert list(out.keys()) == ["a", "b"]
    assert list(out["a"].keys()) == ["c", "d"]


def test_canonicalize_normalizes_tuples_and_numpy_scalars() -> None:
    out = canonicalize({"t": (1, 2), "n": np.int64(7), "f": np.float64(1.5)})
    assert out["t"] == [1, 2] and isinstance(out["t"], list)
    assert out["n"] == 7 and isinstance(out["n"], int)
    assert out["f"] == 1.5 and isinstance(out["f"], float)


def test_canonicalize_sorts_mixed_type_keys() -> None:
    # str(1)="1", str("10")="10", str(2)="2" -> non-comparable keys still order.
    out = canonicalize({2: "b", 1: "a", "10": "c"})
    assert list(out.keys()) == [1, "10", 2]


def test_packb_is_order_independent() -> None:
    assert packb({"a": 1, "b": 2}) == packb({"b": 2, "a": 1})


def test_packb_unpackb_roundtrip() -> None:
    obj = {"name": "x", "nums": [1, 2, 3], "nested": {"k": "v"}}
    assert unpackb(packb(obj)) == obj


def test_packb_tuples_decode_as_lists() -> None:
    assert unpackb(packb({"pair": (1, 2)})) == {"pair": [1, 2]}


def test_unpackb_empty_is_none() -> None:
    assert unpackb(b"") is None


def test_unpackb_allows_non_string_keys() -> None:
    assert unpackb(packb({1: "a", 2: "b"})) == {1: "a", 2: "b"}


def test_default_codec_owns_nothing() -> None:
    codec = get_extra_codec("unregistered-format")
    encoded = codec.encode([{"k": "v"}, None], np.array([1, 2], dtype=np.int64))
    assert isinstance(encoded, EncodedExtra)
    assert encoded.owned_keys == frozenset()
    assert encoded.header_blob is None
    assert encoded.int_columns == {}
    assert encoded.flags == {}
    assert codec.decode_header(None, {}) is None
    assert codec.decode(0, None, {}, 1, {}) is None


def test_register_and_get_codec(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(extra_codec, "_CODECS", {})
    sentinel = _DefaultExtraCodec()
    register_extra_codec("myfmt", sentinel)
    assert get_extra_codec("myfmt") is sentinel
    assert get_extra_codec("other") is extra_codec._DEFAULT_CODEC


def test_encoded_extra_defaults_are_independent() -> None:
    a, b = EncodedExtra(), EncodedExtra()
    assert a.owned_keys == frozenset()
    assert a.header_blob is None
    assert a.int_columns == {} and a.flags == {}
    assert a.int_columns is not b.int_columns  # per-instance default_factory
    assert a.flags is not b.flags
