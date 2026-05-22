# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the AggregationCodec."""

import pytest

from zephon.core.checkpoint import AggregationCodec

SAMPLE_STATE = {
    "version": 1,
    "world": {"canonical_replicas": 4, "world_size": 2},
    "inflight": {
        "0": {
            "5": {
                "version": 1,
                "seed": 42,
                "components": [["ds_a", [[0, 0, i] for i in range(100)]]],
                "component_order": ["ds_a"],
                "total_samples": 100,
            }
        }
    },
    "progress": {"0": {"chunk_id": 5, "offset": 50}},
    "lane_next_cid": {"0": 6},
    "lane_ws_state": {"0": {"lane_id": 0}},
    "last_round_id": "12345",
    "checkpoint_reload_count": 0,
}


@pytest.mark.parametrize(
    "serializer,compressor",
    [
        ("json", "none"),
        ("json", "zstd"),
        ("msgpack", "none"),
        ("msgpack", "zstd"),
    ],
)
def test_codec_roundtrip(serializer, compressor):
    codec = AggregationCodec(serializer=serializer, compressor=compressor)
    encoded = codec.encode(SAMPLE_STATE)
    decoded = codec.decode(encoded)
    assert decoded == SAMPLE_STATE


def test_default_codec_is_msgpack_zstd():
    codec = AggregationCodec()
    encoded = codec.encode({"key": "value"})
    decoded = codec.decode(encoded)
    assert decoded == {"key": "value"}


def test_zstd_compression_reduces_size():
    codec_compressed = AggregationCodec(serializer="msgpack", compressor="zstd")
    codec_plain = AggregationCodec(serializer="msgpack", compressor="none")
    compressed = codec_compressed.encode(SAMPLE_STATE)
    plain = codec_plain.encode(SAMPLE_STATE)
    assert len(compressed) < len(plain)


def test_invalid_serializer_raises():
    with pytest.raises(ValueError, match="Unknown serializer"):
        AggregationCodec(serializer="xml")


def test_invalid_compressor_raises():
    with pytest.raises(ValueError, match="Unknown compressor"):
        AggregationCodec(compressor="lz4")


def test_msgpack_preserves_int_keys():
    """msgpack preserves int keys natively (unlike JSON which stringifies)."""
    codec = AggregationCodec(serializer="msgpack", compressor="none")
    data = {0: "a", 1: "b"}
    decoded = codec.decode(codec.encode(data))
    assert decoded == {0: "a", 1: "b"}


def test_decode_rejects_truncated_payload():
    """A truncated zstd payload must surface as a decode error, not silent corruption."""
    codec = AggregationCodec(serializer="msgpack", compressor="zstd")
    encoded = codec.encode(SAMPLE_STATE)
    truncated = encoded[: len(encoded) // 2]
    with pytest.raises(Exception):
        codec.decode(truncated)


def test_decode_rejects_codec_mismatch():
    """A payload produced with one codec cannot be silently read with another."""
    msgpack_zstd = AggregationCodec(serializer="msgpack", compressor="zstd")
    json_plain = AggregationCodec(serializer="json", compressor="none")
    encoded = msgpack_zstd.encode({"key": "value"})
    with pytest.raises(Exception):
        json_plain.decode(encoded)


def test_decode_rejects_random_garbage():
    """Random bytes must not decode to a dict — guards against silent corruption."""
    codec = AggregationCodec()
    with pytest.raises(Exception):
        codec.decode(b"\x00\x01\x02\x03not-a-real-payload")


# ---------------------------------------------------------------------------
# Arbitrary-precision integer handling
#
# msgpack's wire format tops out at int64 / uint64. numpy's PCG64
# ``bit_generator.state`` exposes 128-bit Python ints, which land in the
# cursor checkpoint via ``block_rng_snapshot["rng_state"]`` whenever
# ``shuffle_block_size`` is set. Without explicit handling, ``packb``
# raises ``OverflowError`` mid-pack on real checkpoints — see the codec's
# ``_wrap_bigints`` / ``_bigint_ext_hook``.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        (1 << 64),  # smallest int wider than uint64
        -(1 << 63) - 1,  # smallest int below int64
        (1 << 128) - 1,  # full 128-bit (PCG64 ``state``/``inc`` width)
        -(1 << 200),  # well past either boundary
    ],
)
def test_msgpack_roundtrips_arbitrary_precision_ints(value):
    codec = AggregationCodec(serializer="msgpack", compressor="none")
    payload = {"deep": [{"nested": {"value": value}}]}
    assert codec.decode(codec.encode(payload)) == payload


@pytest.mark.parametrize("value", [(1 << 64) - 1, -(1 << 63), 0, 1, -1])
def test_msgpack_native_range_unchanged(value):
    """Ints inside [-2^63, 2^64-1] must encode without the ext-type wrapper."""
    codec = AggregationCodec(serializer="msgpack", compressor="none")
    assert codec.decode(codec.encode({"v": value})) == {"v": value}


def test_msgpack_roundtrips_numpy_pcg64_state():
    """The real-world failure mode: PCG64 state from ``np.random.default_rng``.

    Mirrors how ``_DatasetCursor.checkpoint_state()`` embeds the state into
    ``block_rng_snapshot["rng_state"]`` in the engine local state dict.
    """
    np = pytest.importorskip("numpy")
    state = np.random.default_rng(42).bit_generator.state
    local = {
        "lane_ws_state": {
            0: {
                "cursor_states": {
                    "ds_a": {
                        "version": 1,
                        "position": 100,
                        "epoch": 0,
                        "block_rng_snapshot": {
                            "rng_state": state,
                            "block_count": 7,
                        },
                    }
                }
            }
        },
    }
    codec = AggregationCodec()  # default msgpack+zstd, what production uses
    decoded = codec.decode(codec.encode(local))
    assert decoded == local

    # numpy must be able to re-seed from the decoded state.
    rng = np.random.default_rng(0)
    rng.bit_generator.state = decoded["lane_ws_state"][0]["cursor_states"]["ds_a"][
        "block_rng_snapshot"
    ]["rng_state"]
    assert rng.bit_generator.state == state


def test_msgpack_preserves_bool_through_bigint_walk():
    """``bool`` is an ``int`` subclass — the wrapper must not coerce it."""
    codec = AggregationCodec(serializer="msgpack", compressor="none")
    payload = {"t": True, "f": False, "big": 1 << 100}
    decoded = codec.decode(codec.encode(payload))
    assert decoded == payload
    assert isinstance(decoded["t"], bool) and isinstance(decoded["f"], bool)
