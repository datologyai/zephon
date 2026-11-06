# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Read ArrayRecord v1 counts using only postscript and footer ranges.

Layout: google/array_record cpp/layout.proto. Both metadata chunks use
Riegeli's simple encoding (google/riegeli doc/riegeli_records_file_format.md).
No sample data or native reader is needed. Unknown encodings fail explicitly.
"""

from collections.abc import Iterator

from zephon._internal.io.formats.array_record_support.highwayhash import highwayhash64
from zephon._internal.io.storage import StorageBackend

_BLOCK = 1 << 16
_MAGIC = 0x71930E704FDAE05E


def _varint(data: bytes, pos: int) -> tuple[int, int]:
    value = 0
    for shift in range(0, 70, 7):
        if pos >= len(data):
            raise ValueError("Truncated ArrayRecord metadata varint")
        byte = data[pos]
        pos += 1
        if shift == 63 and byte > 1:
            break
        value |= (byte & 127) << shift
        if byte < 128:
            return value, pos
    raise ValueError("Invalid ArrayRecord metadata varint")


def _fields(data: bytes) -> Iterator[tuple[int, int | bytes]]:
    pos = 0
    while pos < len(data):
        tag, pos = _varint(data, pos)
        number, wire = tag >> 3, tag & 7
        if number == 0:
            raise ValueError("Invalid ArrayRecord metadata field")
        if wire == 0:
            value, pos = _varint(data, pos)
            yield number, value
            continue
        if wire == 2:
            size, pos = _varint(data, pos)
        elif wire in (1, 5):
            size = 8 if wire == 1 else 4
        else:
            raise ValueError(f"Unsupported ArrayRecord protobuf wire type {wire}")
        end = pos + size
        if end > len(data):
            raise ValueError("Truncated ArrayRecord metadata field")
        if wire == 2:
            yield number, data[pos:end]
        pos = end


def _unblock(data: bytes, offset: int, length: int) -> bytes:
    """Remove intervening 24-byte block headers, checking their hashes."""
    parts = []
    pos = 0
    while length:
        absolute = offset + pos
        if absolute % _BLOCK == 0:
            header = data[pos : pos + 24]
            if len(header) != 24 or int.from_bytes(
                header[:8], "little"
            ) != highwayhash64(header[8:]):
                raise ValueError("Invalid Riegeli block header checksum")
            pos += 24
            absolute += 24
        elif absolute % _BLOCK <= 24:
            raise ValueError("Invalid Riegeli chunk offset")
        take = min(length, _BLOCK - absolute % _BLOCK)
        part = data[pos : pos + take]
        if len(part) != take:
            raise ValueError("Truncated Riegeli metadata chunk")
        parts.append(part)
        length -= take
        pos += take
    return b"".join(parts)


def _physical_end(offset: int, length: int) -> int:
    # The first chunk byte can lie at a block boundary (before its header).
    while length:
        if offset % _BLOCK == 0:
            offset += 24
        take = min(length, _BLOCK - offset % _BLOCK)
        offset += take
        length -= take
    return offset


def _decompress(data: bytes, codec: int, max_size: int) -> bytes:
    if codec == 0:
        if len(data) > max_size:
            raise ValueError("Invalid Riegeli decoded size")
        return data
    expected, pos = _varint(data, 0)
    if expected > max_size:
        raise ValueError("Invalid Riegeli decompressed size")
    try:
        import cramjam
    except ImportError as exc:
        raise ImportError(
            'ArrayRecord metadata compression requires pip install "zephon[array_record]"'
        ) from exc
    codecs = {
        ord("z"): cramjam.zstd.decompress_into,
        ord("b"): cramjam.brotli.decompress_into,
        ord("s"): cramjam.snappy.decompress_raw_into,
    }
    if codec not in codecs:
        raise ValueError(f"Unsupported Riegeli compression type {codec}")
    output = bytearray(expected)
    try:
        actual = codecs[codec](data[pos:], output)
    except Exception as exc:
        raise ValueError("Invalid compressed ArrayRecord metadata") from exc
    if actual != expected:
        raise ValueError("Incorrect ArrayRecord decompressed size")
    return bytes(output)


def _first_record(chunk: bytes) -> tuple[bytes, int]:
    """Decode only the first record, without constructing the chunk index."""
    header, data = chunk[:40], chunk[40:]
    num_records = int.from_bytes(header[25:32], "little")
    decoded_size = int.from_bytes(header[32:40], "little")
    if header[24] != ord("r") or not num_records or not data:
        raise ValueError("Unsupported or empty ArrayRecord metadata chunk")
    if int.from_bytes(header[16:24], "little") != highwayhash64(data):
        raise ValueError("Invalid Riegeli metadata data checksum")
    codec = data[0]
    sizes_length, pos = _varint(data, 1)
    end = pos + sizes_length
    if end > len(data):
        raise ValueError("Truncated Riegeli record sizes")
    sizes = _decompress(data[pos:end], codec, num_records * 10)
    first_size, _ = _varint(sizes, 0)
    values = _decompress(data[end:], codec, decoded_size)
    if len(values) != decoded_size or first_size > len(values):
        raise ValueError("Invalid Riegeli metadata record size")
    return values[:first_size], num_records


def _chunk(
    path: str,
    storage: StorageBackend,
    offset: int,
    limit: int,
) -> tuple[bytes, int]:
    def read(start: int, size: int) -> bytes:
        data = bytes(storage.read_range(path, start, length=size))
        if len(data) != size:
            raise ValueError(f"Incomplete ArrayRecord metadata range at {start}")
        return data

    if not 0 <= offset < limit:
        raise ValueError("Invalid ArrayRecord footer offset")
    prefix = read(offset, min(_BLOCK, limit - offset))
    header = _unblock(prefix, offset, 40)
    if int.from_bytes(header[:8], "little") != highwayhash64(header[8:]):
        raise ValueError("Invalid Riegeli chunk header checksum")
    length = 40 + int.from_bytes(header[8:16], "little")
    if length > limit - offset:
        raise ValueError("ArrayRecord metadata chunk exceeds its bounds")
    end = _physical_end(offset, length)
    if end > limit:
        raise ValueError("ArrayRecord metadata chunk exceeds its bounds")
    if end > offset + len(prefix):
        prefix += read(offset + len(prefix), end - offset - len(prefix))
    return _first_record(_unblock(prefix, offset, length))


def read_row_count(path: str, storage: StorageBackend, size: int) -> int:
    """Return the v1 record count, fetching metadata through the backend."""
    if size < _BLOCK or size % _BLOCK:
        raise ValueError("Invalid ArrayRecord file size")
    postscript_offset = size - _BLOCK
    postscript, _ = _chunk(path, storage, postscript_offset, size)
    fields = dict(_fields(postscript))
    footer_offset = fields.get(1)
    if fields.get(2) != _MAGIC or type(footer_offset) is not int:
        raise ValueError("Invalid ArrayRecord postscript")
    footer, footer_records = _chunk(path, storage, footer_offset, postscript_offset)
    metadata = dict(_fields(footer)).get(1)
    if not isinstance(metadata, bytes):
        raise ValueError("Missing ArrayRecord footer metadata")
    fields = dict(_fields(metadata))
    if fields.get(1) != 1:
        raise ValueError(f"Unsupported ArrayRecord version {fields.get(1)!r}")
    chunks, count = fields.get(2), fields.get(3)
    if (
        type(chunks) is not int
        or chunks != footer_records - 1
        or type(count) is not int
        or not 0 <= count < 1 << 63
        or count < chunks
        or (chunks == 0 and count != 0)
    ):
        raise ValueError("Invalid ArrayRecord footer counts")
    return count
