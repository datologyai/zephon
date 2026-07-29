from pathlib import Path

import pytest

from zephon._internal.io.resolvers.utils import compute_file_hash


def test_compute_file_hash_md5_and_sha256(tmp_path: Path) -> None:
    p = tmp_path / "x.bin"
    p.write_bytes(b"abc")
    assert compute_file_hash(p, "md5") == "900150983cd24fb0d6963f7d28e17f72"
    assert (
        compute_file_hash(p, "sha256")
        == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_compute_file_hash_unsupported(tmp_path: Path) -> None:
    p = tmp_path / "y.bin"
    p.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        _ = compute_file_hash(p, "madeup")
