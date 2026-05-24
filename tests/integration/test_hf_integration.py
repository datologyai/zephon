# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Live integration tests for the ``hf://`` storage backend.

These exercise the backend against the real HuggingFace Datasets Server +
``huggingface.co`` resolve URLs. They are gated by ``pytest.mark.integration``
(skipped by default; ``make integration`` / ``--run-integration`` to run).

Uses ``rajpurkar/squad`` because it's small, public, and parquet-native with
a stable layout.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

# All tests here mock-free-talk to HF; if huggingface_hub isn't installed we
# can't drive ``download()`` or build resolve URLs. Skip the whole module
# rather than per-test for clarity.
pytest.importorskip("huggingface_hub")
pytest.importorskip("pyarrow")

from zephon.io.storage import HFBackend  # noqa: E402
from zephon.io.storage._hf_uri import parse_hf_uri  # noqa: E402

_SQUAD_TRAIN_URI = "hf://rajpurkar/squad/plain_text/train"


def test_dataset_from_path_streams_squad() -> None:
    """``Dataset.from_path`` lists + discovers SQuAD without predownloading."""
    from zephon.io.dataset import Dataset

    ds = Dataset.from_path("squad", _SQUAD_TRAIN_URI)
    assert ds.path == _SQUAD_TRAIN_URI
    assert ds.backend["kind"] == "parquet"
    assert len(ds.shard_index) >= 1
    assert sum(ds.shard_index.values()) > 0


def test_download_squad_shard_via_http_get(tmp_path: Path) -> None:
    """``HFBackend.download`` streams a real shard to disk via ``http_get``."""
    backend = HFBackend()
    names = backend.listdir(_SQUAD_TRAIN_URI)
    assert names, "Expected at least one parquet shard"

    src = f"{_SQUAD_TRAIN_URI}/{names[0]}"
    dst = tmp_path / names[0]
    backend.download(src, str(dst))

    expected = backend.stat(src)["size"]
    assert dst.stat().st_size == expected
    assert not (tmp_path / (names[0] + ".incomplete")).exists()


def test_hf_actually_honors_range_reads() -> None:
    """Verify HF returns HTTP 206 for byte-range GETs, not a sliced 200.

    ``HFBackend.read_range`` accepts both 206 (proper) and 200 (slice
    client-side) responses, so a passing ``len(data) == N`` check only tells
    us we got the right bytes — not that HF actually supports range reads.
    If HF's CDN ever stopped honoring Range, our backend would silently
    fall back to downloading whole shards and slicing, which would tank
    training throughput on large files. Assert the wire-level behavior
    directly.
    """
    import requests

    backend = HFBackend()
    shards = backend._list_shards(parse_hf_uri(_SQUAD_TRAIN_URI))
    assert shards, "Expected at least one shard"
    first = next(iter(shards.values()))

    resp = requests.get(
        first.url,
        headers={"Range": "bytes=0-63", "Accept-Encoding": "identity"},
        allow_redirects=True,
        timeout=60,
    )
    assert resp.status_code == 206, (
        f"HF returned {resp.status_code} for a Range request; our 200 "
        "slicing fallback would be hiding the regression"
    )
    assert resp.headers.get("Content-Length") == "64", (
        f"Expected Content-Length: 64, got {resp.headers.get('Content-Length')!r}"
    )
    assert len(resp.content) == 64


def test_minimal_pipeline_consumes_squad(tmp_path: Path) -> None:
    """Drive a tiny pipeline end-to-end against a live ``hf://`` dataset.

    Mirrors the JSONL pipeline smoke test in ``test_pipeline.py``: build a
    Dataset, wire it through StaticMixtureWorkSource + Pipeline, and prove
    we can pull a microbatch out the other end. Validates the whole stack
    — HFBackend listing, footer range reads, parquet discovery, the cache
    manager downloading shards just-in-time via ``http_get``, parquet
    random access — against real HF.
    """
    from zephon.api import Pipeline as PublicPipeline
    from zephon.io import Dataset
    from zephon.work import MixtureSpec, StaticMixtureWorkSource

    ds = Dataset.from_path("squad", _SQUAD_TRAIN_URI)
    assert ds.backend["kind"] == "parquet"
    assert sum(ds.shard_index.values()) > 0

    work_source = StaticMixtureWorkSource(
        [ds],
        mixture=MixtureSpec({ds.name: 1.0}).weights,
        chunk_size=1,
        seed=0,
        shuffle_shards=False,
    )
    pipe = (
        PublicPipeline(work_source)
        .options(io_options={"cache": {"enabled": True, "root": tmp_path / "cache"}})
        .batch(microbatch_size=2, drop_last=False)
    )

    iterator = iter(pipe)
    try:
        batch = next(iterator)
    finally:
        iterator.close()

    # microbatch_size=2 + chunk_size=1 + drop_last=False ⇒ exactly 2 rows.
    assert len(batch) == 2, f"Expected microbatch of 2 records, got {len(batch)}"
    for record in batch.records:
        payload = record.payload
        assert isinstance(payload, dict), (
            f"Expected dict payload, got {type(payload).__name__}"
        )
        for column in ("id", "context", "question", "answers"):
            assert column in payload, (
                f"Expected SQuAD column {column!r}, got {sorted(payload)}"
            )
        # Catch the "right schema, garbage values" failure mode that
        # checking only key presence would let through.
        assert isinstance(payload["id"], str) and payload["id"], (
            "Expected non-empty SQuAD id"
        )
        assert isinstance(payload["context"], str) and payload["context"], (
            "Expected non-empty SQuAD context"
        )
        assert isinstance(payload["question"], str) and payload["question"], (
            "Expected non-empty SQuAD question"
        )


def test_explicit_revision_serves_bytes_via_hf_hub_url() -> None:
    """End-to-end validation of the ``@<rev>`` rebuild path.

    Builds the rebuilt resolve URL by feeding a real ``refs/convert/parquet``
    commit SHA into the URI grammar, then proves it works by range-reading
    the first 64 bytes of a shard. This is the canary that catches us
    shipping a URL builder that passes string-equality unit tests but
    produces 404s against HF.
    """
    from huggingface_hub import HfApi

    # Discover the current commit SHA on refs/convert/parquet so the
    # rebuilt /resolve/<sha>/... URL points at a real revision.
    api = HfApi()
    info = api.dataset_info("rajpurkar/squad", revision="refs/convert/parquet")
    sha = info.sha
    assert sha, "Expected a SHA on refs/convert/parquet"

    backend = HFBackend()
    uri = f"hf://rajpurkar/squad@{sha}/plain_text/train"
    shards = backend._list_shards(parse_hf_uri(uri))
    assert shards, "Expected at least one shard"
    first = next(iter(shards.values()))

    # Rebuilt URL is the canonical resolve form pinned to <sha>, not the
    # /api/.../parquet/ form /parquet hands back.
    assert "/datasets/rajpurkar/squad/resolve/" in first.url
    assert sha in first.url

    # And the URL actually serves bytes — a range read returns 64 bytes
    # from the file at the pinned revision.
    data = backend.read_range(f"{uri}/{first.filename}", 0, length=64)
    assert len(data) == 64
