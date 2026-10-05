# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Live integration tests for the ``hf://`` storage backend.

These exercise the backend against the real HuggingFace Hub, Datasets Server
and ``huggingface.co`` resolve URLs. They are gated by ``pytest.mark.integration``
(skipped by default; ``make integration`` / ``--run-integration`` to run).

``rajpurkar/squad`` is small, public and uploaded as Parquet, so it exercises
both sources: its uploaded files by default and its conversion via ``~parquet``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

# Resolving uploaded files needs ``datasets``; downloads need huggingface_hub.
datasets = pytest.importorskip("datasets")
pytest.importorskip("huggingface_hub")
pytest.importorskip("pyarrow")

import requests  # noqa: E402

from zephon._internal.io.storage import HFBackend  # noqa: E402
from zephon._internal.io.storage._hf_uri import parse_hf_uri  # noqa: E402
from zephon._internal.io.storage.hf import _DATASETS_SERVER_PARQUET_URL  # noqa: E402

_SQUAD_TRAIN_URI = "hf://rajpurkar/squad/plain_text/train"


@pytest.fixture(scope="module", autouse=True)
def _require_hub() -> None:
    """Skip the module on HF outages (5xx/unreachable); 4xx still fails loudly.

    A fixture rather than an import-time check so unit-test collection makes no HTTP call.
    """
    probes = [
        ("https://huggingface.co/api/datasets/rajpurkar/squad/refs", {}),
        (
            _DATASETS_SERVER_PARQUET_URL,
            {"dataset": "rajpurkar/squad", "config": "plain_text"},
        ),
    ]
    for url, params in probes:
        try:
            resp = requests.get(url, params=params, timeout=30)
        except requests.RequestException as exc:
            pytest.skip(f"HuggingFace unreachable: {exc}")
        if resp.status_code >= 500:
            pytest.skip(f"HuggingFace unavailable (HTTP {resp.status_code}): {url}")


def test_dataset_from_path_pins_squad_uploads() -> None:
    """``Dataset.from_path`` resolves SQuAD's uploaded Parquet and pins the commit."""
    from zephon.io.dataset import Dataset

    ds = Dataset.from_path("squad", _SQUAD_TRAIN_URI)
    assert ds.path is not None
    parts = parse_hf_uri(ds.path)
    assert parts.is_frozen and parts.source == "original"
    assert ds.backend["kind"] == "parquet"
    assert ds.shard_count() >= 1
    assert ds.total() > 0


def test_download_squad_shard_via_http_get(tmp_path: Path) -> None:
    """``HFBackend.download`` streams a real shard to disk via ``http_get``."""
    backend = HFBackend()
    root = backend.canonical_root(_SQUAD_TRAIN_URI)
    names = backend.listdir(root)
    assert names, "Expected at least one parquet shard"

    src = f"{root}/{names[0]}"
    dst = tmp_path / "shard.parquet"
    backend.download(src, str(dst))

    assert dst.stat().st_size == backend.stat(src)["size"]
    assert not (tmp_path / "shard.parquet.incomplete").exists()


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
    backend = HFBackend()
    root = backend.canonical_root(_SQUAD_TRAIN_URI)
    _, _, url = backend._resolve_file(f"{root}/{backend.listdir(root)[0]}")

    # Use the backend's retrying session, not a bare request: HF's CDN resets
    # connections, and 206/200 pass through untouched so the assert is intact.
    resp = backend._get_session().get(
        url,
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

    Validates the whole stack — resolution, footer range reads, parquet
    discovery, the cache manager downloading shards just-in-time via
    ``http_get``, parquet random access — against real HF.
    """
    from zephon import Pipeline as PublicPipeline
    from zephon.io import Dataset
    from zephon.work import MixtureSpec, StaticMixtureWorkSource

    ds = Dataset.from_path("squad", _SQUAD_TRAIN_URI)
    assert ds.backend["kind"] == "parquet"
    assert ds.total() > 0

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
        assert isinstance(payload["id"], str) and payload["id"]
        assert isinstance(payload["context"], str) and payload["context"]
        assert isinstance(payload["question"], str) and payload["question"]


def test_commit_pin_serves_uploaded_bytes() -> None:
    """``@<commit>`` reads the uploaded files at that commit."""
    backend = HFBackend()
    commit = backend.commit_for("rajpurkar/squad", "main")
    root = backend.canonical_root(f"hf://rajpurkar/squad@{commit}/plain_text/train")

    assert parse_hf_uri(root).revision == commit
    data = backend.read_range(f"{root}/{backend.listdir(root)[0]}", 0, length=64)
    assert len(data) == 64


def test_forced_conversion_serves_parquet_bytes() -> None:
    """``@~parquet`` pins HF's conversion commit and reads from it."""
    backend = HFBackend()
    root = backend.canonical_root("hf://rajpurkar/squad@~parquet/plain_text/train")
    parts = parse_hf_uri(root)

    assert parts.source == "parquet"
    assert parts.revision == backend.conversion_commit("rajpurkar/squad")
    data = backend.read_range(f"{root}/{backend.listdir(root)[0]}", 0, length=4)
    assert data == b"PAR1"


def test_fmt_parquet_selects_the_conversion_of_a_jsonl_upload() -> None:
    root = HFBackend().canonical_root(
        "hf://databricks/databricks-dolly-15k/train", "parquet"
    )
    assert parse_hf_uri(root).source == "parquet"


def test_compressed_jsonl_uploads_resolve_to_uploaded_files() -> None:
    """A large compressed-JSONL upload resolves to its uploaded files from metadata alone."""
    backend = HFBackend()
    root = backend.canonical_root("hf://mlfoundations/dclm-baseline-1.0/train")
    names = backend.listdir(root)

    assert parse_hf_uri(root).source == "original"
    assert len(names) > 20_000
    assert all(name.endswith(".jsonl.zst") for name in names)


def test_unreadable_upload_with_partial_conversion_explains_both(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """c4 uploads JSON (not JSONL) and HF converted only part of it."""
    monkeypatch.delenv("ZEPHON_HF_ALLOW_PARTIAL", raising=False)
    with pytest.raises(ValueError) as exc:
        HFBackend().canonical_root("hf://allenai/c4/en/validation")
    message = str(exc.value)
    assert "uploaded files:" in message and "json.gz" in message
    assert "parquet conversion:" in message and "partial" in message


def test_local_directory_does_not_shadow_the_hub_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A same-named local directory must not supply the remote dataset's config."""
    shadow = tmp_path / "databricks" / "databricks-dolly-15k"
    shadow.mkdir(parents=True)
    (shadow / "README.md").write_text(
        "---\nconfigs:\n- config_name: shadow\n  data_files: train.jsonl\n"
        "  default: true\n---\n",
        encoding="utf-8",
    )
    (shadow / "train.jsonl").write_text('{"a": 1}\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    major, minor = (int(part) for part in datasets.__version__.split(".")[:2])
    if (major, minor) < (4, 8):
        # Only ``datasets`` 4.8+ can skip the local directory, so it's refused.
        with pytest.raises(RuntimeError, match="would shadow"):
            HFBackend().canonical_root("hf://databricks/databricks-dolly-15k/train")
        return

    root = HFBackend().canonical_root("hf://databricks/databricks-dolly-15k/train")

    parts = parse_hf_uri(root)
    assert (parts.source, parts.config) == ("original", "default")
