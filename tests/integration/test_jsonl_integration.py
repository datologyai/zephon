import json
from pathlib import Path
from typing import Iterable

import pytest

from zephon.api import Pipeline as PublicPipeline
from zephon.io.dataset import Dataset
from zephon.work.base import MixtureReadConfig, MixtureReadMode
from zephon.work.static_mixture import StaticMixtureWorkSource

pytestmark = pytest.mark.integration


def _write_jsonl_dir(
    root: Path, shard_files: int, values: Iterable[int], language: str
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    vals = list(values)
    # Split values across shard_files as evenly as possible
    per = len(vals) // shard_files
    extra = len(vals) % shard_files
    offset = 0
    for f in range(shard_files):
        take = per + (1 if f < extra else 0)
        chunk = vals[offset : offset + take]
        offset += take
        lines = []
        for v in chunk:
            meta = {
                "language": language,
                # Alternate license deterministically to exercise metadata shape
                "license": "MIT" if (v // 2) % 2 == 0 else "CC",
            }
            lines.append(json.dumps({"text": int(v), "meta": meta}))
        (root / f"data_{f}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _prepare_datasets(
    tmp: Path, total: int = 600, files: int = 6
) -> tuple[Dataset, Dataset]:
    # Even numbers → JavaScript; odd numbers → HTML
    js_vals = list(range(0, total, 2))
    html_vals = list(range(1, total, 2))

    js_dir = tmp / "js"
    html_dir = tmp / "html"
    _write_jsonl_dir(js_dir, files, js_vals, "JavaScript")
    _write_jsonl_dir(html_dir, files, html_vals, "HTML")

    ds_js = Dataset.from_path("JavaScript", str(js_dir), fmt="jsonl")
    ds_html = Dataset.from_path("HTML", str(html_dir), fmt="jsonl")
    return ds_js, ds_html


def _project_items(pipe: PublicPipeline) -> list[tuple[str, int]]:
    from zephon.core.constants import SampleRecord

    id_to_name: dict[int, str] = {
        i: ds.name for i, ds in pipe.ws.datasets_by_id.items()
    }
    out: list[tuple[str, int]] = []
    for rec in pipe:
        assert isinstance(rec, SampleRecord)
        ds_id = int(rec.meta.sample_id[0])
        name = id_to_name.get(ds_id, str(ds_id))
        val = int(rec.payload.get("text", 0))
        out.append((name, val))
    return out


def _run_pipeline(
    datasets: tuple[Dataset, Dataset],
    *,
    deterministic: bool,
    workers: int,
    chunk_size: int,
    mode: MixtureReadMode,
    cache_enabled: bool = False,
    cache_root: Path | None = None,
    stage_prefetch: int = 0,
    final_prefetch: int = 0,
) -> list[tuple[str, int]]:
    ds_js, ds_html = datasets
    mixture = {ds_js.name: 0.6, ds_html.name: 0.4}
    work = StaticMixtureWorkSource(
        [ds_js, ds_html],
        mixture,
        chunk_size=chunk_size,
        seed=123,
        shuffle_shards=True,
        shuffle_within_shard=True,
    )

    pipe = PublicPipeline(work)
    pipe = pipe.decode_text()
    pipe = pipe._delay(max_delay_ms=2.0, parallelism=8)
    pipe = pipe.tokenize(tokenizer_id="__fallback__", parallelism=8)
    pipe = pipe.options(
        deterministic=deterministic,
        max_workers=workers,
        default_stage_prefetch=stage_prefetch,
        prefetch_batches=final_prefetch,
        mixture_config=MixtureReadConfig(mode=mode, seed=999),
        io_options={
            "cache": {
                "enabled": cache_enabled,
                "root": str(cache_root) if cache_root else None,
            }
        },
    )
    return _project_items(pipe)


def test_jsonl_integration_filter_js_and_html(tmp_path: Path) -> None:
    ds_js, ds_html = _prepare_datasets(tmp_path, total=200, files=4)

    # Only JS dataset using public API
    work_js = StaticMixtureWorkSource([ds_js], {ds_js.name: 1.0}, chunk_size=32, seed=1)
    pipe_js = (
        PublicPipeline(work_js)
        .decode_text()
        ._delay(max_delay_ms=2.0, parallelism=8)
        .tokenize(tokenizer_id="__fallback__", parallelism=8)
    )
    pipe_js = pipe_js.options(
        deterministic=True,
        mixture_config=MixtureReadConfig(
            mode=MixtureReadMode.WEIGHTED_ROUND_ROBIN, seed=7
        ),
    )
    out_js = _project_items(pipe_js)
    assert out_js, "No outputs for JS"
    assert all(name == "JavaScript" and (val % 2 == 0) for name, val in out_js)

    # Only HTML dataset using public API
    work_html = StaticMixtureWorkSource(
        [ds_html], {ds_html.name: 1.0}, chunk_size=32, seed=2
    )
    pipe_html = (
        PublicPipeline(work_html)
        .decode_text()
        ._delay(max_delay_ms=2.0, parallelism=8)
        .tokenize(tokenizer_id="__fallback__", parallelism=8)
    )
    pipe_html = pipe_html.options(
        deterministic=True,
        mixture_config=MixtureReadConfig(
            mode=MixtureReadMode.WEIGHTED_ROUND_ROBIN, seed=7
        ),
    )
    out_html = _project_items(pipe_html)
    assert out_html, "No outputs for HTML"
    assert all(name == "HTML" and (val % 2 == 1) for name, val in out_html)


@pytest.mark.parametrize("chunk_size", [32])
@pytest.mark.parametrize("workers", [2, 8])
@pytest.mark.parametrize(
    "mode", [MixtureReadMode.WEIGHTED_ROUND_ROBIN, MixtureReadMode.WEIGHTED_RANDOM]
)
@pytest.mark.parametrize("cache_enabled", [False, True])
@pytest.mark.parametrize("stage_prefetch", [0, 4])
@pytest.mark.parametrize("final_prefetch", [0, 16])
def test_jsonl_integration_reproducibility(
    tmp_path: Path,
    chunk_size: int,
    workers: int,
    mode: MixtureReadMode,
    cache_enabled: bool,
    stage_prefetch: int,
    final_prefetch: int,
    repro_iters: int,
) -> None:
    datasets = _prepare_datasets(tmp_path, total=300, files=3)

    # Run N times with base workers and N times with doubled workers
    runs_a: list[list[tuple[str, int]]] = []
    runs_b: list[list[tuple[str, int]]] = []

    for _ in range(repro_iters):
        out_a = _run_pipeline(
            datasets,
            deterministic=True,
            workers=workers,
            chunk_size=chunk_size,
            mode=mode,
            cache_enabled=cache_enabled,
            cache_root=tmp_path / ".cache",
            stage_prefetch=stage_prefetch,
            final_prefetch=final_prefetch,
        )
        runs_a.append(out_a)

    for _ in range(repro_iters):
        out_b = _run_pipeline(
            datasets,
            deterministic=True,
            workers=workers * 2,
            chunk_size=chunk_size,
            mode=mode,
            cache_enabled=cache_enabled,
            cache_root=tmp_path / ".cache",
            stage_prefetch=stage_prefetch,
            final_prefetch=final_prefetch,
        )
        runs_b.append(out_b)

    # Check internal consistency
    for i in range(1, len(runs_a)):
        assert runs_a[i] == runs_a[0]
    for i in range(1, len(runs_b)):
        assert runs_b[i] == runs_b[0]
    # And equivalence across worker counts
    assert runs_a[0] == runs_b[0]
