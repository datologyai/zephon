from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Iterable

import pytest

# Skip vortex tests if vortex-data is not available (requires Python 3.11+)
vortex = pytest.importorskip("vortex.io", reason="vortex-data not installed")
import vortex

from zephon.api import Pipeline as PublicPipeline
from zephon.io.dataset import Dataset
from zephon.work.base import MixtureReadConfig, MixtureReadMode
from zephon.work.static_mixture import StaticMixtureWorkSource

pytestmark = pytest.mark.integration


def _create_vortex_file(path: Path, rows: list[dict[str, object]]) -> None:
    """Create a Vortex file from a list of row dictionaries."""
    vortex_array = vortex.array(rows)
    vortex.io.write(vortex_array, str(path))


def _write_vortex_dir(
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
        rows = []
        for v in chunk:
            meta = {
                "language": language,
                # Alternate license deterministically to exercise metadata shape
                "license": "MIT" if (v // 2) % 2 == 0 else "CC",
            }
            rows.append({"text": int(v), "meta": meta})
        _create_vortex_file(root / f"data_{f}.vortex", rows)


def _prepare_datasets(
    tmp: Path, total: int = 600, files: int = 6
) -> tuple[Dataset, Dataset]:
    # Even numbers → JavaScript; odd numbers → HTML
    js_vals = list(range(0, total, 2))
    html_vals = list(range(1, total, 2))

    js_dir = tmp / "js"
    html_dir = tmp / "html"
    _write_vortex_dir(js_dir, files, js_vals, "JavaScript")
    _write_vortex_dir(html_dir, files, html_vals, "HTML")

    ds_js = Dataset.from_path("JavaScript", str(js_dir), fmt="vortex")
    ds_html = Dataset.from_path("HTML", str(html_dir), fmt="vortex")
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
        payload = rec.payload
        assert isinstance(payload, dict)
        val = int(payload.get("text", 0))
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
    runner_kind: str = "threads",
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
    pipe = pipe.tokenize(
        tokenizer_id="__fallback__",
        parallelism=8,
        preserve_upstream_payload=True,
    )
    pipe = pipe.options(
        deterministic=deterministic,
        runner=runner_kind,
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


def _build_repro_param_cases() -> list[pytest.ParameterSet]:
    chunk_sizes = [32]
    worker_counts = [2, 8]
    cases: list[pytest.ParameterSet] = []

    def add_cases(
        runner_kind: str,
        modes: Iterable[MixtureReadMode],
        cache_options: Iterable[bool],
        stage_prefetch_values: Iterable[int],
        final_prefetch_values: Iterable[int],
    ) -> None:
        for (
            chunk_size,
            workers,
            mode,
            cache_enabled,
            stage_prefetch,
            final_prefetch,
        ) in product(
            chunk_sizes,
            worker_counts,
            modes,
            cache_options,
            stage_prefetch_values,
            final_prefetch_values,
        ):
            case_id = (
                f"{runner_kind}-w{workers}-mode-{mode.name}-cache-"
                f"{'on' if cache_enabled else 'off'}-stage{stage_prefetch}-final{final_prefetch}"
            )
            cases.append(
                pytest.param(
                    chunk_size,
                    workers,
                    mode,
                    cache_enabled,
                    stage_prefetch,
                    final_prefetch,
                    runner_kind,
                    id=case_id,
                )
            )

    add_cases(
        "threads",
        [MixtureReadMode.WEIGHTED_ROUND_ROBIN, MixtureReadMode.WEIGHTED_RANDOM],
        [False, True],
        [0, 4],
        [0, 16],
    )
    add_cases(
        "inline",
        [MixtureReadMode.WEIGHTED_ROUND_ROBIN],
        [True],
        [0],
        [0],
    )
    add_cases(
        "process",
        [MixtureReadMode.WEIGHTED_ROUND_ROBIN],
        [True],
        [0],
        [0],
    )
    return cases


_VORTEX_REPRO_CASES = _build_repro_param_cases()


@pytest.mark.parametrize("runner_kind", ["threads", "inline", "process"])
def test_vortex_integration_filter_js_and_html(
    tmp_path: Path, runner_kind: str
) -> None:
    ds_js, ds_html = _prepare_datasets(tmp_path, total=200, files=4)

    # Only JS dataset using public API
    work_js = StaticMixtureWorkSource([ds_js], {ds_js.name: 1.0}, chunk_size=32, seed=1)
    pipe_js = (
        PublicPipeline(work_js)
        .decode_text()
        ._delay(max_delay_ms=2.0, parallelism=8)
        .tokenize(
            tokenizer_id="__fallback__",
            parallelism=8,
            preserve_upstream_payload=True,
        )
    )
    pipe_js = pipe_js.options(
        deterministic=True,
        runner=runner_kind,
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
        .tokenize(
            tokenizer_id="__fallback__",
            parallelism=8,
            preserve_upstream_payload=True,
        )
    )
    pipe_html = pipe_html.options(
        deterministic=True,
        runner=runner_kind,
        mixture_config=MixtureReadConfig(
            mode=MixtureReadMode.WEIGHTED_ROUND_ROBIN, seed=7
        ),
    )
    out_html = _project_items(pipe_html)
    assert out_html, "No outputs for HTML"
    assert all(name == "HTML" and (val % 2 == 1) for name, val in out_html)


@pytest.mark.parametrize(
    "chunk_size, workers, mode, cache_enabled, stage_prefetch, final_prefetch, runner_kind",
    _VORTEX_REPRO_CASES,
)
def test_vortex_integration_reproducibility(
    tmp_path: Path,
    chunk_size: int,
    workers: int,
    mode: MixtureReadMode,
    cache_enabled: bool,
    stage_prefetch: int,
    final_prefetch: int,
    repro_iters: int,
    runner_kind: str,
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
            runner_kind=runner_kind,
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
            runner_kind=runner_kind,
        )
        runs_b.append(out_b)

    for i in range(1, len(runs_a)):
        assert runs_a[i] == runs_a[0]
    for i in range(1, len(runs_b)):
        assert runs_b[i] == runs_b[0]
    assert runs_a[0] == runs_b[0]


@pytest.mark.parametrize("runner_kind", ["process", "inline"])
def test_vortex_runner_threads_parity(tmp_path: Path, runner_kind: str) -> None:
    datasets = _prepare_datasets(tmp_path, total=120, files=4)
    common_kwargs = dict(
        deterministic=True,
        workers=4,
        chunk_size=32,
        mode=MixtureReadMode.WEIGHTED_ROUND_ROBIN,
        cache_enabled=False,
        cache_root=tmp_path / ".cache",
        stage_prefetch=2,
        final_prefetch=8,
    )
    out_threads = _run_pipeline(datasets, runner_kind="threads", **common_kwargs)
    out_other = _run_pipeline(datasets, runner_kind=runner_kind, **common_kwargs)
    assert out_threads == out_other
