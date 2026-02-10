import json
import warnings
from pathlib import Path

from zephon.api import Pipeline as PublicPipeline
from zephon.io import Dataset
from zephon.work import MixtureSpec, StaticMixtureWorkSource


def test_pipeline_with_cache(tmp_path: Path) -> None:
    # Build a small JSONL dataset
    shard0 = tmp_path / "shard0.jsonl"
    shard1 = tmp_path / "shard1.jsonl"
    shard0.write_text(
        "\n".join(json.dumps({"text": f"sample {i}", "value": i}) for i in range(3)),
        encoding="utf-8",
    )
    shard1.write_text(
        "\n".join(json.dumps({"text": f"sample {i}", "value": i}) for i in range(3, 5)),
        encoding="utf-8",
    )
    jsonl_dataset = Dataset.from_path("demo", str(tmp_path))

    cache_root = tmp_path / "cache"
    work_source = StaticMixtureWorkSource(
        [jsonl_dataset],
        mixture=MixtureSpec({jsonl_dataset.name: 1.0}).weights,
        chunk_size=1,
        seed=11,
        shuffle_shards=False,
    )
    pipe = (
        PublicPipeline(work_source)
        .decode_text()
        .options(io_options={"cache": {"enabled": True, "root": cache_root}})
        .batch(microbatch_size=2, drop_last=False)
    )
    iterator = iter(pipe)
    try:
        next(iterator)
    finally:
        iterator.close()
    assert cache_root.exists()


def test_unknown_options_warn(tmp_path: Path) -> None:
    """Unknown keys passed to .options() should emit a warning, not be silently ignored."""
    shard = tmp_path / "shard.jsonl"
    shard.write_text(json.dumps({"text": "hello"}), encoding="utf-8")
    ds = Dataset.from_path("demo", str(tmp_path))
    ws = StaticMixtureWorkSource(
        [ds],
        mixture=MixtureSpec({ds.name: 1.0}).weights,
        chunk_size=1,
        seed=0,
    )
    pipe = PublicPipeline(ws)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipe.options(no_such_option=42, deterministic=True)

    unknown_warnings = [w for w in caught if "no_such_option" in str(w.message)]
    assert len(unknown_warnings) == 1, (
        f"Expected exactly 1 warning for unknown option, got {len(unknown_warnings)}"
    )
    # deterministic is valid and should NOT trigger a warning
    deterministic_warnings = [w for w in caught if "deterministic" in str(w.message)]
    assert len(deterministic_warnings) == 0, "Valid option should not trigger a warning"
