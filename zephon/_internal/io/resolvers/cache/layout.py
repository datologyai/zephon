"""Names and validation shared by the on-disk cache layouts."""

from collections.abc import Iterable
from pathlib import Path

PARQUET_RG_CACHE_SUBDIR = ".parquet-rg-cache"


def validate_raw_cache_dataset_names(root: Path, names: Iterable[str]) -> None:
    """Reject dataset paths that escape the raw cache or enter its RG subtree."""
    canonical_root = root.expanduser().resolve()
    reserved = (canonical_root / PARQUET_RG_CACHE_SUBDIR).resolve()
    for name in names:
        name_path = Path(name)
        candidate = (canonical_root / name_path).resolve()
        if (
            name_path.is_absolute()
            or candidate == canonical_root
            or canonical_root not in candidate.parents
            or candidate == reserved
            or reserved in candidate.parents
        ):
            raise ValueError(
                f"Dataset name {name!r} is unsafe inside cache root {canonical_root}"
            )


__all__ = ["PARQUET_RG_CACHE_SUBDIR", "validate_raw_cache_dataset_names"]
