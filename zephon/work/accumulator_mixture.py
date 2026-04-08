# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Deprecated alias — use StaticMixtureWorkSource instead."""

import warnings

from zephon.work.static_mixture import StaticMixtureWorkSource


class AccumulatorMixtureWorkSource(StaticMixtureWorkSource):
    """Deprecated: use ``StaticMixtureWorkSource`` (now accumulator-based by default).

    This subclass exists only for backward compatibility and will be removed
    in a future release. It preserves the ``chunk_size=1024`` default that
    existing ``AccumulatorMixtureWorkSource`` users expect.
    """

    def __init__(self, *args, chunk_size: int = 1024, **kwargs) -> None:  # type: ignore[override]
        warnings.warn(
            "AccumulatorMixtureWorkSource is deprecated; use StaticMixtureWorkSource instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, chunk_size=chunk_size, **kwargs)
