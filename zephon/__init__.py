# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Zephon: A scalable and flexible multimodal data loader."""

from zephon._version import __version__  # noqa: F401
from zephon.api import Pipeline

__all__ = [
    "Pipeline",
    "__version__",
]
