# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Zephon: A scalable and flexible multimodal data loader."""

from zephon._version import __version__  # noqa: F401
from zephon.api import Pipeline
from zephon.utils.torchdata_compat import install_torchdata_patch

install_torchdata_patch()  # noop if users never install torchdata

__all__ = [
    "Pipeline",
    "__version__",
]
