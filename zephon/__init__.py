# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Zephon: A scalable and flexible multimodal data loader."""

# Import semaphore tracker early to install hooks before any multiprocessing.
# This module auto-activates if ZEPHON_SEMAPHORE_LEAK_DEBUG=1
import zephon.debug.semaphore_tracker  # noqa: F401  # type: ignore[reportUnusedImport]  # isort: skip

from zephon._version import __version__  # noqa: F401
from zephon.api import Pipeline
from zephon.utils.litdata_compat import install_litdata_patch
from zephon.utils.torchdata_compat import install_torchdata_patch

install_torchdata_patch()  # noop if users never install torchdata
install_litdata_patch()  # noop if users never install litdata

__all__ = [
    "Pipeline",
    "__version__",
]
