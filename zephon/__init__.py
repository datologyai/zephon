# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Zephon: A scalable and flexible multimodal data loader.

Quickstart surface. The rest of the public API lives in semantic namespaces:
``zephon.ops`` (op authoring kit), ``zephon.work`` (work sources),
``zephon.io`` (datasets + store options), ``zephon.types`` (data model),
``zephon.options`` (RuntimeOptions), ``zephon.build_index`` (index-build CLI),
``zephon.validation``, ``zephon.observability``, ``zephon.debug``. Implementation
machinery lives under ``zephon._internal`` and is not public API.
"""

# Import semaphore tracker early to install hooks before any multiprocessing.
# This module auto-activates if ZEPHON_SEMAPHORE_LEAK_DEBUG=1
import zephon.debug.semaphore_tracker  # noqa: F401  # type: ignore[reportUnusedImport]  # isort: skip

from zephon._internal.utils.litdata_compat import install_litdata_patch
from zephon._internal.utils.torchdata_compat import install_torchdata_patch
from zephon._version import __version__  # noqa: F401
from zephon.io import Dataset, InMemoryShard
from zephon.pipeline import Pipeline
from zephon.types import SampleBatch, SampleMeta, SampleRecord
from zephon.work import MixtureSpec, StaticMixtureWorkSource, WorkSource

install_torchdata_patch()  # noop if users never install torchdata
install_litdata_patch()  # noop if users never install litdata

__all__ = [
    "Dataset",
    "InMemoryShard",
    "MixtureSpec",
    "Pipeline",
    "SampleBatch",
    "SampleMeta",
    "SampleRecord",
    "StaticMixtureWorkSource",
    "WorkSource",
    "__version__",
]
