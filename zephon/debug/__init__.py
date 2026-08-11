# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Public exploration and diagnostic utilities for zephon.

Use :class:`DatasetInspector` for interactive dataset exploration. Lower-level
runtime diagnostics are typically enabled via environment variables:

- ZEPHON_SEMAPHORE_LEAK_DEBUG=1: Track semaphore registrations with resource_tracker
- ZEPHON_SEMAPHORE_DEBUG=1: Track SafeSemLock instances
"""

from zephon.debug.inspector import DatasetInspector
from zephon.debug.semaphore_registry import dump_semaphore_registry
from zephon.debug.semaphore_tracker import (
    dump_semaphore_leak_report,
    install_debug_hooks,
)

__all__ = [
    "DatasetInspector",
    "dump_semaphore_leak_report",
    "dump_semaphore_registry",
    "install_debug_hooks",
]
