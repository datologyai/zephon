# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Debug and diagnostic utilities for zephon.

These modules are for development/debugging and are typically enabled
via environment variables:

- ZEPHON_SEMAPHORE_LEAK_DEBUG=1: Track semaphore registrations with resource_tracker
- ZEPHON_SEMAPHORE_DEBUG=1: Track SafeSemLock instances
"""

from zephon.debug.semaphore_registry import dump_semaphore_registry
from zephon.debug.semaphore_tracker import (
    dump_semaphore_leak_report,
    install_debug_hooks,
)

__all__ = [
    "dump_semaphore_leak_report",
    "dump_semaphore_registry",
    "install_debug_hooks",
]
