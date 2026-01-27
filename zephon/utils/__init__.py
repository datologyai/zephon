# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Helper utilities shared across runtime components."""

from zephon.debug.semaphore_registry import dump_semaphore_registry
from zephon.utils.buffering import buffered_iterable
from zephon.utils.fault_handling import (
    ShutdownWatchdog,
    dump_all_threads,
    setup_faulthandler,
)
from zephon.utils.gc import (
    cleanup_semaphores,
    collect_with_finalizers,
    disable_gc,
    is_gil_disabled,
)
from zephon.utils.seeding import batch_seed
from zephon.utils.semaphore import SafeSemLock
from zephon.utils.swrr import SmoothWeightedRoundRobin, swrr_iterate

__all__ = [
    "buffered_iterable",
    "batch_seed",
    "cleanup_semaphores",
    "collect_with_finalizers",
    "disable_gc",
    "dump_semaphore_registry",
    "is_gil_disabled",
    "SafeSemLock",
    "setup_faulthandler",
    "dump_all_threads",
    "ShutdownWatchdog",
    "SmoothWeightedRoundRobin",
    "swrr_iterate",
]
