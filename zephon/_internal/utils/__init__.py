# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Helper utilities shared across runtime components."""

from zephon._internal.utils.buffering import buffered_iterable
from zephon._internal.utils.fault_handling import (
    ShutdownWatchdog,
    dump_all_threads,
    setup_faulthandler,
)
from zephon._internal.utils.gc import (
    cleanup_semaphores,
    collect_with_finalizers,
    disable_gc,
    is_gil_disabled,
)
from zephon._internal.utils.seeding import batch_seed
from zephon._internal.utils.semaphore import SafeSemLock
from zephon._internal.utils.swrr import SmoothWeightedRoundRobin, swrr_iterate
from zephon.debug.semaphore_registry import dump_semaphore_registry

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
