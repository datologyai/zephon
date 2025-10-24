"""Compatibility helpers for TorchData stateful dataloader.

This module installs a small runtime patch that normalizes the
"worker_0" snapshot across TorchData variants so checkpoint/restore
behaves consistently regardless of multiprocessing layout.
"""

import copy
import sys
import threading
from types import ModuleType
from typing import Any, Callable

from wrapt.importer import register_post_import_hook

_LOCK = threading.Lock()

_WATCH = (
    "torchdata",
    "torchdata.stateful_dataloader",
    "torchdata.dataloader2.stateful_dataloader",
)


def _unify_worker0_snapshot(
    state_dict: dict,
    SNAPSHOT_K: str,
    WORKER_SNAPSHOTS_K: str,
    worker_key_fn: Callable[[int], str] | None,
) -> dict:
    snap = state_dict.get(SNAPSHOT_K)
    if not snap:
        return state_dict
    ws = snap.get(WORKER_SNAPSHOTS_K)
    if not ws:
        return state_dict

    # prefer "worker_0", else first available
    key0 = worker_key_fn(0) if worker_key_fn else "worker_0"
    base = ws.get(key0)
    if base is None:
        try:
            _, base = next(iter(ws.items()))
        except StopIteration:
            return state_dict

    new_ws = {k: copy.deepcopy(base) for k in ws.keys()}
    new_snap = dict(snap)
    new_snap[WORKER_SNAPSHOTS_K] = new_ws
    out = dict(state_dict)
    out[SNAPSHOT_K] = new_snap
    return out


def _apply_patch(module: ModuleType) -> None:
    name = getattr(module, "__name__", "")
    # Resolve the stateful_dataloader module object WITHOUT importing new modules
    if name in (
        "torchdata.stateful_dataloader",
        "torchdata.dataloader2.stateful_dataloader",
    ):
        sd = module
    elif name == "torchdata":
        sd = sys.modules.get("torchdata.stateful_dataloader") or sys.modules.get(
            "torchdata.dataloader2.stateful_dataloader"
        )
        if sd is None:
            return
    else:
        return

    with _LOCK:
        SDL = getattr(sd, "StatefulDataLoader", None)
        if SDL is None:
            return
        if getattr(SDL, "_mylib_worker0_unifier", False):
            return  # already patched

        # Grab iterator constants if present (needed just for key names)
        MP = getattr(sd, "_StatefulMultiProcessingDataLoaderIter", None)
        if MP is not None:
            SNAPSHOT_K = MP._SNAPSHOT
            WORKER_SNAPSHOTS_K = MP._WORKER_SNAPSHOTS
        else:
            # Fallback names used by all current torchdata variants
            SNAPSHOT_K = "_snapshot"
            WORKER_SNAPSHOTS_K = "_worker_snapshots"

        _orig_state_dict = SDL.state_dict

        def _state_dict_unify_worker0(self: Any) -> dict:
            # call original
            out = _orig_state_dict(self)
            # Try to obtain worker_key function from a live iterator if present
            worker_key_fn = None
            try:
                it = self._iterator
                worker_key_fn = getattr(it, "_worker_key", None)
            except Exception:
                pass
            # rewrite
            return _unify_worker0_snapshot(
                out, SNAPSHOT_K, WORKER_SNAPSHOTS_K, worker_key_fn
            )

        SDL.state_dict = _state_dict_unify_worker0
        SDL._mylib_worker0_unifier = True  # sentinel


def install_torchdata_patch():
    """Install the TorchData patch for consistent worker-0 snapshots."""
    # If any watched module is already loaded, patch immediately.
    for n in _WATCH:
        mod = sys.modules.get(n)
        if mod is not None:
            _apply_patch(mod)

    # And hook future imports.
    if register_post_import_hook is not None:
        for n in _WATCH:
            register_post_import_hook(_apply_patch, n)


__all__ = ["install_torchdata_patch"]  # dont export our internal globals and stuff
