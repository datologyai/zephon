"""Watchdog-side recovery glue for resilient-worker stage runners.

This module owns the OS-specific bits that the
:class:`ProcessStageRunner` watchdog needs after a worker is detected
as signal-terminated, so ``zephon/runners/process.py`` doesn't have to
care about POSIX semaphores, ``/proc`` introspection, or
result-queue lifecycle.

Public API:

- :func:`recover_result_queue` — run the recovery strategy chain.
- :func:`close_abandoned_result_queues` — shutdown helper for queues
  rotated out by recovery.

Background.  ``mp.Queue._feed`` does
``with _wlock: send_bytes(...)``, so a worker that crashes inside that
critical section leaves the non-robust POSIX semaphore permanently
acquired.  Subsequent writers block forever on ``wacquire``; the pump
never sees results; the pipeline hangs.

The race is dramatically more likely under free-threaded Python
(``PYTHON_GIL=0``) where ``_feed`` runs concurrently with the worker's
main thread rather than cooperatively.

This module implements a tiered strategy chain:

1. **Queue rotation** (``parallelism == 1`` only).  Replace
   ``state.result_queue`` with a fresh queue and abandon the old one.
   No ``_wlock`` inspection — the wedge becomes irrelevant because no
   one ever writes to or reads from the old queue again.  Cross-platform.
   The pump reads ``state.result_queue`` each iteration with no caching,
   so the swap is transparent (modulo one ~50 ms blocking ``get()`` on
   the old queue that times out).  Also handles partial-pipe-data
   automatically.
2. **Timed acquire** (``parallelism > 1``, all platforms).
   ``Lock.acquire(timeout=1.0)``.  If acquired the lock was healthy and
   we give it back.  If timed out, escalate.
3. **`/proc/PID/syscall` introspection** (Linux only,
   ``parallelism > 1``).  After a timed-acquire timeout, check whether
   any live worker thread is currently in a ``write``/``writev``/
   ``pwrite``/``pwritev`` syscall to the result-queue transport fd
   (pipe or socketpair, per ``NamedQueue``'s transport).  If yes,
   the lock is legitimately held by a live writer (e.g. blocked in
   ``os.write`` under pump backpressure); leave it alone.  If no, the
   lock holder must have been the dead worker; force-release once.

Each strategy can be disabled via ``ZEPHON_RECLAIM_DISABLE`` (a
comma-separated list of strategy names: ``rotation``, ``timed_acquire``,
``proc_syscall``) so the chain can be exercised in isolation during
testing.

On macOS without ``/proc``, the ``parallelism > 1`` case has no reliable
strong signal beyond the timed acquire heuristic.  Leaving the lock
alone there is safer than blindly releasing under prolonged backpressure
(a live ``send_bytes`` blocked in ``os.write`` for 1 s+ would be
falsely identified as wedged, and force-release would corrupt the pipe
stream by allowing two concurrent writers).  Known limitation.
"""

from __future__ import annotations

import os
import stat
import sys
import time
from multiprocessing.process import BaseProcess
from typing import Any, Callable

from zephon.utils.rank import rank_ctx

#: Env var that controls which strategies are active.  Read **at
#: every recovery call** rather than at module import so tests (and
#: ad-hoc operations) can flip strategies on and off after the module
#: has been loaded.  Comma-separated list of strategy names:
#: ``rotation``, ``timed_acquire``, ``proc_syscall``.
_RECLAIM_DISABLE_ENV = "ZEPHON_RECLAIM_DISABLE"


def _disabled_strategies() -> frozenset[str]:
    return frozenset(
        s.strip()
        for s in os.environ.get(_RECLAIM_DISABLE_ENV, "").split(",")
        if s.strip()
    )


#: Linux write-family syscall numbers (x86_64 + arm64).  Used by
#: :func:`_pid_any_thread_in_write_to_fd` to detect whether a live
#: worker is currently in a ``write*`` syscall on the result-queue pipe
#: fd.  Each architecture has its own number for the same syscall; we
#: accept either set since we don't know the worker's arch up front.
_LINUX_WRITE_SYSCALLS: frozenset[int] = frozenset(
    {
        # x86_64
        1,  # write
        18,  # pwrite64
        20,  # writev
        296,  # pwritev
        328,  # pwritev2
        # arm64
        64,  # write
        66,  # writev
        68,  # pwrite64
        70,  # pwritev
        286,  # pwritev2
    }
)


def recover_result_queue(
    *,
    state: Any,
    dead_proc: BaseProcess,
    worker_index: int,
    make_ipc_queue: Callable[[str], Any],
) -> None:
    """Run the strategy chain to recover ``state.result_queue`` after a worker death.

    Args:
        state: The operator state.  Must have ``parallelism``,
            ``result_queue``, ``workers``, ``_abandoned_result_queues``,
            ``node.name``, ``op_index``, and ``stage_name`` attributes.
        dead_proc: The worker process the watchdog just observed dying.
        worker_index: Index of ``dead_proc`` in ``state.workers``.
        make_ipc_queue: Factory ``str -> NamedQueue`` for queue rotation.
    """
    dead_pid = dead_proc.pid
    disabled = _disabled_strategies()

    # Step 1: rotation (parallelism=1).  Proper fix; no lock inspection.
    if state.parallelism == 1 and "rotation" not in disabled:
        _rotate_result_queue(state, dead_pid, make_ipc_queue)
        return

    wlock = _unwrap_wlock(state.result_queue)
    if wlock is None:
        return

    # Step 2: timed acquire — catches healthy lock cycles cheaply.
    if "timed_acquire" not in disabled:
        try:
            acquired = wlock.acquire(block=True, timeout=1.0)
        except Exception as exc:  # noqa: BLE001 - report and continue
            # Probe failed.  We can't tell whether the lock is wedged;
            # don't act, but surface loudly so the symptom is visible.
            print(
                f"[zephon] WARNING: probe-acquire on result_queue._wlock "
                f"raised {exc!r} for dead worker pid={dead_pid}. Cannot "
                f"determine lock state; pipeline may hang. "
                f"({rank_ctx()})",
                file=sys.stderr,
                flush=True,
            )
            return
        if acquired:
            try:
                wlock.release()
            except Exception as exc:  # noqa: BLE001 - report loudly
                # We successfully acquired _wlock but failed to release
                # it.  The lock is now held by the parent process — every
                # future writer will block on it.  This is *worse* than
                # the original wedge: we created a permanent block.  No
                # safe recovery from here; surface loudly so a human can
                # intervene (kill the run, restart the pipeline).
                print(
                    f"[zephon] CRITICAL: failed to release "
                    f"result_queue._wlock after probe-acquire: {exc!r}. "
                    f"Lock now held by parent process; all writers will "
                    f"block. Likely indicates corrupted semaphore state. "
                    f"({rank_ctx()})",
                    file=sys.stderr,
                    flush=True,
                )
            return
        # Else: acquire timed out → some writer has held the lock for
        # the full 1 s.  We don't yet know if that's a wedge (dead
        # holder) or a live writer mid-``send_bytes`` under
        # backpressure; escalate to step 3.

    # Step 3: /proc/PID/syscall (Linux only).  Distinguishes wedge from
    # live-writer-mid-send by inspecting per-thread kernel syscall state
    # of every remaining live worker.  We match by pipe inode, not fd
    # number — fd numbers are process-local and the worker's fd for the
    # same pipe is generally NOT the same as the parent's fd.
    if sys.platform.startswith("linux") and "proc_syscall" not in disabled:
        pipe_inode = _get_pipe_write_inode(state.result_queue)
        if pipe_inode is None:
            return  # can't introspect — leave alone
        if not _proc_syscall_no_live_writer(state.workers, dead_proc, pipe_inode):
            # Either a live writer was observed, or /proc was unreadable
            # in a way that prevented discrimination.  Conservative:
            # leave ``_wlock`` alone.  Pipeline may hang if the lock was
            # actually wedged, but we don't risk corrupting the pipe
            # stream by force-releasing while a live writer holds the
            # lock.
            return
        # /proc multi-sampling positively confirmed no live writer is
        # writing.  Before force-releasing, do a final non-blocking
        # acquire to catch the tight race where a live writer
        # transitioned out of ``write`` between our last sample and
        # this point and (hypothetically) released the lock.  If we
        # acquire successfully, the lock is actually free — release
        # back and skip the force-release path entirely.
        try:
            recheck = wlock.acquire(block=False)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(
                f"[zephon] WARNING: post-/proc recheck-acquire raised "
                f"{exc!r} for dead worker pid={dead_pid}. Skipping "
                f"force-release. ({rank_ctx()})",
                file=sys.stderr,
                flush=True,
            )
            return
        if recheck:
            # Lock was actually free — no wedge after all.  Release
            # back, no force-release.
            try:
                wlock.release()
            except Exception as exc:  # noqa: BLE001 - report loudly
                print(
                    f"[zephon] CRITICAL: failed to release "
                    f"result_queue._wlock after recheck-acquire: "
                    f"{exc!r}. Lock now held by parent process; all "
                    f"writers will block. ({rank_ctx()})",
                    file=sys.stderr,
                    flush=True,
                )
            return

        # Genuinely held + no live writer → wedged.  Reclaim.
        print(
            f"[zephon] reclaiming wedged result_queue._wlock from "
            f"dead worker pid={dead_pid} worker_idx={worker_index} "
            f"(/proc/syscall confirms no live writer; {rank_ctx()})",
            file=sys.stderr,
            flush=True,
        )
        try:
            wlock.release()
        except Exception as exc:  # noqa: BLE001 - report and continue
            # Forced release of the wedged lock failed.  The wedge
            # persists; the pipeline will hang.  Nothing more to do
            # from here, but surface so the symptom is visible.
            print(
                f"[zephon] WARNING: failed to release wedged "
                f"result_queue._wlock for dead worker pid={dead_pid}: "
                f"{exc!r}. Wedge persists; pipeline may hang. "
                f"({rank_ctx()})",
                file=sys.stderr,
                flush=True,
            )
        return

    # Fall-through: no enabled strategy could safely act (macOS with
    # parallelism > 1, or all strategies disabled).  Known limitation.
    print(
        f"[zephon] WARNING: no recovery strategy applicable for dead "
        f"worker pid={dead_pid} worker_idx={worker_index} "
        f"(parallelism={state.parallelism}, platform={sys.platform}, "
        f"disabled={sorted(disabled) or 'none'}). If result_queue._wlock "
        f"is wedged, the pipeline will hang. ({rank_ctx()})",
        file=sys.stderr,
        flush=True,
    )


def close_abandoned_result_queues(state: Any, close_fn: Callable[[Any], None]) -> None:
    """Close every result queue rotated out of ``state`` by recovery.

    Queue rotation appends old queues to
    ``state._abandoned_result_queues`` rather than closing them
    in-place — closing at rotation time would race with the pump's
    in-flight ``get()`` on the old queue.  At runner shutdown the
    pump is already stopped so it's safe to close.

    ``close_fn`` is the runner-side close helper (typically a closure
    over ``self._close_ipc_queue(q, hard=hard)``); the watchdog
    module doesn't need to know the close semantics.
    """
    abandoned: list[Any] = getattr(state, "_abandoned_result_queues", [])
    for old_queue in abandoned:
        close_fn(old_queue)
    abandoned.clear()


def _rotate_result_queue(
    state: Any,
    dead_pid: int | None,
    make_ipc_queue: Callable[[str], Any],
) -> None:
    """Replace ``state.result_queue`` with a fresh queue.

    The old queue is appended to ``state._abandoned_result_queues`` and
    closed at shutdown — closing it now is unsafe because the pump may
    still be inside an old ``get()``.  Garbage in the old pipe (partial
    frames, stuck ``_wlock``) is harmless because nothing ever reads
    from or writes to the old queue again.
    """
    op_label = state.node.name or f"op{state.op_index}"
    queue_label = f"{state.stage_name}:{op_label}"
    new_queue = make_ipc_queue(f"result:{queue_label}")
    old_queue = state.result_queue
    state.result_queue = new_queue
    state._abandoned_result_queues.append(old_queue)
    print(
        f"[zephon] rotated result_queue after worker death "
        f"(stage={state.stage_name!r} op={state.node.name!r} "
        f"dead_pid={dead_pid}; {rank_ctx()})",
        file=sys.stderr,
        flush=True,
    )


def _unwrap_wlock(result_queue: Any) -> Any:
    """Return the inner ``mp.synchronize.Lock`` from NamedQueue's wlock.

    Peels the ``SafeSemLock`` wrapper from ``result_queue._wlock``;
    returns ``None`` if the queue has no ``_wlock`` attribute.
    """
    wrapper = getattr(result_queue, "_wlock", None)
    if wrapper is None:
        return None
    inner = getattr(wrapper, "_sem", None)
    return inner if inner is not None else wrapper


# --- Linux /proc-based write-end-inode introspection -----------------------
#
# We match by **inode** rather than by fd number because fd numbers
# are process-local: when ``mp.Queue`` is sent across processes via
# forkserver / spawn, the receiving process gets the transport via
# ``recv_fds(SCM_RIGHTS)``, which yields a *fresh fd number* pointing to
# the same kernel object.  The parent's fd 5 may be the worker's
# fd 8.  Matching by ``arg0 == parent_fd`` would never identify the
# worker as a live writer.
#
# Inodes survive fd renumbering.  ``NamedQueue``'s transport is either a
# pipe or an AF_UNIX socketpair; ``/proc/PID/fd/N`` symlinks resolve to
# ``pipe:[INODE]`` / ``socket:[INODE]``, so we can map worker fds → inode
# and compare to the parent's write-end inode.  A pipe's two ends share
# one inode; a socketpair's two endpoints have *distinct* inodes — the
# match still works because workers hold SCM_RIGHTS dups of the parent's
# writer endpoint, which carry the writer's inode.

# Number of /proc/PID/syscall samples to take when looking for a live
# writer.  A live worker that's actively producing should be observed
# in `write` at least once across multiple samples even when the
# kernel briefly reports them as "running" or they're in the
# microsecond userspace gap between two ``os.write`` calls of a
# multi-chunk send.
_PROC_SAMPLES = 8
#: Wall time between samples in seconds.  Total sample window is
#: ``_PROC_SAMPLES * _PROC_SAMPLE_INTERVAL_S``.
_PROC_SAMPLE_INTERVAL_S = 0.025


def _get_pipe_write_inode(result_queue: Any) -> int | None:
    """Return the inode of ``result_queue``'s write-end fd, or ``None``.

    ``os.fstat(fd).st_ino`` works regardless of process-local fd
    numbering — the inode is shared by every dup of the same pipe or
    socketpair endpoint.

    Only pipe and socket fds qualify: those are the two ``NamedQueue``
    transports, and the only fd types
    :func:`_find_worker_fds_for_inode` can resolve in ``/proc/PID/fd``.
    Any other fd type returns ``None`` — the recovery chain then leaves
    the lock alone rather than force-releasing on a vacuous
    "no worker holds this fd" match.
    """
    writer = getattr(result_queue, "_writer", None)
    if writer is None:
        return None
    try:
        st = os.fstat(writer.fileno())
    except Exception:  # noqa: BLE001 - best effort
        return None
    if not (stat.S_ISFIFO(st.st_mode) or stat.S_ISSOCK(st.st_mode)):
        return None
    return int(st.st_ino)


def _proc_syscall_no_live_writer(
    workers: list[BaseProcess],
    dead_proc: BaseProcess,
    pipe_inode: int,
) -> bool:
    """Linux-only: confirm no live worker is currently writing to the pipe.

    Returns True iff we **positively confirm** no live worker (other
    than ``dead_proc``) is observed in a write-family syscall to the
    pipe with ``pipe_inode`` across :data:`_PROC_SAMPLES` samples
    spaced :data:`_PROC_SAMPLE_INTERVAL_S` apart.

    Returning True is the *strong* signal — only safe to force-release
    on True.  Returns False if:

    - any live worker is observed in a write to the target pipe
      across any sample, OR
    - we couldn't read the relevant ``/proc`` files (permission,
      transient access issue) for a worker that ``proc.is_alive()``
      claimed was alive — in this case we can't discriminate, and
      the safe action is to leave the lock alone.

    Process-disappeared between checks (FileNotFoundError on a
    ``/proc/PID/...`` path) is treated as "this worker is gone, can't
    be a live writer", consistent with our overall semantics.

    Implementation:

    1. For each live worker, resolve its ``/proc/PID/fd/*`` symlinks
       once to find which fds point to the target pipe inode.  Workers
       don't open new pipes during a recovery window so this mapping
       is stable.
    2. Sample :data:`_PROC_SAMPLES` times.  At each sample, walk every
       live worker's threads via ``/proc/PID/task/TID/syscall`` and
       check whether any thread is in a ``write*`` syscall whose
       ``arg0`` (fd) is in the worker's pipe-fd set.
    3. If any sample shows a live writer, return False immediately.
    """
    live_fds_by_pid: dict[int, set[int]] = {}
    for proc in workers:
        if proc is dead_proc or not proc.is_alive():
            continue
        pid = proc.pid
        if pid is None:
            continue
        fds, ok = _find_worker_fds_for_inode(pid, pipe_inode)
        if not ok:
            # Couldn't read /proc/PID/fd for a live worker.  Be
            # conservative — could be holding the lock under
            # backpressure.
            return False
        live_fds_by_pid[pid] = fds

    if not live_fds_by_pid:
        # No live workers → dead worker was the only writer.
        return True

    for sample_idx in range(_PROC_SAMPLES):
        if sample_idx > 0:
            time.sleep(_PROC_SAMPLE_INTERVAL_S)
        for pid, target_fds in live_fds_by_pid.items():
            if not target_fds:
                # Worker has no fd for our pipe — can't be a writer.
                continue
            saw_writer, ok = _pid_any_thread_in_write_to_fds(pid, target_fds)
            if not ok:
                # Couldn't read syscall state for a live worker.
                # Conservative.
                return False
            if saw_writer:
                return False
    return True


def _find_worker_fds_for_inode(pid: int, inode: int) -> tuple[set[int], bool]:
    """Resolve fds in process ``pid`` that point to a pipe or socket with ``inode``.

    Returns ``(fds, ok)`` where ``fds`` is the set of file descriptors
    in process ``pid`` that point to a pipe or socket with the given
    inode, and ``ok`` is False if we couldn't read ``/proc/PID/fd`` in
    a way that suggests the process is alive but inaccessible.

    Reads ``/proc/PID/fd/N`` symlinks; pipe fds resolve to the form
    ``pipe:[NNNNN]`` and socket fds to ``socket:[NNNNN]`` — both
    ``NamedQueue`` transports must match, else every worker looks like
    a non-writer and the caller would falsely force-release.

    A FileNotFoundError on the directory means the process exited
    between ``proc.is_alive()`` and this read — we treat that as
    ``ok=True`` with an empty set, since the process is gone and
    therefore can't be a live writer anymore.

    Other OSErrors (notably PermissionError) yield ``ok=False`` —
    the caller should treat that as "unknown, do not force-release".
    """
    matching: set[int] = set()
    fd_dir = f"/proc/{pid}/fd"
    try:
        names = os.listdir(fd_dir)
    except FileNotFoundError:
        return matching, True  # process gone — empty set is correct
    except OSError:
        return matching, False  # access failure
    for name in names:
        try:
            target = os.readlink(f"{fd_dir}/{name}")
        except FileNotFoundError:
            # fd vanished between listdir and readlink — legitimate,
            # this fd is gone, just keep walking.
            continue
        except OSError:
            # Any other readlink error (PermissionError or other
            # OSError) is "we can't tell what this fd points to".  If
            # the unreadable fd happens to be the result-pipe fd, a
            # silent skip would let the caller conclude "no matching
            # fd → not a writer" and force-release.  Fail closed.
            return matching, False
        if target.endswith("]") and target.startswith("pipe:["):
            inode_text = target[6:-1]
        elif target.endswith("]") and target.startswith("socket:["):
            inode_text = target[8:-1]
        else:
            continue
        try:
            target_inode = int(inode_text)
        except ValueError:
            continue
        if target_inode != inode:
            continue
        try:
            matching.add(int(name))
        except ValueError:
            continue
    return matching, True


def _pid_any_thread_in_write_to_fds(
    pid: int, target_fds: set[int]
) -> tuple[bool, bool]:
    """Return ``(saw_writer, ok)`` for a Linux process.

    ``saw_writer`` is True iff any thread of ``pid`` is currently
    observed in a write-family syscall whose first argument (the fd)
    is in ``target_fds``.  ``ok`` is False if we couldn't read
    ``/proc/PID/task/`` in a way that suggests the process is alive
    but inaccessible — caller should treat as "unknown".

    Per-thread ``FileNotFoundError`` on the syscall file is just a
    transient (the thread exited mid-walk); we skip that thread and
    keep going.  ``FileNotFoundError`` on the task dir itself means
    the whole process exited; ``saw_writer=False, ok=True``.

    Reads ``/proc/PID/task/TID/syscall``.  Format is either
    ``running`` (currently in user mode), ``-1 <sp> <pc>`` (not in
    any syscall), or ``<num> <arg0> ... <arg5> <sp> <pc>`` where
    ``num`` is decimal and ``arg0..pc`` are hex (with ``0x`` prefix).
    """
    task_dir = f"/proc/{pid}/task"
    try:
        tids = os.listdir(task_dir)
    except FileNotFoundError:
        return False, True  # process gone
    except OSError:
        return False, False  # access failure
    for tid in tids:
        try:
            with open(f"{task_dir}/{tid}/syscall", "r") as f:
                line = f.read().strip()
        except FileNotFoundError:
            # Thread exited mid-walk; skip it.
            continue
        except PermissionError:
            return False, False  # access failure on a live thread
        except OSError:
            continue  # other transient — skip this thread
        if not line or line == "running" or line.startswith("-1"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            syscall_num = int(parts[0])
        except ValueError:
            continue
        if syscall_num not in _LINUX_WRITE_SYSCALLS:
            continue
        try:
            arg_fd = int(parts[1], 16)
        except ValueError:
            continue
        if arg_fd in target_fds:
            return True, True
    return False, True
