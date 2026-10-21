"""Temporary diagnostics for PR #61's Linux cache-lock timeout."""

import gc
import json
import sys
import threading
import time
import traceback
from collections import deque
from pathlib import Path

import filelock

_events = deque(maxlen=150)
_timer = None


def _event(kind, lock, **extra):
    _events.append(
        dict(
            time=time.monotonic(),
            kind=kind,
            lock=id(lock),
            path=lock.lock_file,
            thread=threading.get_ident(),
            counter=lock.lock_counter,
            fd=lock._context.lock_file_fd,
            **extra,
        )
    )


def _wrap(original, name):
    def wrapped(lock, *args, **kwargs):
        if not lock.lock_file.endswith('.reset.lock'):
            return original(lock, *args, **kwargs)
        _event(name + ':enter', lock)
        try:
            result = original(lock, *args, **kwargs)
        except Exception as exc:
            _event(name + ':error', lock, error=repr(exc))
            if not isinstance(exc, filelock.Timeout) or lock.is_locked:
                print('CACHE LOCK ERROR', traceback.format_exc(), file=sys.stderr, flush=True)
            raise
        _event(name + ':exit', lock)
        return result

    return wrapped


def pytest_configure(config):
    if sys.version_info[:2] != (3, 10):
        return
    print('CACHE LOCK DIAGNOSTICS', sys.version, filelock.__version__, flush=True)
    for name in ('acquire', 'release'):
        setattr(filelock.BaseFileLock, name, _wrap(getattr(filelock.BaseFileLock, name), name))


def pytest_runtest_setup(item):
    global _timer
    if sys.version_info[:2] != (3, 10):
        return
    if 'test_parquet_integration_reproducibility' not in item.nodeid:
        return
    _timer = threading.Timer(10, _dump)
    _timer.daemon = True
    _timer.start()


def pytest_runtest_teardown(item):
    global _timer
    if _timer is not None:
        _timer.cancel()
        _timer = None


def _dump():
    print('CACHE LOCK SNAPSHOT BEGIN', file=sys.stderr, flush=True)
    print(json.dumps(list(_events)), file=sys.stderr, flush=True)
    for obj in gc.get_objects():
        if not isinstance(obj, filelock.BaseFileLock) or not obj.lock_file.endswith('.reset.lock'):
            continue
        holds = []
        for state, thread in getattr(obj, '_thread_holds', {}).copy().values():
            holds.append(dict(fd=state.lock_file_fd, counter=state.lock_counter,
                              thread=thread.ident, alive=thread.is_alive(),
                              acquiring=repr(state.acquiring), releasing=repr(state.releasing)))
        print(json.dumps(dict(lock=id(obj), path=obj.lock_file, holds=holds)), file=sys.stderr, flush=True)
    for fd in Path('/proc/self/fd').iterdir():
        try:
            target = fd.readlink()
            if '.reset.lock' in str(target):
                print('FD', fd.name, target, Path('/proc/self/fdinfo', fd.name).read_text(),
                      file=sys.stderr, flush=True)
        except OSError:
            pass
    try:
        print('PROC LOCKS', Path('/proc/locks').read_text(), file=sys.stderr, flush=True)
    except OSError:
        pass
    for thread, frame in sys._current_frames().items():
        print('THREAD', thread, ''.join(traceback.format_stack(frame)), file=sys.stderr, flush=True)
    print('CACHE LOCK SNAPSHOT END', file=sys.stderr, flush=True)
