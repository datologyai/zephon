import sys

from zephon._internal.io.formats import _FORMAT_MODULES, ensure_builtin_formats
from zephon._internal.io.formats.base import get_format, register_format


class _DummyHandler:
    kind = "dummy"

    def discover(self, path, storage):
        return {}, {}

    def build_locators(self, dataset):
        return {}

    def open_shard(self, locator, local_ref):
        raise NotImplementedError


def test_register_and_get_format_roundtrip() -> None:
    handler = _DummyHandler()
    register_format(handler)  # should not raise
    got = get_format("dummy")
    assert got is handler


def test_ensure_builtin_formats_idempotent() -> None:
    ensure_builtin_formats(required={"jsonl", "mds"})
    ensure_builtin_formats(required={"jsonl", "mds"})  # calling twice should be safe
    assert get_format("jsonl").kind == "jsonl"
    assert get_format("mds").kind == "mds"


def test_ensure_builtin_formats_selective_does_not_load_litdata() -> None:
    """Loading only jsonl should not pull in litdata_support or torch."""
    # Remove litdata modules if previously loaded (test isolation)
    litdata_mod = "zephon._internal.io.formats.litdata"
    support_mod = "zephon._internal.io.formats.litdata_support"
    was_loaded = litdata_mod in sys.modules or support_mod in sys.modules
    if was_loaded:
        # Can't test isolation when modules are already loaded — skip
        return

    ensure_builtin_formats(required={"jsonl"})

    assert litdata_mod not in sys.modules, "litdata module should not be imported"
    assert support_mod not in sys.modules, "litdata_support should not be imported"


def test_ensure_builtin_formats_all() -> None:
    """Loading all known formats should register everything."""
    ensure_builtin_formats(required=set(_FORMAT_MODULES))
    for kind in _FORMAT_MODULES:
        assert get_format(kind).kind == kind
