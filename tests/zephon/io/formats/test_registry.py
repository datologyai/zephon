from zephon.io.formats import ensure_builtin_formats
from zephon.io.formats.base import get_format, register_format


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
    ensure_builtin_formats()
    ensure_builtin_formats()  # calling twice should be safe
    # builtin formats should be registered
    assert get_format("jsonl").kind == "jsonl"
    assert get_format("mds").kind == "mds"
