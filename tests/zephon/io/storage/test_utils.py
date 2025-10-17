from pathlib import Path

from zephon.io.storage._utils import OpenViaDownloadMixin, TempLocalFile, split_url


def test_split_url_variants() -> None:
    assert split_url("s3://bucket/path/to.obj") == ("s3", "bucket", "path/to.obj")
    assert split_url("gs://b/x/y") == ("gs", "b", "x/y")
    assert split_url("gcs://b/") == ("gcs", "b", "")
    assert split_url("/abs/x") == ("", "", "abs/x")
    assert split_url("relative/x") == ("", "", "relative/x")


def test_temp_local_file_lifecycle(tmp_path: Path) -> None:
    p = tmp_path / "t.txt"
    p.write_text("ok", encoding="utf-8")

    t = TempLocalFile(p, "rb")
    with t as fh:
        assert fh.read() == b"ok"
    # path removed after context manager exits
    assert not p.exists()


class _FakeDL(OpenViaDownloadMixin):
    def download(self, src: str, dst: str, timeout=None):  # type: ignore[override]
        Path(dst).write_text(Path(src).read_text(encoding="utf-8"), encoding="utf-8")


def test_open_via_download_mixin(tmp_path: Path) -> None:
    src = tmp_path / "src.txt"
    src.write_text("hello", encoding="utf-8")
    backend = _FakeDL()
    with backend.open(str(src), "r", encoding="utf-8") as fh:  # type: ignore[arg-type]
        assert fh.read() == "hello"
    # underlying temp file is cleaned up by the context manager
