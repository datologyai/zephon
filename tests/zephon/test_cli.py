# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

import zephon.cli as cli


@dataclass
class _FakeEntryPoint:
    """Stands in for ``importlib.metadata.EntryPoint`` without an installed dist."""

    name: str
    target: Callable[[argparse.ArgumentParser], None]
    value: str = "fake_plugin:register"
    dist: None = None

    def load(self) -> Callable[[argparse.ArgumentParser], None]:
        return self.target


def _use_plugins(monkeypatch: pytest.MonkeyPatch, *eps: _FakeEntryPoint) -> None:
    monkeypatch.setattr(cli, "_plugin_entry_points", lambda: list(eps))


@pytest.fixture(autouse=True)
def _no_installed_plugins(monkeypatch: pytest.MonkeyPatch) -> None:
    # Keep tests independent of whatever happens to be installed in the venv.
    _use_plugins(monkeypatch)


def test_index_builds_jsonl_index(tmp_path: Path, capsys) -> None:
    (tmp_path / "a.jsonl").write_text('{"i": 0}\n{"i": 1}\n', encoding="utf-8")

    assert cli.main(["index", "jsonl", str(tmp_path)]) == 0

    data = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
    assert [shard["num_rows"] for shard in data["shards"]] == [2]
    assert "Created" in capsys.readouterr().out


def test_index_passes_options(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_build_index(fmt, dataset_dir, **kwargs):
        seen.update(fmt=fmt, dir=dataset_dir, **kwargs)
        return "/out/idx.json"

    monkeypatch.setattr(cli, "build_index", fake_build_index)

    code = cli.main(
        ["index", "parquet", "s3://b/ds", "-o", "/out/idx.json", "-j", "3", "-q"]
    )

    assert code == 0
    assert seen == {
        "fmt": "parquet",
        "dir": "s3://b/ds",
        "output_path": "/out/idx.json",
        "progress": False,
        "max_workers": 3,
    }


def test_index_reports_errors(tmp_path: Path, capsys) -> None:
    assert cli.main(["index", "jsonl", str(tmp_path / "missing")]) == 1
    assert "zephon index: error:" in capsys.readouterr().err


def test_index_rejects_bad_max_workers(capsys) -> None:
    assert cli.main(["index", "jsonl", "/data", "--max-workers", "0"]) == 2
    assert "--max-workers" in capsys.readouterr().err


def test_index_rejects_unknown_format(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["index", "lance", "/data"])
    assert exc.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_no_command_prints_help(capsys) -> None:
    assert cli.main([]) == 2
    out = capsys.readouterr().out
    assert "usage: zephon" in out
    assert "index" in out


def test_plugin_command_is_registered_and_runs(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    def register(parser: argparse.ArgumentParser) -> None:
        parser.description = "Say hello."
        parser.add_argument("who")
        parser.add_argument("--loud", action="store_true")
        parser.set_defaults(func=run)

    def run(args: argparse.Namespace) -> int:
        greeting = f"hello {args.who}"
        print(greeting.upper() if args.loud else greeting)
        return 7

    _use_plugins(monkeypatch, _FakeEntryPoint("hello", register))

    assert cli.main(["hello", "world", "--loud"]) == 7
    assert capsys.readouterr().out == "HELLO WORLD\n"

    with pytest.raises(SystemExit):
        cli.main(["--help"])
    assert "Say hello." in capsys.readouterr().out

    with pytest.raises(SystemExit):
        cli.main(["hello", "--help"])
    help_text = capsys.readouterr().out
    assert "usage: zephon hello" in help_text
    assert "--loud" in help_text


def test_plugin_handler_returning_none_exits_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def register(parser: argparse.ArgumentParser) -> None:
        parser.set_defaults(func=lambda args: None)

    _use_plugins(monkeypatch, _FakeEntryPoint("noop", register))

    assert cli.main(["noop"]) == 0


def test_broken_plugin_is_skipped(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    def broken(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--half")
        raise RuntimeError("boom")

    _use_plugins(monkeypatch, _FakeEntryPoint("broken", broken))

    with pytest.raises(SystemExit):
        cli.main(["broken"])
    err = capsys.readouterr().err
    assert "failed to load plugin command 'broken'" in err
    assert "boom" in err
    assert "invalid choice" in err


def test_plugin_cannot_shadow_builtin_or_earlier_plugin(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    calls: list[str] = []

    def make(tag: str) -> Callable[[argparse.ArgumentParser], None]:
        def register(parser: argparse.ArgumentParser) -> None:
            calls.append(tag)
            parser.set_defaults(func=lambda args: 0)

        return register

    _use_plugins(
        monkeypatch,
        _FakeEntryPoint("index", make("shadow-index")),
        _FakeEntryPoint("dup", make("first")),
        _FakeEntryPoint("dup", make("second")),
    )

    assert cli.main(["dup"]) == 0
    assert calls == ["first"]
    err = capsys.readouterr().err
    assert "skipping plugin command 'index'" in err
    assert "skipping plugin command 'dup'" in err


def test_python_dash_m_zephon() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "zephon", "index", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "usage: zephon index" in proc.stdout
