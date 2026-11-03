# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""The ``zephon`` command-line entrypoint.

Run ``zephon --help`` (or ``python -m zephon --help``) for the list of
subcommands. Zephon ships ``zephon index``; other installed packages can add
their own subcommands through the ``zephon.cli`` entry-point group.

Plugin contract: the entry point's name is the subcommand name, and its object
is a callable that takes the subcommand's :class:`argparse.ArgumentParser`,
adds arguments to it, and sets ``func`` to the handler. The handler receives
the parsed :class:`argparse.Namespace` and returns an exit code (``None`` means
0). For example, a package ``zephon-foo`` would declare::

    [project.entry-points."zephon.cli"]
    foo = "zephon_foo.cli:register"

with::

    def register(parser: argparse.ArgumentParser) -> None:
        parser.description = "Do foo things."
        parser.add_argument("target")
        parser.set_defaults(func=run)

    def run(args: argparse.Namespace) -> int:
        ...

Every installed plugin is registered on each invocation, so keep heavy imports
inside the handler rather than at module level. A plugin that fails to load,
or whose name is already taken, is reported on stderr and skipped, so one
broken install cannot take down the rest of the CLI.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from importlib.metadata import EntryPoint, entry_points

from zephon.build_index import _FORMAT_INDEX_MODULES, build_index

#: Entry-point group that third-party packages use to add subcommands.
ENTRY_POINT_GROUP = "zephon.cli"

_Register = Callable[[argparse.ArgumentParser], None]


def _register_index(parser: argparse.ArgumentParser) -> None:
    parser.description = (
        "Build an index.json for a dataset so Zephon can discover its shards "
        "without scanning them. DATASET_DIR may be a local path or a remote "
        "storage root (e.g. s3://bucket/dataset)."
    )
    parser.add_argument("format", choices=sorted(_FORMAT_INDEX_MODULES))
    parser.add_argument("dataset_dir", metavar="DATASET_DIR")
    parser.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        help="where to write the index (default: DATASET_DIR/index.json)",
    )
    parser.add_argument(
        "-j",
        "--max-workers",
        type=int,
        default=8,
        metavar="N",
        help="shards to inspect concurrently (default: %(default)s)",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="suppress progress output"
    )
    parser.set_defaults(func=_run_index)


def _run_index(args: argparse.Namespace) -> int:
    if args.max_workers < 1:
        print("zephon index: error: --max-workers must be >= 1", file=sys.stderr)
        return 2
    try:
        build_index(
            args.format,
            args.dataset_dir,
            output_path=args.output,
            progress=not args.quiet,
            max_workers=args.max_workers,
        )
    except Exception as exc:
        print(f"zephon index: error: {exc}", file=sys.stderr)
        return 1
    return 0


_BUILTIN_COMMANDS: dict[str, tuple[str, _Register]] = {
    "index": ("build an index.json for a dataset", _register_index),
}


def _plugin_entry_points() -> list[EntryPoint]:
    return sorted(entry_points(group=ENTRY_POINT_GROUP), key=lambda ep: ep.name)


def _warn(message: str) -> None:
    print(f"zephon: warning: {message}", file=sys.stderr)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zephon", description="Zephon command-line tools."
    )
    subparsers = parser.add_subparsers(title="commands", metavar="COMMAND")

    for name, (summary, register) in _BUILTIN_COMMANDS.items():
        register(subparsers.add_parser(name, help=summary))

    taken = set(_BUILTIN_COMMANDS)
    for ep in _plugin_entry_points():
        source = ep.dist.name if ep.dist is not None else ep.value
        if ep.name in taken:
            _warn(f"skipping plugin command {ep.name!r} from {source}: name in use")
            continue
        # Configure a standalone parser first, so a register() that raises
        # leaves no half-built command in the subparsers.
        plugin = argparse.ArgumentParser(prog=f"zephon {ep.name}", add_help=False)
        try:
            ep.load()(plugin)
        except Exception as exc:
            _warn(f"failed to load plugin command {ep.name!r} from {source}: {exc}")
            continue
        taken.add(ep.name)
        # ``parents`` carries over the arguments, groups and defaults (``func``).
        subparsers.add_parser(
            ep.name,
            parents=[plugin],
            help=plugin.description,
            description=plugin.description,
            epilog=plugin.epilog,
            formatter_class=plugin.formatter_class,
        )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``zephon`` CLI and return its exit code.

    Args:
        argv: Arguments after the program name; defaults to ``sys.argv[1:]``.

    Returns:
        The handler's exit code: 0 on success, nonzero on failure. Usage errors
        raise :class:`SystemExit` with code 2, as :mod:`argparse` does.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 2
    code = func(args)
    return 0 if code is None else code


__all__ = ["ENTRY_POINT_GROUP", "main"]
