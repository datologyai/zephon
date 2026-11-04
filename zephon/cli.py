# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""The ``zephon`` command-line entrypoint.

Run ``zephon --help`` (or ``python -m zephon --help``) for the list of
subcommands. Zephon ships ``zephon index``; other installed packages can add
their own subcommands through the ``zephon.cli`` entry-point group.

Plugin contract: the entry point's name is the subcommand name, and its object
is a callable that takes the subcommand's :class:`argparse.ArgumentParser`
(created with ``prog="zephon <name>"`` and the default ``-h/--help``), adds
arguments to it, and sets ``func`` to the handler. That parser is the one that
parses the subcommand's arguments, so anything configured on it applies. The handler receives
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

Running a command loads only that command's plugin; ``zephon --help`` loads
every plugin to list it, so keep heavy imports inside the handler rather than
at module level. Built-in commands take precedence over plugins. A plugin
whose name is already taken, or that fails to load or register, is reported
on stderr and skipped, so one broken install cannot take down the rest of the
CLI.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
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
        _error("--max-workers must be >= 1", prog="zephon index")
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
        _error(str(exc), prog="zephon index")
        return 1
    return 0


@dataclass(frozen=True)
class _Command:
    """A subcommand: how to get its ``register`` callable, and where it came from."""

    load: Callable[[], _Register]
    source: str | None = None  # distribution that provides a plugin command
    summary: str | None = None  # command-list help for built-ins


_BUILTIN_COMMANDS: dict[str, _Command] = {
    "index": _Command(
        lambda: _register_index, summary="build an index.json for a dataset"
    ),
}


def _plugin_entry_points() -> list[EntryPoint]:
    return sorted(entry_points(group=ENTRY_POINT_GROUP), key=lambda ep: ep.name)


def _warn(message: str) -> None:
    print(f"zephon: warning: {message}", file=sys.stderr)


def _error(message: str, prog: str = "zephon") -> None:
    print(f"{prog}: error: {message}", file=sys.stderr)


def _commands() -> dict[str, _Command]:
    """Return built-in and plugin commands by name, without loading any plugin.

    Built-ins take precedence, then the first plugin (by entry-point name order)
    to claim a name; later claimants are reported and skipped.
    """
    commands = dict(_BUILTIN_COMMANDS)
    for ep in _plugin_entry_points():
        source = ep.dist.name if ep.dist is not None else ep.value
        if ep.name in commands:
            _warn(f"skipping plugin command {ep.name!r} from {source}: name in use")
            continue
        commands[ep.name] = _Command(ep.load, source=source)
    return commands


def _configure(name: str, command: _Command) -> argparse.ArgumentParser:
    """Build the parser for ``zephon <name>`` by running its ``register``.

    Raises whatever loading or registering the command raises, including
    :class:`SystemExit` from a ``parser.error()`` call.
    """
    parser = argparse.ArgumentParser(prog=f"zephon {name}")
    command.load()(parser)
    return parser


def _load_failure(name: str, command: _Command, exc: BaseException) -> str:
    return f"failed to load plugin command {name!r} from {command.source}: {exc}"


def _build_parser(commands: dict[str, _Command]) -> argparse.ArgumentParser:
    """Build the top-level parser, used only for help and unknown commands.

    Each plugin is registered into a throwaway parser just to read its
    description. Commands are dispatched to their own parser in :func:`main`,
    so nothing here has to reproduce a plugin's parser configuration.
    """
    parser = argparse.ArgumentParser(
        prog="zephon", description="Zephon command-line tools."
    )
    subparsers = parser.add_subparsers(title="commands", metavar="COMMAND")
    for name, command in commands.items():
        summary = command.summary
        if command.source is not None:
            try:
                summary = _configure(name, command).description
            except (Exception, SystemExit) as exc:
                _warn(_load_failure(name, command, exc))
                continue
        # argparse %-formats help strings; descriptions are plain text.
        help_text = summary.replace("%", "%%") if summary else summary
        subparsers.add_parser(name, help=help_text, add_help=False)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``zephon`` CLI and return its exit code.

    Args:
        argv: Arguments after the program name; defaults to ``sys.argv[1:]``.

    Returns:
        The handler's exit code: 0 on success, nonzero on failure (1 if the
        requested plugin command fails to load). Usage errors raise
        :class:`SystemExit` with code 2, as :mod:`argparse` does.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = _commands()

    command = commands.get(argv[0]) if argv else None
    if command is None:
        # No command, ``--help`` or an unknown command: argparse prints help or
        # a usage error and exits.
        parser = _build_parser(commands)
        parser.parse_args(argv)
        parser.print_help()
        return 2

    name = argv[0]
    try:
        parser = _configure(name, command)
    except (Exception, SystemExit) as exc:
        _error(_load_failure(name, command, exc))
        return 1
    args = parser.parse_args(argv[1:])
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 2
    code = func(args)
    return 0 if code is None else code


__all__ = ["ENTRY_POINT_GROUP", "main"]
