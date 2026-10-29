"""Standalone CPU benchmarks for Zephon PR #70; requires only NumPy and Torch.

Run: python benchmark_stack_sequences.py --json samples.json
Each --snippet NAME prints an independent Python REPL example.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import itertools
import json
import math
import os
import platform
import statistics
import timeit
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

_np = np

BASE_REF = "0c4cc5db2b26517ff1686fa4d32bce9bffddd903"
PRE_TITUS_REF = "c184603e112ae53dc27d126b375d993d93603a3a"
UTILS_FILE = "zephon/_internal/utils/tensor_utils.py"
BASE_STACK_LINES = (72, 105)
POST_STACK_LINES = (72, 125)
BASE_POSITIONS_LINES = (140, 154)
MINH_POSITIONS_LINES = (140, 158)


# zephon/_internal/utils/tensor_utils.py:72-105, base 0c4cc5db2b26517ff1686fa4d32bce9bffddd903
def _stack_base(sequences: list[Any], dtype: Any, framework: str | None) -> Any:
    if framework == "torch":
        import torch

        if sequences and isinstance(sequences[0], torch.Tensor):
            return torch.stack(sequences).to(dtype=dtype)
        if sequences and "numpy" in type(sequences[0]).__module__:
            import numpy as np

            return torch.from_numpy(np.array(sequences)).to(dtype=dtype)
        return torch.tensor(sequences, dtype=dtype)
    elif framework == "numpy":
        import numpy as np

        return np.array(
            [np.asarray(seq, dtype=dtype) for seq in sequences], dtype=dtype
        )
    else:
        return [list(seq) for seq in sequences]


# zephon/_internal/utils/tensor_utils.py:72-125, production post-Titus plus reverse change.
def _stack_post(sequences: list[Any], dtype: Any, framework: str | None) -> Any:
    if framework == "torch":
        import torch

        if sequences and isinstance(sequences[0], torch.Tensor):
            return torch.stack(sequences).to(dtype=dtype)
        if sequences and "numpy" in type(sequences[0]).__module__:
            import numpy as np

            if dtype == torch.int64 and sequences[0].dtype == np.int64:
                return torch.from_numpy(np.array(sequences)).to(dtype=dtype)
            numpy_dtype = {
                torch.bool: np.bool_,
                torch.uint8: np.uint8,
                torch.int8: np.int8,
                torch.int16: np.int16,
                torch.int32: np.int32,
                torch.int64: np.int64,
                torch.float16: np.float16,
                torch.float32: np.float32,
                torch.float64: np.float64,
                torch.complex64: np.complex64,
                torch.complex128: np.complex128,
            }.get(dtype)
            if numpy_dtype is not None and all(
                (
                    isinstance(seq, np.ndarray)
                    and np.can_cast(seq.dtype, numpy_dtype, casting="safe")
                    for seq in sequences
                )
            ):
                return torch.from_numpy(np.array(sequences, dtype=numpy_dtype))
            return torch.from_numpy(np.array(sequences)).to(dtype=dtype)
        return torch.tensor(sequences, dtype=dtype)
    elif framework == "numpy":
        import numpy as np

        return np.array([np.asarray(seq) for seq in sequences], dtype=dtype)
    else:
        return [list(seq) for seq in sequences]


# zephon/_internal/ops/pack_sequences.py:140-154, base 0c4cc5db2b26517ff1686fa4d32bce9bffddd903
def _positions_base(segment_lengths: Sequence[int]) -> _np.ndarray:
    parts = [_np.arange(int(n), dtype=_np.int32) for n in segment_lengths if n > 0]
    if not parts:
        return _np.empty(0, dtype=_np.int32)
    return _np.concatenate(parts)


# zephon/_internal/ops/pack_sequences.py:140-158, original PR #70 c184603e112ae53dc27d126b375d993d93603a3a
def _positions_minh(segment_lengths: Sequence[int]) -> _np.ndarray:
    parts = [_np.arange(int(n), dtype=_np.int64) for n in segment_lengths if n > 0]
    if not parts:
        return _np.empty(0, dtype=_np.int64)
    return _np.concatenate(parts)


@dataclass
class StackOnly:
    """Time the shared stack call with prebuilt inputs."""

    stack: Callable[..., Any]
    rows: list[Any]
    dtype: Any
    framework: str

    def __call__(self) -> Any:
        return self.stack(self.rows, self.dtype, self.framework)


@dataclass
class BuildAndStack(StackOnly):
    """Include the production positions producer in each timed call."""

    builder: Callable[..., Any]
    segments: tuple[int, ...]

    def __call__(self) -> Any:
        rows = [self.builder(self.segments) for _ in self.rows]
        return self.stack(rows, self.dtype, self.framework)


@dataclass(frozen=True)
class Case:
    """Identify a source snapshot and its expected allocation path."""

    name: str
    stack: Callable[..., Any]
    rows: list[Any]
    builder: Callable[..., Any] | None
    direction: str
    path: str


TITLES = {
    "base_forward": "Base Zephon, NumPy -> Torch path in stack_sequences",
    "minh_forward": "PR #70 pre-Titus edit, NumPy -> Torch path in stack_sequences",
    "titus_forward": "PR #70 post-Titus edit, NumPy -> Torch path in stack_sequences",
    "base_reverse": "Base Zephon, Torch -> NumPy path in stack_sequences (reverse)",
    "maxi_reverse": "Maxi suggestion, Torch -> NumPy path in stack_sequences (reverse)",
}


def fixtures(
    batch_size: int, sequence_length: int
) -> tuple[list[Case], tuple[int, ...], Any]:
    """Build equal values with each version's intended storage width."""
    half = sequence_length // 2
    segments = (half, sequence_length - half)
    rows32 = [_positions_base(segments) for _ in range(batch_size)]
    rows64 = [_positions_minh(segments) for _ in range(batch_size)]
    torch_rows = [torch.from_numpy(row) for row in rows32]
    cases = [
        Case(
            "base_forward",
            _stack_base,
            rows32,
            _positions_base,
            "forward",
            "torch_cast",
        ),
        Case(
            "minh_forward",
            _stack_base,
            rows64,
            _positions_minh,
            "forward",
            "wrap_final",
        ),
        Case(
            "titus_forward",
            _stack_post,
            rows32,
            _positions_base,
            "forward",
            "wrap_final",
        ),
        Case("base_reverse", _stack_base, torch_rows, None, "reverse", "row_casts"),
        Case("maxi_reverse", _stack_post, torch_rows, None, "reverse", "row_views"),
    ]
    expected = np.array(
        [list(range(half)) + list(range(sequence_length - half))] * batch_size,
        dtype=np.int64,
    )
    return cases, segments, expected


def operation(case: Case, scope: str, segments: tuple[int, ...]) -> StackOnly:
    """Use one callable type for all versions in a measurement scope."""
    dtype, framework = (
        (torch.long, "torch") if case.direction == "forward" else (np.int64, "numpy")
    )
    if scope == "positions_plus_stack":
        assert case.builder is not None
        return BuildAndStack(
            case.stack, case.rows, dtype, framework, case.builder, segments
        )
    return StackOnly(case.stack, case.rows, dtype, framework)


def verify(case: Case, measured: StackOnly, expected: Any) -> None:
    """Reject incorrect values, source mutation or an unexpected copy path."""
    captures = []
    originals = [
        row.clone() if isinstance(row, torch.Tensor) else row.copy()
        for row in case.rows
    ]
    original_from_numpy, original_asarray = torch.from_numpy, np.asarray

    def capture_torch(array: Any) -> Any:
        result = original_from_numpy(array)
        captures.append((array, result))
        return result

    def capture_numpy(row: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_asarray(row, *args, **kwargs)
        captures.append((row, result))
        return result

    try:
        if case.direction == "forward":
            torch.from_numpy = capture_torch
        else:
            np.asarray = capture_numpy
        result = measured()
    finally:
        torch.from_numpy, np.asarray = original_from_numpy, original_asarray
    values = result.numpy() if isinstance(result, torch.Tensor) else result
    np.testing.assert_array_equal(values, expected)
    assert result.dtype == (torch.long if case.direction == "forward" else np.int64)
    if case.direction == "forward":
        assert len(captures) == 1
        array, wrapped = captures[0]
        assert array.dtype == (np.int32 if case.path == "torch_cast" else np.int64)
        assert (result.data_ptr() == wrapped.data_ptr()) == (case.path == "wrap_final")
    else:
        assert len(captures) == len(case.rows)
        for row, view in captures:
            assert view.dtype == (np.int32 if case.path == "row_views" else np.int64)
            assert (view.ctypes.data == row.data_ptr()) == (case.path == "row_views")
            assert not np.shares_memory(result, view)
    result[0, 0] = -1
    for row, original in zip(case.rows, originals):
        if isinstance(row, torch.Tensor):
            assert torch.equal(row, original)
        else:
            np.testing.assert_array_equal(row, original)


def repl_snippet(name: str) -> str:
    """Return an independent reproduction for one benchmark section."""
    forward = name.endswith("forward")
    stack = _stack_post if name in ("titus_forward", "maxi_reverse") else _stack_base
    builder = _positions_minh if name == "minh_forward" else _positions_base
    line_range = POST_STACK_LINES if stack is _stack_post else BASE_STACK_LINES
    imports = """from __future__ import annotations
import gc
import statistics
import timeit
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any
import numpy as np
import torch
_np = np
"""
    code = (
        imports
        + f"\n# {UTILS_FILE}:{line_range[0]}-{line_range[1]}\n"
        + inspect.getsource(stack)
    )
    code += "\n" + inspect.getsource(StackOnly)
    if forward:
        bounds = (
            MINH_POSITIONS_LINES if name == "minh_forward" else BASE_POSITIONS_LINES
        )
        code += f"\n# zephon/_internal/ops/pack_sequences.py:{bounds[0]}-{bounds[1]}\n"
        code += inspect.getsource(builder) + "\n" + inspect.getsource(BuildAndStack)
        code += f"\nsegments = (2048, 2048)\nrows = [{builder.__name__}(segments) for _ in range(8)]\n"
        code += f"operations = {{'stack only': StackOnly({stack.__name__}, rows, torch.long, 'torch'), 'positions + stack': BuildAndStack({stack.__name__}, rows, torch.long, 'torch', {builder.__name__}, segments)}}\n"
    else:
        code += "\nrows = [torch.tensor(list(range(2048)) * 2, dtype=torch.int32) for _ in range(8)]\n"
        code += f"operations = {{'stack only': StackOnly({stack.__name__}, rows, np.int64, 'numpy')}}\n"
    code += """
# Same historical proxy batch shape: 8 rows x 4096 positions.
# Experiment decision: 18 samples; timeit calibrates its standard 0.2-second block.
for label, measured in operations.items():
    timer = timeit.Timer(measured)
    iterations = timer.autorange()[0]
    timer.timeit(number=iterations)
    samples = []
    for _ in range(18):
        gc.collect()
        samples.append(timer.timeit(number=iterations) / iterations * 1e6)
    print(f"{label}: {statistics.mean(samples):.2f} +/- {statistics.stdev(samples):.2f} microseconds (mean +/- sample SD)")
"""
    return code


def main() -> None:
    """Measure counterbalanced comparisons and retain every sample."""
    parser = argparse.ArgumentParser(description=__doc__)
    # Historical proxy trainer: universe commit 2868701f, settings.py.
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=4096)
    # Three repetitions of all six forward orders; reverse uses all two orders nine times.
    parser.add_argument("--samples", type=int, default=18)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--snippet", choices=tuple(TITLES))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.snippet:
        print(repl_snippet(args.snippet))
        return
    assert args.batch_size > 0 and args.sequence_length >= 2
    assert args.samples > 1 and args.samples % math.factorial(3) == 0
    cases, segments, expected = fixtures(args.batch_size, args.sequence_length)
    by_name = {case.name: case for case in cases}
    groups = {
        "forward_stack_only": (
            [case for case in cases if case.direction == "forward"],
            "stack_only",
        ),
        "forward_positions_plus_stack": (
            [case for case in cases if case.direction == "forward"],
            "positions_plus_stack",
        ),
        "reverse_stack_only": (
            [case for case in cases if case.direction == "reverse"],
            "stack_only",
        ),
    }
    for group, scope in groups.values():
        for case in group:
            verify(case, operation(case, scope, segments), expected)
    for name in ("titus_forward", "maxi_reverse"):
        mutant = replace(by_name[name], stack=_stack_base)
        try:
            verify(mutant, operation(mutant, "stack_only", segments), expected)
        except AssertionError:
            pass
        else:
            raise AssertionError(
                "Allocation gate accepted the old implementation: " + name
            )
    report: dict[str, Any] = {
        "metadata": {
            "host": platform.node(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "shape": [args.batch_size, args.sequence_length],
            "torch_threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "samples_per_case": args.samples,
            "load_before": os.getloadavg(),
            "base_ref": BASE_REF,
            "pre_titus_ref": PRE_TITUS_REF,
            "policy": "warm, equal iterations within each group; all orders balanced; timeit GC disabled equally",
        },
        "verification": "exact values/dtypes, independent output, allocation paths and old-code mutants passed",
        "samples": [],
        "summary": {},
    }

    def save() -> None:
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            temp = args.json.with_suffix(".tmp")
            temp.write_text(json.dumps(report, indent=2) + "\n")
            temp.replace(args.json)

    save()
    print(report["verification"], flush=True)
    if args.verify_only:
        return
    threads = (torch.get_num_threads(), torch.get_num_interop_threads())
    for group_name, (group, scope) in groups.items():
        timers = {
            case.name: timeit.Timer(operation(case, scope, segments)) for case in group
        }
        iterations = max(timer.autorange()[0] for timer in timers.values())
        for timer in timers.values():
            timer.timeit(number=iterations)
        orders = list(itertools.permutations(case.name for case in group))
        for cycle in range(args.samples // len(orders)):
            for order_index, order in enumerate(orders):
                for position, name in enumerate(order):
                    assert threads == (
                        torch.get_num_threads(),
                        torch.get_num_interop_threads(),
                    )
                    gc.collect()
                    elapsed = timers[name].timeit(number=iterations)
                    report["samples"].append(
                        {
                            "group": group_name,
                            "case": name,
                            "cycle": cycle,
                            "order_index": order_index,
                            "position": position,
                            "order": order,
                            "iterations": iterations,
                            "seconds_per_call": elapsed / iterations,
                        }
                    )
                save()
            print(f"{group_name}: completed balanced cycle {cycle + 1}", flush=True)
        report["summary"][group_name] = {}
        for case in group:
            verify(case, operation(case, scope, segments), expected)
            samples = [
                item["seconds_per_call"] * 1e6
                for item in report["samples"]
                if item["group"] == group_name and item["case"] == case.name
            ]
            stats = {
                "mean_us": statistics.mean(samples),
                "stdev_us": statistics.stdev(samples),
                "median_us": statistics.median(samples),
                "min_us": min(samples),
                "max_us": max(samples),
            }
            report["summary"][group_name][case.name] = stats
            print(
                f"{group_name} / {case.name}: {stats['mean_us']:.2f} +/- {stats['stdev_us']:.2f} microseconds",
                flush=True,
            )
        save()
    report["metadata"]["load_after"] = os.getloadavg()
    report["complete"] = True
    save()


if __name__ == "__main__":
    main()
