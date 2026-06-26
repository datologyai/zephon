# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Validation harness for user-supplied operators and accumulators.

The framework guarantees deterministic parallel execution only when the
callables passed to ``Pipeline.add_op`` honor several contracts that the
type system cannot enforce: ``process_many`` must be stateless across
calls, the accumulator must drain on ``flush()`` and reset on
``flush(reset=True)``, and so on.  This module runs a synthetic-sample
smoke test that catches the most common violations before the engine
starts.  See the "Accumulators and Operators" page in the published
documentation for the contracts being checked.

The harness is intentionally a smoke test, not a proof: it cannot catch
violations that only surface after many iterations or under specific data
shapes.  Errors here are real bugs; passing this harness is necessary but
not sufficient for correctness.

Unlike the rest of zephon (custom exception subclasses + ``warnings.warn``),
this module aggregates findings into a :class:`ValidationReport` of
string-coded :class:`Issue` records.  Aggregation lets the user see every
problem in their op at once; the tradeoff is identification by string
equality on ``issue.code`` rather than by ``isinstance``.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import inspect
import pickle
import textwrap
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

from zephon.core.accumulators import Accumulator
from zephon.core.constants import SampleBatch, SampleMeta, SampleRecord

if TYPE_CHECKING:
    from zephon.api.pipeline import Pipeline


_DOC_LINK = (
    "https://datologyai.github.io/zephon/understanding/accumulators_operators.html"
)


@dataclass(frozen=True)
class Issue:
    """A single validation finding."""

    severity: str  # "error" | "warning"
    code: str
    op_name: str
    message: str
    doc_link: str = _DOC_LINK


@dataclass
class ValidationReport:
    """Result of running the validation harness against a pipeline."""

    issues: list[Issue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when no errors were reported (warnings are allowed)."""
        return not any(i.severity == "error" for i in self.issues)

    @property
    def errors(self) -> list[Issue]:
        """Subset of issues with severity ``error``."""
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[Issue]:
        """Subset of issues with severity ``warning``."""
        return [i for i in self.issues if i.severity == "warning"]

    def format(self) -> str:
        """Human-readable multi-line summary suitable for raising."""
        if not self.issues:
            return "ValidationReport: ok (no user ops to validate or all checks passed)"
        lines = [
            f"ValidationReport: {len(self.errors)} error(s), {len(self.warnings)} warning(s)"
        ]
        for issue in self.issues:
            lines.append(
                f"  [{issue.severity}] {issue.code} in {issue.op_name!r}: {issue.message}"
            )
            lines.append(f"      → {issue.doc_link}")
        return "\n".join(lines)


class ValidationError(RuntimeError):
    """Raised when ``Pipeline.validate(strict=True)`` finds at least one error."""

    def __init__(self, report: ValidationReport) -> None:
        super().__init__(report.format())
        self.report = report


# ---------------------------------------------------------------------------
# Synthetic sample generation
# ---------------------------------------------------------------------------


def _synthesize_records(*, n_lanes: int = 3, per_lane: int = 4) -> list[SampleRecord]:
    """Build ``n_lanes * per_lane`` synthetic ``SampleRecord`` instances.

    Lane ids are ``[0, n_lanes)``, with monotone ``chunk_id`` and
    ``chunk_offset`` per lane.  Sample ids conform to the standard
    ``SampleId = (DatasetId, ShardId, LocalSampleId)`` triple
    (``(0, lane, offset)``) so ops that unpack ``meta.sample_id`` into its
    three components validate instead of tripping on a malformed tuple.
    Payloads are generic dicts; kwargs-form
    ops whose ``process_many`` requires a specific payload shape (e.g.
    tokenized text) will surface a loud ``OP_REJECTS_GENERIC_PAYLOAD``
    warning and run unvalidated.  Instance-form `BaseOp` subclasses
    without an overridden ``validation_samples()`` skip the runtime
    probes entirely via ``OP_INSTANCE_RUNTIME_CHECKS_SKIPPED`` — the
    synthetic records are never fed to them.

    Synthetic inputs are always ``SampleRecord``; ``StreamItem`` also
    permits ``SampleBatch``, but constructing a meaningful synthetic
    batch shape for an arbitrary user op is out of scope.  Ops taking
    ``SampleBatch`` input thus degrade through the same payload-shape
    paths above.  Outputs *are* allowed to be ``SampleBatch`` (e.g.
    anything downstream of ``Pipeline.batch``): see
    :func:`_iter_records` for the flattening the per-record checks rely
    on.
    """
    records: list[SampleRecord] = []
    for lane in range(n_lanes):
        for offset in range(per_lane):
            meta = SampleMeta(
                sample_id=(0, lane, offset),
                lane_id=lane,
                chunk_id=offset // 2,
                chunk_offset=offset,
            )
            payload = {"text": f"sample_{lane}_{offset}", "value": offset}
            records.append(SampleRecord(meta=meta, payload=payload))
    return records


def _iter_records(items: list[Any]) -> list[SampleRecord]:
    """Flatten ``list[StreamItem]`` into the underlying records.

    ``process_many`` may return ``SampleRecord`` instances directly or
    ``SampleBatch`` wrappers (e.g. ops downstream of
    :py:meth:`~zephon.api.Pipeline.batch`).  ``SampleBatch`` exposes
    ``.records`` rather than ``.meta``, so the per-record sample-identity
    and lineage checks below would ``AttributeError`` without this
    expansion.  The per-record invariants the validator enforces
    (sample_id preservation, lineage rules) still apply inside batched
    outputs, so flattening is the right semantic: the rule is checked
    against every record the op emits regardless of how they're
    grouped.
    """
    out: list[SampleRecord] = []
    for item in items:
        if isinstance(item, SampleBatch):
            out.extend(item.records)
        else:
            out.append(item)
    return out


def _ids(records: list[Any]) -> list[Any]:
    return [r.meta.sample_id for r in _iter_records(records)]


def _interleave_by_lane(records: list[SampleRecord]) -> list[SampleRecord]:
    """Round-robin ``records`` across their lanes.

    Groups by ``lane_id`` (preserving first-seen lane order and intra-lane
    order) then emits one record per lane per round, so a single push spans
    lanes in interleaved order regardless of the input ordering.  Used by the
    lane-purity probe to force per-call bucketing to route across lanes.
    """
    by_lane: dict[Any, list[SampleRecord]] = {}
    for r in records:
        by_lane.setdefault(r.meta.lane_id, []).append(r)
    groups = list(by_lane.values())
    out: list[SampleRecord] = []
    for i in range(max((len(g) for g in groups), default=0)):
        for g in groups:
            if i < len(g):
                out.append(g[i])
    return out


def _first_non_stream_item(out: Any) -> tuple[Any, str] | None:
    """Locate the first output element that is not a well-formed ``StreamItem``.

    ``process_many`` must return a list of :class:`SampleRecord` /
    :class:`SampleBatch`, and every record inside a ``SampleBatch`` must be a
    :class:`SampleRecord` (the engine enforces this at runtime — see
    :mod:`zephon.core.notify`).  The op-level checks reach into
    ``record.meta`` via :func:`_iter_records` / :func:`_ids`, so any other
    shape would ``AttributeError`` there with a confusing traceback.

    Returns ``(offending_value, location)`` for the first bad element, or
    ``None`` when every element is a valid ``StreamItem``.
    """
    if not isinstance(out, list):
        return out, "the process_many return value (expected a list)"
    for i, item in enumerate(out):
        if isinstance(item, SampleRecord):
            continue
        if isinstance(item, SampleBatch):
            for j, rec in enumerate(item.records):
                if not isinstance(rec, SampleRecord):  # pyright: ignore[reportUnnecessaryIsInstance]
                    return rec, f"output[{i}].records[{j}]"
            continue
        return item, f"output[{i}]"
    return None


def _drain_batches(batches: list[Any]) -> list[Any]:
    """Flatten ``[(batch, wait_ns), ...]`` into a single element list."""
    out: list[Any] = []
    for batch, _wait in batches:
        out.extend(batch)
    return out


def _outputs_equivalent(out_a: list[Any], out_b: list[Any]) -> bool:
    """Best-effort structural equality across two ``process_many`` outputs.

    Compares record count and sample IDs first (cheap, catches most bugs),
    then falls back to ``pickle.dumps`` for full payload equality.  Returns
    True if the payloads cannot be pickled — payload-level drift is then
    invisible to the harness.

    Both outputs may mix ``SampleRecord`` and ``SampleBatch``; ID
    extraction routes through :func:`_iter_records` so the cheap-path
    check works regardless of how records are grouped.
    """
    if len(out_a) != len(out_b):
        return False
    if _ids(out_a) != _ids(out_b):
        return False
    try:
        return pickle.dumps(out_a) == pickle.dumps(out_b)
    except Exception:
        return True


def _lineage_extends_or_equals(
    out_lin: tuple[Any, ...], in_lin: tuple[Any, ...]
) -> bool:
    """True if ``out_lin`` equals ``in_lin`` or extends it (``in_lin`` is a prefix)."""
    if len(out_lin) < len(in_lin):
        return False
    return out_lin[: len(in_lin)] == in_lin


def _resolve_validation_records(
    op: Any, op_name: str
) -> tuple[list[SampleRecord], bool, list[Issue]]:
    """Pick the records to probe ``op`` with.

    Calls ``op.validation_samples()`` if the op exposes one.  When it
    returns a non-empty list of :class:`SampleRecord`, those records are
    used for every op-level check below.  ``None`` / missing falls back
    to the validator's synthetic records silently.  Anything else
    (raises, wrong type, empty list) surfaces
    ``OP_VALIDATION_SAMPLES_FACTORY_FAILED`` and also falls back — the
    validator never dies on a buggy factory.

    Returns ``(records, user_provided, issues)``.  ``user_provided`` is
    ``True`` only when the factory returned a valid non-empty batch the
    validator could adopt; it is the gate that callers use to decide
    whether to run runtime probes against instance-form ops (whose
    ``setup``-built state we don't establish).
    """
    issues: list[Issue] = []
    factory = getattr(op, "validation_samples", None)
    if not callable(factory):
        return _synthesize_records(), False, issues
    try:
        user_records: Any = factory()
    except Exception as exc:  # noqa: BLE001 — user code
        issues.append(
            Issue(
                severity="warning",
                code="OP_VALIDATION_SAMPLES_FACTORY_FAILED",
                op_name=op_name,
                message=(
                    f"`validation_samples()` raised "
                    f"{type(exc).__name__}: {exc}. Falling back to the "
                    "validator's synthetic records — this op may trip "
                    "OP_REJECTS_GENERIC_PAYLOAD as a result."
                ),
            )
        )
        return _synthesize_records(), False, issues
    if user_records is None:
        return _synthesize_records(), False, issues
    if (
        not isinstance(user_records, list)
        or not user_records
        or not all(isinstance(r, SampleRecord) for r in user_records)
    ):
        issues.append(
            Issue(
                severity="warning",
                code="OP_VALIDATION_SAMPLES_FACTORY_FAILED",
                op_name=op_name,
                message=(
                    "`validation_samples()` must return a non-empty list of "
                    f"SampleRecord; got {type(user_records).__name__}. "
                    "Falling back to synthetic records."
                ),
            )
        )
        return _synthesize_records(), False, issues
    return user_records, True, issues


# ---------------------------------------------------------------------------
# Op-level checks
# ---------------------------------------------------------------------------


def _check_op_determinism(
    op: Any, op_name: str, samples: list[SampleRecord]
) -> list[Issue]:
    """``process_many(elems) == process_many(elems)`` must hold across calls.

    Each call is wrapped independently so that divergent exception behaviour
    (succeeds on one call, raises on another) is reported as a determinism
    failure, not as a payload-shape warning.
    """
    results: list[tuple[list[Any] | None, BaseException | None]] = []
    for _ in range(2):
        try:
            out = op.process_many(copy.deepcopy(samples))
            results.append((out, None))
        except BaseException as exc:  # noqa: BLE001 — we re-classify below
            results.append((None, exc))

    (out_a, exc_a), (out_b, exc_b) = results

    # Case 1: exactly one call raised.  That is itself a determinism violation.
    if (exc_a is None) != (exc_b is None):
        raised = exc_a if exc_a is not None else exc_b
        assert raised is not None
        return [
            Issue(
                severity="error",
                code="OP_NONDETERMINISTIC",
                op_name=op_name,
                message=(
                    f"process_many succeeded on one call and raised "
                    f"{type(raised).__name__}({raised}) on the other for "
                    f"identical input. This is a determinism violation, "
                    f"typically caused by hidden state or module-global RNG."
                ),
            )
        ]

    # Case 2: both calls raised.
    #
    # A *different* exception across two identical calls is positive evidence of
    # nondeterminism → blocking error.
    #
    # The *same* exception twice means the op consistently rejects the
    # validator's synthetic input.  We deliberately do NOT try to classify this
    # as "needs a payload shape" vs "just broken": the validator only ever feeds
    # one input shape and never sees the op succeed, so it has no basis to tell
    # them apart, and the exception type is a leaky proxy (a payload-shape
    # mismatch can surface as KeyError, ValueError, struct.error, ...).  A
    # consistent raise is *absence of evidence* (no op-level check could run),
    # not evidence of a bug, so we never block iteration on it — we emit a loud
    # NOT-VALIDATED warning pointing at the validation_samples escape hatch.
    if exc_a is not None and exc_b is not None:
        same = type(exc_a) is type(exc_b) and str(exc_a) == str(exc_b)
        if not same:
            return [
                Issue(
                    severity="error",
                    code="OP_NONDETERMINISTIC",
                    op_name=op_name,
                    message=(
                        f"process_many raised differently across two identical "
                        f"calls: {type(exc_a).__name__}({exc_a}) vs "
                        f"{type(exc_b).__name__}({exc_b})."
                    ),
                )
            ]
        return [
            Issue(
                severity="warning",
                code="OP_REJECTS_GENERIC_PAYLOAD",
                op_name=op_name,
                message=(
                    f"!! THIS OP WAS NOT VALIDATED !! "
                    f"process_many consistently raised {type(exc_a).__name__} "
                    f"on the validator's synthetic input: {exc_a}. The validator "
                    f"defaults to generic dict payloads "
                    f"({{'text': str, 'value': int}}); ops that need a specific "
                    f"payload shape or content cannot be auto-validated unless "
                    f"they supply realistic records. None of the op-level checks "
                    f"ran for this op: determinism, cross-call state, "
                    f"statelessness, and sample identity were all skipped. For "
                    f"better coverage, supply `validation_samples=lambda: [...]` "
                    f"on `add_op` (kwargs form) or override "
                    f"`validation_samples()` on your `BaseOp` subclass; either "
                    f"should return a small batch of `SampleRecord` instances "
                    f"with the payload shape your op expects."
                ),
            )
        ]

    # Case 3: both calls succeeded.  Compare structurally including payloads.
    assert out_a is not None and out_b is not None
    if not _outputs_equivalent(out_a, out_b):
        return [
            Issue(
                severity="error",
                code="OP_NONDETERMINISTIC",
                op_name=op_name,
                message=(
                    "process_many produced different outputs across two "
                    "identical calls (sample IDs or payloads diverged). "
                    "Operators must be stateless: mutable instance "
                    "attributes, module-level RNG, or closure state captured "
                    "by reference all break determinism. Move cross-"
                    "invocation state into the accumulator."
                ),
            )
        ]
    return []


def _check_op_output_types(
    op: Any, op_name: str, samples: list[SampleRecord]
) -> list[Issue]:
    """``process_many`` must emit ``SampleRecord`` / ``SampleBatch`` elements.

    The engine raises ``TypeError("Unsupported element type ...; expected
    SampleBatch or SampleRecord")`` for anything else (plain dicts, lists,
    tuples).  This check surfaces that contract violation as a clean Issue at
    validation time, and lets :func:`validate_pipeline` skip the structural
    checks (determinism, cross-call state, sample identity) that reach into
    ``record.meta`` and would otherwise crash with an ``AttributeError``.
    """
    try:
        out = op.process_many(copy.deepcopy(samples))
    except Exception:
        return []  # surfaced by the determinism check
    bad = _first_non_stream_item(out)
    if bad is None:
        return []
    value, where = bad
    return [
        Issue(
            severity="error",
            code="OP_OUTPUT_NOT_STREAM_ITEM",
            op_name=op_name,
            message=(
                f"process_many returned {type(value).__name__} at {where}; "
                f"expected SampleBatch or SampleRecord. The engine rejects "
                f"non-StreamItem output at runtime. Map payloads in place and "
                f"return the SampleRecord they belong to (use "
                f"SampleMeta.child(i) for fan-out) instead of returning bare "
                f"payloads."
            ),
        )
    ]


def _check_op_no_cross_call_state(
    op: Any, op_name: str, samples: list[SampleRecord]
) -> list[Issue]:
    """``process_many`` must not carry state across calls with distinct inputs.

    The basic determinism check probes two identical calls, which misses
    state that only surfaces when *different* inputs flow through the op
    in sequence (e.g. caching the previous batch and leaking it into the
    next one).  This check uses an A-B-A pattern: call with input A, then
    with B, then with A again, and compare the two A outputs.  If they
    differ, the op carried state derived from B into the second A call.
    """
    samples_a = copy.deepcopy(samples)
    samples_b = copy.deepcopy(samples)
    # Disambiguate samples_b IDs so the op sees a distinct second input.
    # Bump the leading (DatasetId) component so the id stays a valid SampleId
    # triple; ``*sid[1:]`` keeps this robust to user-supplied records of any
    # tuple arity rather than crashing on the unpack.
    for r in samples_b:
        sid = r.meta.sample_id
        r.meta = dataclasses.replace(r.meta, sample_id=(sid[0] + 1, *sid[1:]))

    try:
        out_a1 = op.process_many(copy.deepcopy(samples_a))
        op.process_many(copy.deepcopy(samples_b))
        out_a2 = op.process_many(copy.deepcopy(samples_a))
    except Exception:
        return []  # surfaced by determinism check

    if not _outputs_equivalent(out_a1, out_a2):
        return [
            Issue(
                severity="error",
                code="OP_CROSS_CALL_STATE",
                op_name=op_name,
                message=(
                    "process_many produced different outputs for the same "
                    "input depending on what input preceded it. This "
                    "indicates state being carried between invocations — "
                    "cross-invocation state belongs in the accumulator, "
                    "not on the op."
                ),
            )
        ]
    return []


def _check_op_self_mutation(
    op: Any, op_name: str, samples: list[SampleRecord]
) -> list[Issue]:
    """Op ``__dict__`` should not change as a result of running ``process_many``.

    Warning only: a tokenizer with an internal cache may legitimately mutate
    its own state without breaking determinism.  This signals a *suspicious*
    pattern, not a definite bug.

    The exception catch is narrow — we silence pickle-related failures (some
    legitimate ops are unpicklable) but let everything else propagate so
    other checks can see them.
    """
    try:
        before = pickle.dumps(op)
    except (pickle.PicklingError, TypeError, AttributeError):
        return []  # unpicklable op — fall back is just not running this check
    try:
        op.process_many(copy.deepcopy(samples))
    except Exception:
        return []  # process_many raised — surfaced by determinism check
    try:
        after = pickle.dumps(op)
    except (pickle.PicklingError, TypeError, AttributeError):
        return []
    if before == after:
        return []
    return [
        Issue(
            severity="warning",
            code="OP_MUTATES_SELF",
            op_name=op_name,
            message=(
                "Op state changed during process_many. If you are caching a "
                "read-only resource (e.g. tokenizer warmup), this is fine. "
                "If you are tracking samples or maintaining a counter, move "
                "it into the accumulator."
            ),
        )
    ]


# ---------------------------------------------------------------------------
# Static-analysis helpers (heuristics that run even when the synthetic-payload
# runtime probe is skipped, e.g. when the op rejects generic payloads).
# ---------------------------------------------------------------------------


def _user_callables(op: Any) -> list[tuple[str, Callable[..., Any]]]:
    """Return the user-supplied callables on an op for AST/cloudpickle inspection.

    For ops added via the kwargs form of ``Pipeline.add_op`` (i.e.
    ``_FunctionalOp`` wrappers), the inner ``_process_many_fn`` /
    ``_process_one_fn`` attributes hold the user's actual callables.
    For user `BaseOp` subclasses attached via the instance form, the
    bound methods on the op carry the user's source.
    """
    pm_fn = getattr(op, "_process_many_fn", None)
    if pm_fn is not None:
        out: list[tuple[str, Callable[..., Any]]] = [("process_many", pm_fn)]
        po_fn = getattr(op, "_process_one_fn", None)
        if po_fn is not None:
            out.append(("process_one", po_fn))
        return out
    out = []
    if hasattr(op, "process_many"):
        out.append(("process_many", op.process_many))
    if hasattr(op, "process_one"):
        out.append(("process_one", op.process_one))
    return out


def _get_ast(fn: Callable[..., Any]) -> Optional[ast.AST]:
    """Best-effort AST for a callable; returns None for anything we can't read."""
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        return None
    try:
        return ast.parse(textwrap.dedent(src))
    except SyntaxError:
        return None


class _SelfMutationScanner(ast.NodeVisitor):
    """AST walk: collect statements that mutate ``self.<attr>`` in a method body.

    Catches the canonical "I dropped a counter on the op" pattern even when
    the synthetic-payload runtime probe is skipped.  No-op on free
    functions (no ``self`` in scope).
    """

    MUTATORS = frozenset(
        {
            "append",
            "extend",
            "insert",
            "pop",
            "remove",
            "clear",
            "update",
            "setdefault",
            "add",
            "discard",
            "sort",
            "__setitem__",
            "__delitem__",
        }
    )

    def __init__(self) -> None:
        self.hits: list[tuple[int, str]] = []

    @staticmethod
    def _is_self_attr(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        )

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if self._is_self_attr(target):
                assert isinstance(target, ast.Attribute)
                self.hits.append((node.lineno, f"self.{target.attr} = ..."))
            elif isinstance(target, ast.Subscript) and self._is_self_attr(target.value):
                assert isinstance(target.value, ast.Attribute)
                self.hits.append((node.lineno, f"self.{target.value.attr}[...] = ..."))
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if self._is_self_attr(node.target):
            assert isinstance(node.target, ast.Attribute)
            self.hits.append((node.lineno, f"self.{node.target.attr} <op>= ..."))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        f = node.func
        if (
            isinstance(f, ast.Attribute)
            and f.attr in self.MUTATORS
            and self._is_self_attr(f.value)
        ):
            assert isinstance(f.value, ast.Attribute)
            self.hits.append((node.lineno, f"self.{f.value.attr}.{f.attr}(...)"))
        self.generic_visit(node)


# Known non-deterministic stdlib calls that break replay when used inside
# ``process_many``.  Format: ``module.attr`` (one level deep is enough for
# the common offenders).
_NONDETERMINISTIC_STDLIB_CALLS = frozenset(
    {
        "random.random",
        "random.randint",
        "random.choice",
        "random.choices",
        "random.sample",
        "random.shuffle",
        "random.uniform",
        "random.gauss",
        "time.time",
        "time.monotonic",
        "time.perf_counter",
        "uuid.uuid1",
        "uuid.uuid4",
        "os.urandom",
        "secrets.token_bytes",
        "secrets.token_hex",
    }
)


class _NonDeterministicStdlibScanner(ast.NodeVisitor):
    """AST walk: flag calls to known non-deterministic stdlib helpers.

    Catches the common "random.random() inside a transform" footgun
    statically — useful because the runtime determinism check only fires
    when outputs diverge, which a dead/unused call wouldn't trigger.

    Only matches qualified ``module.attr(...)`` calls.  ``from random
    import random; random()`` produces ``Call(func=Name("random"))`` and
    is not flagged — see the warning text emitted by
    :func:`_check_op_uses_nondeterministic_stdlib`.
    """

    def __init__(self) -> None:
        self.hits: list[tuple[int, str]] = []

    def visit_Call(self, node: ast.Call) -> None:
        f = node.func
        if (
            isinstance(f, ast.Attribute)
            and isinstance(f.value, ast.Name)
            and f"{f.value.id}.{f.attr}" in _NONDETERMINISTIC_STDLIB_CALLS
        ):
            self.hits.append((node.lineno, f"{f.value.id}.{f.attr}(...)"))
        self.generic_visit(node)


def _format_hits(hits: list[tuple[int, str]], limit: int = 5) -> str:
    """Compact, deterministic rendering of scanner hits."""
    head = hits[:limit]
    rendered = ", ".join(f"line {ln}: {snippet}" for ln, snippet in head)
    if len(hits) > limit:
        rendered += f" (+{len(hits) - limit} more)"
    return rendered


def _check_op_writes_to_self(op: Any, op_name: str) -> list[Issue]:
    """Static AST scan flagging ``self.<attr>`` writes in user callables.

    Warning-level heuristic that runs unconditionally — does not depend
    on whether the op accepts synthetic payloads.  Fires on `BaseOp`
    subclass methods attached via the instance form of ``add_op`` and on
    bound methods threaded through the kwargs form.  No-op for free
    functions (no ``self`` in scope) and for callables whose source
    cannot be read (lambdas defined in REPLs, C functions, etc.).
    """
    issues: list[Issue] = []
    for label, fn in _user_callables(op):
        tree = _get_ast(fn)
        if tree is None:
            continue
        scanner = _SelfMutationScanner()
        scanner.visit(tree)
        if not scanner.hits:
            continue
        issues.append(
            Issue(
                severity="warning",
                code="OP_WRITES_TO_SELF",
                op_name=op_name,
                message=(
                    f"{label} writes to self ({_format_hits(scanner.hits)}). "
                    "Read-only resource setup belongs in __init__/setup; "
                    "cross-batch state belongs in the accumulator. If this "
                    "is benign (e.g. memoising a derived constant on first "
                    "call), ignore the warning."
                ),
            )
        )
    return issues


def _check_op_uses_nondeterministic_stdlib(op: Any, op_name: str) -> list[Issue]:
    """Static AST scan flagging known non-deterministic stdlib calls.

    Warning-level heuristic that runs unconditionally.  Catches the
    dead-write case (uses ``random.random`` but discards the value)
    that the runtime determinism check would miss, and produces a
    faster signal for the live case.
    """
    issues: list[Issue] = []
    for label, fn in _user_callables(op):
        tree = _get_ast(fn)
        if tree is None:
            continue
        scanner = _NonDeterministicStdlibScanner()
        scanner.visit(tree)
        if not scanner.hits:
            continue
        issues.append(
            Issue(
                severity="warning",
                code="OP_NONDETERMINISTIC_STDLIB_CALL",
                op_name=op_name,
                message=(
                    f"{label} calls non-deterministic stdlib helper(s) "
                    f"({_format_hits(scanner.hits)}). These break replay; "
                    "seed RNG explicitly (e.g. derive from sample IDs or "
                    "thread a seed through __init__) and avoid wall-clock "
                    "time inside the transform. Note: only qualified "
                    "`module.attr(...)` calls are detected; bare-name "
                    "forms like `from random import random; random()` "
                    "will not be flagged here — the runtime determinism "
                    "check is the authoritative backstop for those."
                ),
            )
        )
    return issues


def _try_cloudpickle(obj: Any) -> Optional[bytes]:
    """Cloudpickle dumps with a broad except — None signals "couldn't snapshot."""
    try:
        import cloudpickle  # local import: heavy-ish, lazy load
    except ImportError:
        return None
    try:
        return cloudpickle.dumps(obj)
    except Exception:
        return None


def _check_op_state_diff(
    op: Any, op_name: str, samples: list[SampleRecord]
) -> list[Issue]:
    """Cloudpickle-based state diff catching closure mutations.

    Complements :func:`_check_op_self_mutation` (which uses plain pickle and
    bails when the op isn't picklable).  Cloudpickle traverses closure
    cells, so this catches the canonical "closure-captured dict mutated
    between calls" pattern in `add_op` kwargs-form ops — even when output
    happens to look right on a single call.
    """
    targets: list[tuple[str, Any]] = [("op", op)]
    for label, fn in _user_callables(op):
        if getattr(fn, "__closure__", None):
            targets.append((f"{label} closure", fn))

    before: dict[str, Optional[bytes]] = {k: _try_cloudpickle(v) for k, v in targets}
    if all(v is None for v in before.values()):
        return []  # nothing we can snapshot

    try:
        op.process_many(copy.deepcopy(samples))
    except Exception:
        return []  # synthetic-payload rejections are owned by other checks

    after: dict[str, Optional[bytes]] = {k: _try_cloudpickle(v) for k, v in targets}
    changed = [
        k
        for k in before
        if before[k] is not None and after[k] is not None and before[k] != after[k]
    ]
    if not changed:
        return []
    return [
        Issue(
            severity="warning",
            code="OP_STATE_CHANGED_DURING_CALL",
            op_name=op_name,
            message=(
                f"After one process_many call, the following changed: "
                f"{', '.join(changed)}. A tokenizer warming an internal "
                "cache would also trigger this; if you're tracking samples "
                "across batches, move that state into the accumulator."
            ),
        )
    ]


def _check_op_sample_identity(
    op: Any, op_name: str, samples: list[SampleRecord]
) -> list[Issue]:
    """Outputs must preserve ``sample_id`` and use ``SampleMeta.child()`` for fan-out.

    Three failure modes are flagged:

    - ``OP_FABRICATES_SAMPLE_ID``: an output's ``sample_id`` is not in the
      input set.  Operators may filter, map, or fan out — they may not
      invent new IDs.
    - ``OP_LINEAGE_NOT_VIA_CHILD``: an output keeps its input's
      ``sample_id`` but its ``lineage`` does not equal or extend the
      input's.  Fan-out must use ``SampleMeta.child(i)`` which appends
      to the lineage tuple.
    - ``OP_FANOUT_DUPLICATE_LINEAGE``: multiple outputs for the same
      ``sample_id`` carry identical lineages.  Each fan-out child must
      have a distinct ``child(i)`` index so cursors stay unique.

    These invariants protect the engine's eviction and replay machinery
    (see ``zephon/core/engine.py`` and ``op_base.py:32-36``).

    Outputs may be a mix of ``SampleRecord`` and ``SampleBatch`` (e.g.
    when the op is downstream of ``Pipeline.batch``); we flatten via
    :func:`_iter_records` before iterating so the per-record checks
    apply across batched outputs without crashing on ``.meta`` access.
    """
    input_meta_by_id = {r.meta.sample_id: r.meta for r in samples}

    try:
        out = op.process_many(copy.deepcopy(samples))
    except Exception:
        return []  # surfaced by determinism check

    issues: list[Issue] = []
    out_lineages_by_id: dict[Any, list[tuple[Any, ...]]] = {}

    # Ops downstream of ``batch`` emit ``SampleBatch`` wrappers — flatten
    # so the per-record invariants below see every emitted record.
    for record in _iter_records(out):
        sid = record.meta.sample_id
        if sid not in input_meta_by_id:
            issues.append(
                Issue(
                    severity="error",
                    code="OP_FABRICATES_SAMPLE_ID",
                    op_name=op_name,
                    message=(
                        f"Output sample_id {sid!r} was not in the input set. "
                        f"Operators must not invent new sample_ids — fan-out "
                        f"uses SampleMeta.child() which preserves sample_id "
                        f"and extends lineage."
                    ),
                )
            )
            continue
        out_lineages_by_id.setdefault(sid, []).append(tuple(record.meta.lineage))

    for sid, lineages in out_lineages_by_id.items():
        input_lin = tuple(input_meta_by_id[sid].lineage)
        for out_lin in lineages:
            if not _lineage_extends_or_equals(out_lin, input_lin):
                issues.append(
                    Issue(
                        severity="error",
                        code="OP_LINEAGE_NOT_VIA_CHILD",
                        op_name=op_name,
                        message=(
                            f"Sample {sid!r}: output lineage {out_lin} does "
                            f"not extend input lineage {input_lin}. Use "
                            f"SampleMeta.child(i) to derive fan-out lineages."
                        ),
                    )
                )
        if len(lineages) > 1 and len(set(lineages)) != len(lineages):
            duplicate = next(lin for lin in lineages if lineages.count(lin) > 1)
            issues.append(
                Issue(
                    severity="error",
                    code="OP_FANOUT_DUPLICATE_LINEAGE",
                    op_name=op_name,
                    message=(
                        f"Sample {sid!r} appears {len(lineages)} times in "
                        f"output with duplicate lineage {duplicate}. Fan-out "
                        f"must use SampleMeta.child(i) with distinct i so "
                        f"every emitted record has a unique cursor."
                    ),
                )
            )
    return issues


# ---------------------------------------------------------------------------
# Accumulator-level checks
# ---------------------------------------------------------------------------


def _make_accumulator_factory(op: Any) -> Callable[[], Accumulator[Any]]:
    """Return a 0-arg factory that produces a fresh accumulator from ``op``.

    Goes through ``op.accumulator(deterministic=True, ctx={})`` so the
    factory-signature detection logic inside the op wrapper is exercised
    end-to-end.  Hardcodes ``deterministic=True`` to validate the strict-
    mode branch; latency-based flushing is inherently non-deterministic
    and can't be tested by this harness anyway.
    """
    return lambda: op.accumulator(deterministic=True, ctx={})


def _check_acc_flush_conservation(
    factory: Callable[[], Accumulator[Any]], op_name: str, samples: list[SampleRecord]
) -> list[Issue]:
    """Pushed samples must all appear in (push_many output ∪ flush output)."""
    try:
        acc = factory()
    except Exception as exc:
        return [
            Issue(
                severity="error",
                code="ACC_FACTORY_RAISED",
                op_name=op_name,
                message=(
                    f"accumulator factory raised {type(exc).__name__}: {exc}. "
                    f"The factory must succeed under deterministic=True with "
                    f"an empty ctx dict."
                ),
            )
        ]

    emitted: list[Any] = []
    try:
        emitted.extend(_drain_batches(acc.push_many(copy.deepcopy(samples))))
        emitted.extend(_drain_batches(acc.flush()))
    except Exception as exc:
        return [
            Issue(
                severity="error",
                code="ACC_PUSH_OR_FLUSH_RAISED",
                op_name=op_name,
                message=(
                    f"push_many/flush raised {type(exc).__name__} on "
                    f"synthetic input: {exc}."
                ),
            )
        ]

    expected_ids = sorted(_ids(samples))
    got_ids = sorted(_ids(emitted))
    if expected_ids != got_ids:
        lost = set(expected_ids) - set(got_ids)
        gained = set(got_ids) - set(expected_ids)
        return [
            Issue(
                severity="error",
                code="ACC_FLUSH_LOSES_SAMPLES",
                op_name=op_name,
                message=(
                    f"Sample conservation failed. push_many + flush emitted "
                    f"{len(got_ids)} samples; expected {len(expected_ids)}. "
                    f"Lost: {sorted(lost)[:5]}; gained: {sorted(gained)[:5]}."
                ),
            )
        ]
    return []


def _check_acc_reset_yields_fresh(
    factory: Callable[[], Accumulator[Any]], op_name: str, samples: list[SampleRecord]
) -> list[Issue]:
    """After ``flush(reset=True)``, the accumulator must behave like a fresh one.

    Concretely: feed round 1, ``flush(reset=True)``, feed round 2 — the output
    on round 2 must equal what a fresh accumulator would have produced when
    fed only round 2.
    """
    round_1 = copy.deepcopy(samples)
    round_2 = copy.deepcopy(samples)
    # Disambiguate so reset failures show as ID divergence rather than "same".
    # Bump the leading (DatasetId) component so the id stays a valid SampleId
    # triple; ``*sid[1:]`` keeps this robust to records of any tuple arity.
    for r in round_2:
        sid = r.meta.sample_id
        r.meta = SampleMeta(
            sample_id=(sid[0] + 1, *sid[1:]),
            lane_id=r.meta.lane_id,
            chunk_id=r.meta.chunk_id,
            chunk_offset=r.meta.chunk_offset,
        )

    try:
        used = factory()
        used.push_many(copy.deepcopy(round_1))
        used.flush(reset=True)
        out_used = _drain_batches(used.push_many(copy.deepcopy(round_2)))
        out_used.extend(_drain_batches(used.flush()))

        fresh = factory()
        out_fresh = _drain_batches(fresh.push_many(copy.deepcopy(round_2)))
        out_fresh.extend(_drain_batches(fresh.flush()))
    except Exception as exc:
        return [
            Issue(
                severity="error",
                code="ACC_RESET_RAISED",
                op_name=op_name,
                message=(
                    f"Reset-then-reuse cycle raised {type(exc).__name__}: {exc}. "
                    f"flush(reset=True) must leave the accumulator usable."
                ),
            )
        ]

    if not _outputs_equivalent(out_used, out_fresh):
        return [
            Issue(
                severity="error",
                code="ACC_RESET_NOT_FRESH",
                op_name=op_name,
                message=(
                    "After flush(reset=True), the accumulator produced "
                    "different output than a freshly constructed one fed the "
                    "same data (sample IDs or payloads diverged). Reset must "
                    "restore the accumulator to a state indistinguishable "
                    "from __init__."
                ),
            )
        ]
    return []


def _check_acc_has_pending_data(
    factory: Callable[[], Accumulator[Any]], op_name: str, samples: list[SampleRecord]
) -> list[Issue]:
    """``has_pending_data()`` must agree with whether ``flush()`` will yield.

    Tests three corners:
    - Fresh accumulator: must be False.
    - After a push whose output was buffered (no batches emitted): must be True.
    - After a push whose output was emitted immediately (Passthrough-shape):
      must be False.
    - After ``flush()`` drained the buffer: must be False.

    The middle two cases let us distinguish Passthrough-shape accumulators
    from buffering-shape accumulators with the same probe.
    """
    try:
        acc = factory()
    except Exception:
        return []  # surfaced by ACC_FACTORY_RAISED in the conservation check

    issues: list[Issue] = []

    if acc.has_pending_data():
        issues.append(
            Issue(
                severity="error",
                code="ACC_HAS_PENDING_LIES",
                op_name=op_name,
                message=(
                    "Fresh accumulator reports has_pending_data()=True before "
                    "any push. has_pending_data() must mirror the actual "
                    "buffer state."
                ),
            )
        )

    # The remaining corners push and flush synthetic records. A
    # payload-reading accumulator (``Accumulator.reads_payload``) can raise
    # on the synthetic record — the conservation check already reports that
    # root cause as ACC_PUSH_OR_FLUSH_RAISED, so we bail quietly here like the
    # sibling checks rather than letting the raw exception escape validate().
    try:
        # Push a single sample and ask whether the buffer state matches what
        # push_many returned. If the sample was buffered (no batch emitted),
        # has_pending_data must be True; if it was passed through immediately,
        # has_pending_data must be False.
        one = copy.deepcopy(samples[:1])
        emitted = sum(len(b) for b, _ in acc.push_many(copy.deepcopy(one)))
        pending = acc.has_pending_data()
        if emitted < len(one) and not pending:
            issues.append(
                Issue(
                    severity="error",
                    code="ACC_HAS_PENDING_LIES",
                    op_name=op_name,
                    message=(
                        f"Pushed {len(one)} sample(s); accumulator buffered "
                        f"{len(one) - emitted} (no batch emitted) but "
                        f"has_pending_data() returned False."
                    ),
                )
            )
        elif emitted == len(one) and pending:
            issues.append(
                Issue(
                    severity="error",
                    code="ACC_HAS_PENDING_LIES",
                    op_name=op_name,
                    message=(
                        "Accumulator emitted every pushed sample immediately "
                        "but still reports has_pending_data()=True."
                    ),
                )
            )

        # After a flush, the buffer must be empty.
        acc.push_many(copy.deepcopy(samples))
        acc.flush()
        if acc.has_pending_data():
            issues.append(
                Issue(
                    severity="error",
                    code="ACC_HAS_PENDING_LIES",
                    op_name=op_name,
                    message=(
                        "Accumulator reports has_pending_data()=True after "
                        "flush() drained the buffer."
                    ),
                )
            )
    except Exception:
        return issues  # push/flush failure surfaced as ACC_PUSH_OR_FLUSH_RAISED
    return issues


def _check_acc_preserves_lane_id(
    factory: Callable[[], Accumulator[Any]], op_name: str, samples: list[SampleRecord]
) -> list[Issue]:
    """Each emitted sample must keep the ``lane_id`` it had on push.

    Accumulators do not own lane metadata — they are routing/buffering
    machinery.  Rewriting ``lane_id`` (whether deliberately or via a
    confused dataclass replace) corrupts the engine's per-lane progress
    bookkeeping and breaks deterministic replay across rank-count
    changes.
    """
    input_lane_by_id = {r.meta.sample_id: r.meta.lane_id for r in samples}

    try:
        acc = factory()
        emitted: list[Any] = []
        emitted.extend(_drain_batches(acc.push_many(copy.deepcopy(samples))))
        emitted.extend(_drain_batches(acc.flush()))
    except Exception:
        return []  # surfaced by other checks

    for record in emitted:
        sid = record.meta.sample_id
        expected = input_lane_by_id.get(sid)
        if expected is None:
            continue  # ID mismatch is the conservation check's domain
        actual = record.meta.lane_id
        if actual != expected:
            return [
                Issue(
                    severity="error",
                    code="ACC_MUTATES_LANE_ID",
                    op_name=op_name,
                    message=(
                        f"Sample {sid!r} entered the accumulator on lane "
                        f"{expected} but was emitted with lane_id={actual}. "
                        f"Accumulators must not mutate sample metadata — lane "
                        f"assignment is owned by the engine."
                    ),
                )
            ]
    return []


def _check_acc_lane_pure_batches(
    factory: Callable[[], Accumulator[Any]], op_name: str, samples: list[SampleRecord]
) -> list[Issue]:
    """Bucketing accumulators must emit batches containing only one lane each.

    Pushes all resolved records (user-supplied when available, else synthetic)
    in a single call so any per-lane bucketing is exercised in one shot.  The
    records are round-robined across lanes first, so a single push interleaves
    lanes regardless of the input ordering — this forces per-call bucketing to
    route across lanes and catches accumulators that batch by contiguous run
    rather than by lane.  Skips Passthrough-shape accumulators — those return
    all input as one batch per push and aren't doing lane grouping; lane purity
    is upstream's concern for them.  Detection: a single ready batch containing
    all pushed samples, with nothing flushed.
    """
    interleaved = _interleave_by_lane(copy.deepcopy(samples))
    try:
        acc = factory()
        ready = list(acc.push_many(interleaved))
        flushed = list(acc.flush())
    except Exception:
        return []  # surfaced by other checks

    # Passthrough-shape: one batch out for one batch in, nothing buffered.
    if len(ready) == 1 and not flushed and len(ready[0][0]) == len(interleaved):
        return []

    for batch, _ in ready + flushed:
        lanes = {r.meta.lane_id for r in batch}
        if len(lanes) > 1:
            return [
                Issue(
                    severity="error",
                    code="ACC_BATCH_NOT_LANE_PURE",
                    op_name=op_name,
                    message=(
                        f"Accumulator emitted a batch spanning lanes "
                        f"{sorted(lanes)}. Bucketing accumulators must "
                        f"produce lane-pure batches so the engine can track "
                        f"per-lane progress correctly."
                    ),
                )
            ]
    return []


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _is_user_op(op: Any) -> bool:
    """Identify ops that should be validated.

    Two shapes count as user code: ``_FunctionalOp`` instances (kwargs
    form of ``Pipeline.add_op`` wraps user callables in one) and any op
    whose class lives outside the ``zephon.*`` package (user `BaseOp`
    subclasses attached via the instance form).  Built-in framework ops
    live in ``zephon.ops`` / ``zephon.core`` and are skipped — they
    have their own dedicated test coverage.
    """
    from zephon.api.pipeline import _FunctionalOp

    if isinstance(op, _FunctionalOp):
        return True
    return not type(op).__module__.startswith("zephon.")


def _guard(op_name: str, label: str, fn: Callable[[], list[Issue]]) -> list[Issue]:
    """Run one validator check, converting an unexpected crash into an Issue.

    :func:`validate_pipeline` runs inside :meth:`Pipeline.__iter__` under the
    ``auto_validation`` setting.  An unhandled exception in any individual
    check would escape ``iter()`` as a raw traceback — defeating even
    ``auto_validation="warn"``, the documented false-positive escape hatch,
    where only an aggregated :class:`ValidationReport` (never an exception)
    should reach the user.  Any check that raises is reported as
    ``OP_VALIDATOR_CRASHED`` so the harness only ever surfaces a report.
    """
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 — last-resort safety net
        return [
            Issue(
                severity="error",
                code="OP_VALIDATOR_CRASHED",
                op_name=op_name,
                message=(
                    f"The {label} check crashed with "
                    f"{type(exc).__name__}: {exc}. This is unexpected — the op "
                    f"could not be fully validated. Please report this; in the "
                    f"meantime set auto_validation='off' to bypass the harness."
                ),
            )
        ]


def validate_pipeline(pipeline: "Pipeline") -> ValidationReport:
    """Run the validation harness against every user op on the pipeline graph.

    Two op shapes are probed: ``_FunctionalOp`` instances (built by the
    kwargs form of :meth:`Pipeline.add_op`) and user `BaseOp` subclasses
    attached via the instance form.  Built-in framework ops are skipped
    via :func:`_is_user_op` since they have their own tests.

    For user `BaseOp` subclasses (instance form), runtime probes only
    run when the user opts in by overriding
    :meth:`zephon.core.op_base.BaseOp.validation_samples` with a
    non-empty batch.  The validator never invokes ``setup``, so calling
    ``process_many`` against a `BaseOp` whose body depends on
    ``setup``-built resources (the canonical pattern — tokenizers,
    model handles, etc.) would raise on any non-trivial body.  A
    consistent raise degrades to the non-blocking
    ``OP_REJECTS_GENERIC_PAYLOAD`` warning regardless of exception type,
    so it would not block iteration — but it *would* emit a noisy
    NOT-VALIDATED warning for every setup-dependent op.  Gating on
    ``validation_samples()`` keeps the signal clean: users opt instance-form
    ops into runtime validation by providing records that work
    pre-setup, and a clearer skip-explanation warning
    (``OP_INSTANCE_RUNTIME_CHECKS_SKIPPED``) is surfaced otherwise.

    For user `BaseOp` subclasses, the accumulator-side checks are also
    skipped — their ``accumulator()`` may legitimately depend on
    ``setup``-built state that the synthetic probe doesn't establish.
    Static AST checks (``_check_op_writes_to_self`` /
    ``_check_op_uses_nondeterministic_stdlib``) run unconditionally for
    every user op.
    """
    # Local import to avoid the pipeline → validate → pipeline circular at module load.
    from zephon.api.pipeline import _FunctionalOp

    report = ValidationReport()
    for node in pipeline._graph.nodes:
        if not _is_user_op(node.op):
            continue
        op = node.op
        name = node.name

        # Static-analysis heuristics run first — they don't depend on the
        # synthetic-payload probe succeeding, so they still report when an
        # op rejects the validator's generic records.  Every check is routed
        # through ``_guard`` so an unexpected crash becomes an Issue rather
        # than escaping ``iter()`` (see ``_guard`` for why this matters).
        report.issues.extend(
            _guard(name, "writes-to-self", lambda: _check_op_writes_to_self(op, name))
        )
        report.issues.extend(
            _guard(
                name,
                "nondeterministic-stdlib",
                lambda: _check_op_uses_nondeterministic_stdlib(op, name),
            )
        )

        # Resolve records once per op — either user-supplied via
        # ``validation_samples()`` or the synthetic fallback.  A buggy
        # factory surfaces ``OP_VALIDATION_SAMPLES_FACTORY_FAILED`` and
        # we fall back transparently.
        samples, user_provided_samples, sample_issues = _resolve_validation_records(
            op, name
        )
        report.issues.extend(sample_issues)

        is_functional = isinstance(op, _FunctionalOp)
        if is_functional or user_provided_samples:
            # Output-type check first: the structural checks below reach into
            # ``record.meta`` and would crash on non-StreamItem output, so when
            # the op emits the wrong shape we report it cleanly here and skip
            # them (they would only add confusing OP_VALIDATOR_CRASHED noise).
            output_type_issues = _guard(
                name,
                "output-types",
                lambda: _check_op_output_types(op, name, samples),
            )
            report.issues.extend(output_type_issues)
            if not output_type_issues:
                report.issues.extend(
                    _guard(
                        name,
                        "determinism",
                        lambda: _check_op_determinism(op, name, samples),
                    )
                )
                report.issues.extend(
                    _guard(
                        name,
                        "cross-call-state",
                        lambda: _check_op_no_cross_call_state(op, name, samples),
                    )
                )
                report.issues.extend(
                    _guard(
                        name,
                        "self-mutation",
                        lambda: _check_op_self_mutation(op, name, samples),
                    )
                )
                report.issues.extend(
                    _guard(
                        name,
                        "state-diff",
                        lambda: _check_op_state_diff(op, name, samples),
                    )
                )
                report.issues.extend(
                    _guard(
                        name,
                        "sample-identity",
                        lambda: _check_op_sample_identity(op, name, samples),
                    )
                )
        else:
            report.issues.append(
                Issue(
                    severity="warning",
                    code="OP_INSTANCE_RUNTIME_CHECKS_SKIPPED",
                    op_name=name,
                    message=(
                        "!! THIS OP WAS NOT VALIDATED !! Runtime checks "
                        "(determinism, cross-call state, self-mutation, "
                        "state-diff, sample identity) were skipped for this "
                        "`BaseOp` instance-form op. The validator does not "
                        "invoke `setup()`, so calling `process_many` against "
                        "an op whose body needs setup-built resources would "
                        "raise spuriously. Override `validation_samples()` on "
                        "your subclass to return a small batch of "
                        "`SampleRecord` instances whose payload your "
                        "`process_many` can consume *without* setup, and the "
                        "full runtime check suite will run."
                    ),
                )
            )

        # Accumulator checks construct the accumulator via ``op.accumulator(...)``
        # with a probe ctx.  For user `BaseOp` subclasses, that call may
        # depend on ``setup``-built state we haven't established, so skip
        # the accumulator probes there to avoid false-positive
        # ``ACC_FACTORY_RAISED`` errors.
        if is_functional:
            factory = _make_accumulator_factory(op)
            report.issues.extend(
                _guard(
                    name,
                    "acc-flush-conservation",
                    lambda: _check_acc_flush_conservation(factory, name, samples),
                )
            )
            report.issues.extend(
                _guard(
                    name,
                    "acc-reset-yields-fresh",
                    lambda: _check_acc_reset_yields_fresh(factory, name, samples),
                )
            )
            report.issues.extend(
                _guard(
                    name,
                    "acc-has-pending-data",
                    lambda: _check_acc_has_pending_data(factory, name, samples),
                )
            )
            report.issues.extend(
                _guard(
                    name,
                    "acc-preserves-lane-id",
                    lambda: _check_acc_preserves_lane_id(factory, name, samples),
                )
            )
            report.issues.extend(
                _guard(
                    name,
                    "acc-lane-pure-batches",
                    lambda: _check_acc_lane_pure_batches(factory, name, samples),
                )
            )
    return report
