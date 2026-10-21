# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Check chat templates for message-based loss selection.

``after_last_user`` needs to know which input assistant owns each generated
range. ``OwnedTracker`` and ``instrument`` attach that ownership during Jinja
rendering; ``OwnedTemplate`` rejects constructs that make ownership ambiguous.
``TemplateVerifier`` inserts markers into sample messages to check that selected
content receives loss, user/tool content does not, and rendering and the complete
assistant mask agree with Hugging Face. These checks cannot prove every branch
of arbitrary content-dependent Jinja code.

Setup checks simple conversations. New message/tool structures are checked on
first use and successful results are cached within each worker process. Every
actual render still checks its owners and character ranges.
Successful setup checks can travel with an operator to another process; its
compiled Jinja template is rebuilt locally before the result's identity is checked.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, TypedDict, TypeGuard, cast, get_args

import numpy as np
from jinja2 import TemplateError, nodes, pass_context
from jinja2.ext import Extension
from jinja2.parser import Parser
from jinja2.runtime import Context

from zephon._internal.utils.tokenizer import (
    generation_mask,
    selected_assistant_indices,
    thread_salted_template,
)
from zephon.ops.config import LossScope
from zephon.validation import Issue, ValidationError, ValidationReport

STATE = "__zephon_owned_generation_state"
_TEXT_FIELDS = ("content", "reasoning_content", "reasoning", "thinking")


def _content_controls(ast: nodes.Node) -> tuple[str, ...]:
    """Find literal text controls that marker probes must retain.

    Exact text and nonempty length comparisons cannot survive replacement with
    ownership markers. Type/emptiness checks and literal substring controls can.
    This checks direct fields and assignment aliases, not arbitrary Jinja dataflow.
    """
    aliases: set[str] = set(_TEXT_FIELDS)

    def reads_text(node: nodes.Node) -> bool:
        if isinstance(node, nodes.Name) and node.name in aliases:
            return True
        if isinstance(node, nodes.Getattr) and node.attr in _TEXT_FIELDS:
            return True
        if (
            isinstance(node, nodes.Getitem)
            and isinstance(node.arg, nodes.Const)
            and node.arg.value in _TEXT_FIELDS
        ):
            return True
        return any(reads_text(child) for child in node.iter_child_nodes())

    assignments = list(ast.find_all(nodes.Assign))
    while True:
        before = len(aliases)
        for assignment in assignments:
            if isinstance(assignment.target, nodes.Name) and reads_text(
                assignment.node
            ):
                aliases.add(assignment.target.name)
        if len(aliases) == before:
            break

    controls = {"<think>", "</think>"}
    for node in ast.find_all((nodes.Compare, nodes.Call)):
        if isinstance(node, nodes.Compare) and reads_text(node):
            left = node.expr
            for operand in node.ops:
                right = operand.expr
                if reads_text(left) or reads_text(right):
                    if (
                        operand.op in {"in", "notin"}
                        and isinstance(left, nodes.Const)
                        and isinstance(left.value, str)
                        and left.value
                    ):
                        controls.add(left.value)
                    elif (
                        isinstance(left, nodes.Filter)
                        and left.name in {"length", "count"}
                        and isinstance(right, nodes.Const)
                        and right.value == 0
                        and operand.op in {"eq", "ne", "gt", "lteq"}
                    ) or (
                        isinstance(right, nodes.Const)
                        and right.value in (None, "")
                        and operand.op in {"eq", "ne"}
                    ):
                        pass
                    else:
                        raise ValueError(
                            f"template line {node.lineno}: content-dependent comparison "
                            "cannot be checked with ownership markers"
                        )
                left = right
        elif (
            isinstance(node, nodes.Call)
            and isinstance(node.node, nodes.Getattr)
            and node.node.attr in {"startswith", "endswith"}
            and reads_text(node.node.node)
        ):
            if (
                len(node.args) != 1
                or not isinstance(node.args[0], nodes.Const)
                or not isinstance(node.args[0].value, str)
                or not node.args[0].value
                or node.kwargs
                or node.dyn_args is not None
                or node.dyn_kwargs is not None
            ):
                raise ValueError(
                    f"template line {node.lineno}: content prefix/suffix checks "
                    "require one nonempty string literal"
                )
            controls.add(node.args[0].value)
    return tuple(sorted(controls, key=lambda text: (-len(text), text)))


@dataclass
class RenderState:
    """Per-render message identities, output position, and generated spans."""

    owners: dict[int, int]
    messages: list[Mapping[str, Any]]
    cursor: int = 0
    spans: list[tuple[int, int, int, int]] = field(default_factory=list)


class OwnedTracker(Extension):
    """Associate generation blocks with their enclosing input message."""

    tags = {"generation"}
    priority = 101

    def parse(self, parser: Parser) -> nodes.CallBlock:
        line = next(parser.stream).lineno
        body = parser.parse_statements(("name:endgeneration",), drop_needle=True)
        block = nodes.CallBlock(
            self.call_method("record", [nodes.Const(line)]), [], [], body
        )
        block.set_lineno(line)
        return block

    @pass_context
    def record(
        self,
        context: Context,
        site: int,
        line: int,
        *candidates: object,
        caller: Callable[[], str],
    ) -> str:
        state = context.resolve(STATE)
        owner = next(
            (
                state.owners[id(value)]
                for value in candidates
                if id(value) in state.owners
            ),
            None,
        )
        if owner is None:
            raise ValueError(f"generation line {line}: no input-message owner")
        if state.messages[owner].get("role") != "assistant":
            raise ValueError(f"generation line {line}: owner {owner} is not assistant")
        value = caller()
        state.spans.append((owner, state.cursor, state.cursor + len(value), site))
        return value


def is_generation(node: nodes.Node) -> TypeGuard[nodes.CallBlock]:
    """Return whether a node calls the ownership generation extension."""
    return (
        isinstance(node, nodes.CallBlock)
        and isinstance(node.call.node, nodes.ExtensionAttribute)
        and node.call.node.identifier == OwnedTracker.identifier
    )


def instrument(
    node: nodes.Node,
    loops: tuple[str, ...] = (),
    buffered: bool = False,
    in_generation: bool = False,
) -> None:
    """Attach enclosing loop variables to direct generation calls.

    Their names must not be rebound outside generation blocks; each block
    captures its owner before rendering its body.
    """
    if isinstance(node, (nodes.Include, nodes.Import, nodes.FromImport, nodes.Extends)):
        raise ValueError(f"template line {node.lineno}: external template structure")
    if is_generation(node):
        if buffered or in_generation:
            raise ValueError(
                f"generation line {node.lineno}: buffered/nested generation"
            )
        if not loops:
            raise ValueError(f"generation line {node.lineno}: no enclosing loop")
        node.call.args += [nodes.Name(name, "load") for name in reversed(loops)]
        for child in node.body:
            instrument(child, loops, buffered, True)
        return
    if not in_generation:
        targets = []
        if isinstance(node, (nodes.Assign, nodes.AssignBlock, nodes.For)):
            targets = [node.target]
        elif isinstance(node, nodes.With):
            targets = node.targets
        if any(
            isinstance(name, nodes.Name) and name.name in loops
            for target in targets
            for name in (target, *target.find_all(nodes.Name))
        ):
            raise ValueError(
                f"template line {node.lineno}: rebinding an enclosing loop variable"
            )
    buffered = buffered or isinstance(
        node, (nodes.Macro, nodes.CallBlock, nodes.FilterBlock, nodes.AssignBlock)
    )
    if isinstance(node, nodes.For):
        buffered = buffered or node.recursive
        owns_generation = any(
            is_generation(descendant)
            for statement in node.body
            for descendant in (statement, *statement.find_all(nodes.CallBlock))
        )
        for child in node.iter_child_nodes():
            stack = loops
            if (
                owns_generation
                and isinstance(node.target, nodes.Name)
                and any(child is item for item in node.body)
            ):
                stack += (node.target.name,)
            instrument(child, stack, buffered, in_generation)
    else:
        for child in node.iter_child_nodes():
            instrument(child, loops, buffered, in_generation)


class OwnedTemplate:
    """Compiled HF Jinja template with per-render generation ownership."""

    def __init__(self, template: str) -> None:
        if STATE in template:
            raise ValueError("template uses a reserved generation-tracker name")
        try:
            from transformers.utils.chat_template_utils import _compile_jinja_template
        except ImportError as error:
            raise RuntimeError(
                "Chat template verification requires transformers>=4.52; upgrade transformers "
                + "or use loss_scope='all_assistant'/'last_span'."
            ) from error
        environment = _compile_jinja_template("").environment.overlay(
            extensions=[OwnedTracker]
        )
        ast = environment.parse(template)
        self.sites: dict[int, int] = {}
        for node in ast.find_all(nodes.CallBlock):
            if is_generation(node):
                site = len(self.sites)
                self.sites[site] = node.lineno
                node.call.args.insert(0, nodes.Const(site))
        if not self.sites:
            raise ValueError("template has no generation blocks")
        self.content_controls = re.compile(
            "(" + "|".join(re.escape(token) for token in _content_controls(ast)) + ")"
        )
        instrument(ast)
        self.compiled = environment.from_string(ast)
        self.identity = hashlib.sha256(template.encode()).hexdigest()[:12]
        self.source = template

    def render(
        self, messages: list[Mapping[str, Any]], context: Mapping[str, Any]
    ) -> tuple[str, list[tuple[int, int, int, int]]]:
        messages = [dict(message) for message in messages]
        owners = {id(message): i for i, message in enumerate(messages)}
        if STATE in context or "messages" in context:
            raise ValueError("reserved generation-tracker context name")
        tools = context.get("tools")
        if tools is not None and (
            not isinstance(tools, (list, tuple))
            or any(not isinstance(tool, dict) for tool in tools)
        ):
            raise ValueError(
                "chat template verification requires tools as JSON-schema dictionaries"
            )
        documents = context.get("documents")
        if documents is not None and (
            not isinstance(documents, (list, tuple))
            or any(not isinstance(doc, dict) for doc in documents)
        ):
            raise ValueError("documents must be dictionaries")
        state = RenderState(owners, messages)
        kwargs = {
            "tools": None,
            "documents": None,
            **context,
            "messages": messages,
            "add_generation_prompt": False,
            STATE: state,
        }
        chunks: list[str] = []
        for chunk in self.compiled.generate(**kwargs):
            chunks.append(chunk)
            state.cursor += len(chunk)
        text = "".join(chunks)
        previous_end = 0
        for _, start, end, site in state.spans:
            if not 0 <= previous_end <= start <= end <= len(text):
                raise ValueError(
                    f"generation line {self.sites[site]}: invalid or overlapping output range"
                )
            previous_end = end
        return text, state.spans


def _issue(code: str, message: str, *, warning: bool = False) -> Issue:
    if not warning and code != "CHAT_INPUT_CONTEXT":
        message += (
            " Check chat_template: generation blocks must be inside the input "
            "assistant's message loop and cover its reasoning, tool calls and "
            "answer. Keep user and tool-result content outside those blocks."
        )
    return Issue(
        severity="warning" if warning else "error",
        code=code,
        op_name="tokenize_chat",
        message=message,
        doc_link="",
    )


def _call_arguments(value: Any) -> Any:
    """Decode JSON arguments for probes; retain non-JSON strings as text."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    if value is not None and not isinstance(value, (dict, list, int, float)):
        raise ValueError("tool-call arguments must be JSON values or text")
    return value


def _call_function(value: Any) -> dict[str, Any]:
    """Return a wrapped or flat tool-call function without changing its format."""
    if not isinstance(value, dict):
        raise ValueError("tool calls must be dictionaries")
    function = value.get("function", value)
    if not isinstance(function, dict) or (
        "function" not in value and "name" not in value
    ):
        raise ValueError("tool calls must contain a function dictionary or a name")
    return function


def _tool_structure(value: Any, *, call: bool) -> tuple[Any, ...]:
    """Describe tool formats without names, argument contents or parameter schemas."""
    if not isinstance(value, dict):
        raise ValueError("tools and tool calls must be dictionaries")
    function = _call_function(value) if call else value.get("function", value)
    if not isinstance(function, dict):
        raise ValueError("tools must contain a function dictionary")
    common = (
        "wrapped" if "function" in value else "flat",
        tuple(sorted(value)),
        tuple(sorted(function)),
        _structure(value.get("type"), "type"),
        _structure(function.get("name")),
    )
    if call:
        arguments = function.get("arguments", {})
        decoded = _call_arguments(arguments)
        return (
            *common,
            type(arguments).__name__,
            ("object", bool(decoded))
            if isinstance(decoded, dict)
            else ("literal", repr(arguments), repr(function.get("name"))),
        )
    parameters = function.get("parameters", function.get("arguments"))
    return (
        *common,
        _structure(function.get("description")),
        type(parameters).__name__,
        bool(parameters),
    )


def _tool_list_structure(value: Any, *, call: bool) -> tuple[Any, ...]:
    """Keep 0/1/2/3+ counts, first/last formats and distinct middle formats."""
    if value is None:
        return ("none",)
    if not isinstance(value, (list, tuple)):
        raise ValueError("tools and tool_calls must be lists or None")
    # Inspect every element before the cache lookup, including repeated formats.
    items = tuple(_tool_structure(item, call=call) for item in value)
    return (
        "list",
        min(len(items), 3),
        items[:1],
        items[-1:],
        tuple(sorted(set(items[1:-1]), key=repr)),
    )


def _structure(value: Any, key: str = "") -> Any:
    """Describe message formats and validate fields on every cache lookup."""
    if key in {"tools", "tool_calls"}:
        return _tool_list_structure(value, call=key == "tool_calls")
    if key in _TEXT_FIELDS:
        if key == "content" and isinstance(value, list):
            if any(
                not isinstance(part, dict)
                or part.get("type") != "text"
                or not isinstance(part.get("text", ""), (str, type(None)))
                for part in value
            ):
                raise ValueError(
                    "chat template verification supports text-only content parts"
                )
            return tuple(_structure(part) for part in value)
        elif value is not None and not isinstance(value, str):
            raise ValueError(f"message {key} must be text or None")
    if isinstance(value, Mapping):
        return tuple((k, _structure(v, k)) for k, v in sorted(value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_structure(item, key) for item in value)
    if (
        key in {"role", "type", "enable_thinking"}
        or value is None
        or isinstance(value, bool)
    ):
        return value
    if isinstance(value, str):
        return (
            "str",
            bool(value),
            "<think>" in value,
            "</think>" in value,
            value.startswith("<tool_response>") and value.endswith("</tool_response>"),
        )
    return type(value).__name__


def _marked_probe(
    messages: Sequence[Mapping[str, Any]],
    scope: LossScope,
    content_controls: re.Pattern[str] | None = None,
) -> tuple[list[Mapping[str, Any]], dict[str, int], set[str]]:
    """Replace text fields with source-owned markers, retaining message structure."""
    marked = [copy.deepcopy(dict(message)) for message in messages]
    markers: dict[str, int] = {}
    required: set[str] = set()
    selected = selected_assistant_indices(messages, scope)

    def marker(owner: int, *, required_output: bool = False) -> str:
        text = f"zephon_probe_{owner}_{len(markers)}_text"
        markers[text] = owner
        if required_output:
            required.add(text)
        return text

    def mark_text(value: str, owner: int) -> str:
        parts = (
            content_controls.split(value) if content_controls is not None else [value]
        )
        for index in range(0, len(parts), 2):
            part = parts[index]
            if part.strip():
                left = part[: len(part) - len(part.lstrip())]
                right = part[len(part.rstrip()) :]
                parts[index] = (
                    left + marker(owner, required_output=owner in selected) + right
                )
        return "".join(parts)

    for i, message in enumerate(marked):
        content = message.get("content")
        if isinstance(content, str) and content:
            message["content"] = mark_text(content, i)
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict) or part.get("type") != "text":
                    raise ValueError(
                        "chat template verification supports text-only content parts"
                    )
                if part.get("text"):
                    part["text"] = mark_text(part["text"], i)
        elif content is not None and not isinstance(content, str):
            raise ValueError("message content must be text, text parts or None")
        for field_name in _TEXT_FIELDS[1:]:
            if message.get(field_name):
                message[field_name] = mark_text(message[field_name], i)
        for call in message.get("tool_calls") or []:
            function = _call_function(call)
            args = function.get("arguments", {})
            is_json = isinstance(args, str)
            args = _call_arguments(args)
            if not isinstance(args, dict):
                continue
            # A string argument detects templates that omit the whole call.
            args = {**args, "zephon_probe": marker(i, required_output=i in selected)}
            function["arguments"] = json.dumps(args) if is_json else args
    return cast(list[Mapping[str, Any]], marked), markers, required


def _check_markers(
    text: str,
    spans: Sequence[tuple[int, int, int, int]],
    messages: Sequence[Mapping[str, Any]],
    markers: Mapping[str, int],
    required: set[str],
) -> list[Issue]:
    issues: list[Issue] = []
    found: set[str] = set()
    pattern = "|".join(re.escape(marker) for marker in markers)
    for match in re.finditer(pattern, text) if pattern else ():
        marker = match.group()
        found.add(marker)
        expected_owner = markers[marker]
        start, end = match.span()
        overlaps = [
            (owner, max(start, a), min(end, b))
            for owner, a, b, _ in spans
            if a < end and b > start
        ]
        assistant = messages[expected_owner].get("role") == "assistant"
        covered = sum(b - a for owner, a, b in overlaps if owner == expected_owner)
        if (not assistant and overlaps) or any(
            owner != expected_owner for owner, _, _ in overlaps
        ):
            issues.append(
                _issue(
                    "CHAT_CONTENT_OWNERSHIP",
                    f"Generated content has the wrong owner for message {expected_owner}.",
                )
            )
        elif marker in required and covered != end - start:
            issues.append(
                _issue(
                    "CHAT_REQUIRED_SUPERVISION",
                    f"Required content for message {expected_owner} is outside its generation ranges.",
                )
            )
    for marker in required - found:
        issues.append(
            _issue(
                "CHAT_REQUIRED_CONTENT",
                f"Required content for message {markers[marker]} was omitted.",
            )
        )
    return issues


# Process-local LRU: successful reports and executed generation sites, keyed by
# tokenizer configuration, template, scope, kwargs and message/tool structure.
# Threads share this lock-protected cache; separate processes have independent
# copies. The lock never covers rendering, so concurrent misses may repeat work.
_VERIFIED_STRUCTURES: OrderedDict[Any, tuple[ValidationReport, frozenset[int]]] = (
    OrderedDict()
)
_VERIFICATION_LOCK = threading.Lock()
_VERIFICATION_CACHE_LIMIT = 1024


@lru_cache(maxsize=32)
def _owned_template(template: str) -> OwnedTemplate:
    """Cache up to 32 compiled templates; each render has its own tracking state."""
    return OwnedTemplate(template)


@dataclass(frozen=True)
class TemplateSetupResult:
    """Portable successful setup checks for one verifier configuration."""

    identity: bytes
    issues: tuple[Issue, ...]
    executed_sites: frozenset[int]


def _changed_range(before: str, after: str) -> tuple[int, int]:
    """Locate the original text between the common prefix and suffix."""
    start = 0
    while start < min(len(before), len(after)) and before[start] == after[start]:
        start += 1
    end, other_end = len(before), len(after)
    while end > start and other_end > start and before[end - 1] == after[other_end - 1]:
        end -= 1
        other_end -= 1
    return start, end


class TemplateVerifier:
    """Verify a compiled template and gate new input structures before rendering."""

    def __init__(
        self,
        tokenizer: Any,
        template: str,
        kwargs: Mapping[str, Any],
        scope: LossScope = "after_last_user",
    ) -> None:
        self.tokenizer = tokenizer
        self.template = _owned_template(template)
        self.kwargs = dict(kwargs)
        self.scope: LossScope = scope
        backend = getattr(tokenizer, "backend_tokenizer", None)
        if not getattr(tokenizer, "is_fast", False) or backend is None:
            raise ValueError(
                "Chat template verification requires a fast Hugging Face tokenizer"
            )
        # Hash the backend directly; the small outer key never copies its JSON.
        # Recompute at construction so tokenizer mutations invalidate old probes.
        identity = hashlib.sha256(backend.to_str().encode()).digest()
        self._identity = hashlib.sha256(
            repr(
                (
                    type(tokenizer),
                    identity,
                    tokenizer.special_tokens_map,
                    self.kwargs,
                    getattr(tokenizer, "truncation_side", None),
                    template,
                    scope,
                )
            ).encode()
        ).digest()
        self.executed_sites: set[int] = set()
        self.setup_result: TemplateSetupResult | None = None

    def _verify_literal_arguments(
        self,
        messages: list[Mapping[str, Any]],
        context: Mapping[str, Any],
        text: str,
        spans: Sequence[tuple[int, int, int, int]],
    ) -> list[Issue]:
        """Check original non-object output against a counterfactual render.

        Empty strings have no argument text; substituting the name checks the call.
        Branches that cannot render the probe fail verification.
        """
        issues: list[Issue] = []
        selected = selected_assistant_indices(messages, self.scope)
        for owner, message in enumerate(messages):
            for index, call in enumerate(message.get("tool_calls") or []):
                args = _call_function(call).get("arguments", {})
                if isinstance(_call_arguments(args), dict):
                    continue
                probe = copy.deepcopy(messages)
                calls = probe[owner]["tool_calls"]
                marker = f"zephon_argument_{owner}_{index}_text"
                if args == "":
                    _call_function(calls[index])["name"] = marker
                else:
                    replacement = {"zephon_probe": marker}
                    _call_function(calls[index])["arguments"] = (
                        json.dumps(replacement)
                        if isinstance(args, str)
                        else replacement
                    )
                changed, _ = self.template.render(probe, context)
                if owner in selected and marker not in changed:
                    issues.append(
                        _issue(
                            "CHAT_TEMPLATE_INVALID", "Argument probe was not rendered."
                        )
                    )
                    continue
                start, end = _changed_range(text, changed)
                if start == end:
                    if owner in selected:
                        issues.append(
                            _issue(
                                "CHAT_REQUIRED_CONTENT",
                                f"Tool-call output for message {owner} was omitted.",
                            )
                        )
                    continue
                fragment = text[start:end]
                issues.extend(
                    _check_markers(
                        fragment,
                        [(i, a - start, b - start, site) for i, a, b, site in spans],
                        messages,
                        {fragment: owner},
                        {fragment} if owner in selected else set(),
                    )
                )
        return issues

    def verify(
        self,
        messages: list[Mapping[str, Any]],
        kwargs: Mapping[str, Any],
        *,
        markers: Mapping[str, int] | None = None,
        required_markers: Sequence[str] = (),
        stock_template: str | None = None,
        check_truncation: bool = True,
        executed_sites: set[int] | None = None,
    ) -> ValidationReport:
        report = ValidationReport()
        try:
            if markers is None:
                probe, markers, required = _marked_probe(
                    messages, self.scope, self.template.content_controls
                )
            else:
                probe, required = copy.deepcopy(messages), set(required_markers)
            context = {**self.tokenizer.special_tokens_map, **kwargs}
            text, spans = self.template.render(probe, context)
            report.issues.extend(
                self._verify_literal_arguments(probe, context, text, spans)
            )
            # HF's cached mask recorder is mutable; use the tokenizer's thread salt.
            hf_template = thread_salted_template(self.template.source)
            hf_text = self.tokenizer.apply_chat_template(
                probe,
                chat_template=hf_template,
                tokenize=False,
                add_generation_prompt=False,
                **kwargs,
            )
            if text != hf_text:
                raise ValueError("owned rendering differs from Hugging Face")
            sites = {site for _, _, _, site in spans}
            with _VERIFICATION_LOCK:
                self.executed_sites.update(sites)
            if executed_sites is not None:
                executed_sites.update(sites)
            report.issues.extend(_check_markers(text, spans, probe, markers, required))
            for limit in (None, 8) if check_truncation else (None,):
                options = (
                    {} if limit is None else {"truncation": True, "max_length": limit}
                )
                hf = self.tokenizer.apply_chat_template(
                    probe,
                    chat_template=hf_template,
                    tokenize=True,
                    return_dict=True,
                    return_assistant_tokens_mask=True,
                    add_generation_prompt=False,
                    **kwargs,
                    **options,
                )
                baseline = generation_mask(
                    hf, spans, selected_assistant_indices(probe, "all_assistant")
                )
                if not np.array_equal(baseline, hf["assistant_masks"]):
                    raise ValueError("generated-token mask differs from Hugging Face")
            if stock_template is not None:
                stock = self.tokenizer.apply_chat_template(
                    probe,
                    chat_template=stock_template,
                    tokenize=False,
                    add_generation_prompt=False,
                    **kwargs,
                )
                if stock != text:
                    report.issues.append(
                        _issue(
                            "CHAT_STOCK_IDENTITY",
                            "Tagged and stock templates render different text.",
                        )
                    )
        except (ValueError, TemplateError) as error:
            report.issues.append(
                _issue(
                    "CHAT_TEMPLATE_INVALID",
                    f"Template {self.template.identity}: {type(error).__name__}: {error}",
                )
            )
        return report

    def setup_report(
        self, *, cached: TemplateSetupResult | None = None
    ) -> ValidationReport:
        """Reuse successful setup checks only for an identical configuration."""
        self.setup_result = None
        if cached is not None and cached.identity == self._identity:
            report = ValidationReport(list(cached.issues))
            if report.ok:
                self.executed_sites.update(cached.executed_sites)
                self.setup_result = cached
                return report
        report = ValidationReport()
        for roles in (
            ("user", "assistant"),
            ("user", "assistant", "user", "assistant"),
        ):
            messages: list[Mapping[str, Any]] = [
                {"role": role, "content": "text"} for role in roles
            ]
            report.issues.extend(
                self._cached_verify(messages, self.kwargs, setup=True).issues
            )
        if report.ok:
            self.setup_result = TemplateSetupResult(
                self._identity, tuple(report.issues), frozenset(self.executed_sites)
            )
        return report

    def _cached_verify(
        self,
        messages: list[Mapping[str, Any]],
        kwargs: Mapping[str, Any],
        *,
        setup: bool = False,
    ) -> ValidationReport:
        """Reuse successful checks across rows, preflight and operator copies.

        Keep message order, field formats, text flags and exact render options.
        Object-argument contents and schemas are omitted. Non-object arguments
        and their tool names remain exact; other names are omitted. Tool/call
        counts use 0/1/2/3+ buckets with first/last and distinct middle formats. These
        probes do not certify branches on omitted values or exact larger counts.
        """
        # Preserve branch controls even when ordinary text shares a cache entry.
        text_controls = []
        for message in messages:
            for field_name in _TEXT_FIELDS:
                value = message.get(field_name)
                values = (
                    [part.get("text") for part in value if isinstance(part, Mapping)]
                    if isinstance(value, list)
                    else [value]
                )
                text_controls.append(
                    tuple(
                        tuple(
                            part if index % 2 else (bool(part), bool(part.strip()))
                            for index, part in enumerate(
                                self.template.content_controls.split(text)
                            )
                        )
                        for text in values
                        if isinstance(text, str)
                    )
                )
        try:
            # Unknown render options remain exact: they may select template modes.
            options = tuple(
                (name, _structure(value, "tools") if name == "tools" else value)
                for name, value in sorted(kwargs.items())
            )
            structure = hashlib.sha256(
                repr((_structure(messages), options, text_controls)).encode()
            ).digest()
        except ValueError as error:
            return ValidationReport(
                [
                    _issue(
                        "CHAT_TEMPLATE_INVALID",
                        f"Template {self.template.identity}: {error}",
                    )
                ]
            )
        key = (self._identity, setup, structure)
        with _VERIFICATION_LOCK:
            cached = _VERIFIED_STRUCTURES.get(key)
            if cached is not None:
                _VERIFIED_STRUCTURES.move_to_end(key)
                self.executed_sites.update(cached[1])
                return cached[0]
        # Rendering and tokenization never hold the cache lock. Concurrent misses
        # may verify twice; failed probes are not cached.
        executed_sites: set[int] = set()
        report = self.verify(
            messages, kwargs, check_truncation=setup, executed_sites=executed_sites
        )
        if report.ok:
            with _VERIFICATION_LOCK:
                _VERIFIED_STRUCTURES[key] = (report, frozenset(executed_sites))
                _VERIFIED_STRUCTURES.move_to_end(key)
                if len(_VERIFIED_STRUCTURES) > _VERIFICATION_CACHE_LIMIT:
                    _VERIFIED_STRUCTURES.popitem(last=False)
        return report

    def verify_input_structure(
        self,
        messages: list[Mapping[str, Any]],
        kwargs: Mapping[str, Any],
        *,
        context: str = "",
    ) -> None:
        """Check an unseen input structure before its training sample is emitted.

        Field validity is checked on every call. Successful cached structures
        skip marker probes; actual rendering still checks owners and ranges.
        """
        report = self._cached_verify(messages, kwargs)
        if not report.ok:
            issues = list(report.issues)
            if context:
                issues.append(_issue("CHAT_INPUT_CONTEXT", context))
            raise ValidationError(ValidationReport(issues))


class _RequiredChatTemplateProbe(TypedDict):
    messages: list[Mapping[str, Any]]


class _ChatTemplateProbe(_RequiredChatTemplateProbe, total=False):
    """Template fixture with optional render context and marker expectations."""

    kwargs: Mapping[str, Any]
    markers: Mapping[str, int]
    required_markers: Sequence[str]


def _verify_chat_template(
    tokenizer: Any,
    chat_template: str | None = None,
    *,
    chat_template_kwargs: Mapping[str, Any] | None = None,
    probes: Sequence[_ChatTemplateProbe] | None = None,
    loss_scope: LossScope = "after_last_user",
    stock_template: str | None = None,
) -> ValidationReport:
    """Check generation ownership, content coverage and HF token-mask parity.

    ``after_last_user`` runs these checks automatically. This entry point is
    useful for template CI, extra fixtures and tagged/stock comparisons.

    Check your template: every selected assistant's reasoning, tool calls,
    answer and any end-of-turn token must appear inside its own ``{% generation %}``
    blocks, within its message loop. Keep user/tool-result text and content read
    from other messages outside those blocks. Inspect rendered text and loss
    masks on representative inputs, because automatic probes can miss
    content-dependent errors.

    Probes retain literal substring, prefix and suffix controls in message text.
    Exact text comparisons and length thresholds beyond emptiness are rejected
    when detected on content fields or their assignment aliases. Coverage checks
    include ``content``, ``reasoning_content``, ``reasoning`` and ``thinking``.

    Args:
        tokenizer: HF-compatible fast tokenizer.
        chat_template: Tagged Jinja source; None uses the tokenizer's template.
        chat_template_kwargs: Static render context.
        loss_scope: Message selection to verify, usually ``after_last_user``.
        probes: Extra fixtures with ``messages`` and optional ``kwargs``.
            By default text is replaced with ownership markers. To supply
            exact fixture text, provide ``markers`` (marker to message index)
            and ``required_markers`` (markers that must appear).
        stock_template: Optional inference template for render-identity checks.

    Returns:
        Findings for mandatory probes and extra fixtures. Unexecuted generation
        sites are warnings; structural, ownership and parity failures are errors.
    """
    try:
        if loss_scope not in get_args(LossScope):
            raise ValueError(f"invalid loss_scope: {loss_scope!r}")
        template = (
            chat_template if chat_template is not None else tokenizer.chat_template
        )
        if not isinstance(template, str) or not template:
            raise ValueError("a generation-tagged chat template is required")
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError("a fast tokenizer is required")
        verifier = TemplateVerifier(
            tokenizer, template, chat_template_kwargs or {}, loss_scope
        )
        report = verifier.setup_report()
        fixtures = list(probes or ())
        if stock_template is not None:
            fixtures.insert(
                0,
                {
                    "messages": [
                        {"role": "user", "content": "hello"},
                        {"role": "assistant", "content": "answer"},
                    ]
                },
            )
        for probe in fixtures:
            kwargs = {**(chat_template_kwargs or {}), **probe.get("kwargs", {})}
            report.issues.extend(
                verifier.verify(
                    probe["messages"],
                    kwargs,
                    markers=probe.get("markers"),
                    required_markers=probe.get("required_markers", ()),
                    stock_template=stock_template,
                ).issues
            )
        uncovered = verifier.template.sites.keys() - verifier.executed_sites
        if uncovered:
            report.issues.append(
                _issue(
                    "CHAT_UNCOVERED_GENERATION",
                    f"Generation sites not exercised (site, line): {[(site, verifier.template.sites[site]) for site in sorted(uncovered)]}",
                    warning=True,
                )
            )
        return report
    except Exception as error:
        return ValidationReport(
            [_issue("CHAT_TEMPLATE_INVALID", f"{type(error).__name__}: {error}")]
        )
