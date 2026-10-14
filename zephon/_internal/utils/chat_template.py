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
from typing import Any, TypeGuard, cast

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
        for child in node.iter_child_nodes():
            stack = loops
            if isinstance(node.target, nodes.Name) and any(
                child is item for item in node.body
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


def _structure(value: Any, key: str = "") -> Any:
    """Describe input structure while excluding conversation text."""
    if isinstance(value, Mapping):
        return tuple((k, _structure(v, k)) for k, v in sorted(value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_structure(item, key) for item in value)
    if (
        key in {"role", "type", "enable_thinking", "name"}
        or value is None
        or isinstance(value, bool)
    ):
        return value
    if isinstance(value, str):
        if key == "arguments":
            try:
                return ("json", _structure(json.loads(value)))
            except ValueError:
                return ("invalid_json",)
        return ("str", bool(value), "<think>" in value, "</think>" in value)
    return type(value).__name__


def _marked_probe(
    messages: Sequence[Mapping[str, Any]],
    scope: LossScope,
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

    for i, message in enumerate(marked):
        content = message.get("content")
        if isinstance(content, str) and content:
            if "</think>" in content:
                thought = marker(i, required_output=i in selected)
                answer = marker(i, required_output=i in selected)
                message["content"] = f"<think>\n{thought}\n</think>\n\n{answer}"
            else:
                message["content"] = marker(i, required_output=i in selected)
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict) or part.get("type") != "text":
                    raise ValueError(
                        "chat template verification supports text-only content parts"
                    )
                if part.get("text"):
                    part["text"] = marker(i, required_output=i in selected)
        elif content is not None and not isinstance(content, str):
            raise ValueError("message content must be text, text parts or None")
        for field_name in ("reasoning_content", "reasoning"):
            if message.get(field_name):
                message[field_name] = marker(i, required_output=i in selected)
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
                raise ValueError("tool calls must contain a function dictionary")
            function = call["function"]
            args = function.get("arguments", {})
            is_json = isinstance(args, str)
            if is_json:
                args = json.loads(args)
            if not isinstance(args, dict):
                raise ValueError(
                    "tool-call arguments must be a dictionary or JSON object"
                )
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
                probe, markers, required = _marked_probe(messages, self.scope)
            else:
                probe, required = copy.deepcopy(messages), set(required_markers)
            context = {**self.tokenizer.special_tokens_map, **kwargs}
            text, spans = self.template.render(probe, context)
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

    def setup_report(self) -> ValidationReport:
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
        return report

    def _cached_verify(
        self,
        messages: list[Mapping[str, Any]],
        kwargs: Mapping[str, Any],
        *,
        setup: bool = False,
    ) -> ValidationReport:
        """Reuse successful checks across rows, preflight and operator copies.

        Structure includes roles, optional fields and tool/schema names, but not
        ordinary text values. This avoids repeating probes for matching inputs;
        it does not certify untested content-dependent template branches.
        """
        structure = hashlib.sha256(
            repr((_structure(messages), _structure(kwargs))).encode()
        ).digest()
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

        Setup cannot anticipate every tool schema or message sequence. Cache hits
        skip probes; diverse inputs may require extra renders for most rows.
        """
        report = self._cached_verify(messages, kwargs)
        if not report.ok:
            issues = list(report.issues)
            if context:
                issues.append(_issue("CHAT_INPUT_CONTEXT", context))
            raise ValidationError(ValidationReport(issues))
