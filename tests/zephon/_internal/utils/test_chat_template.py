# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import pickle
from collections import OrderedDict
from typing import Any

import pytest

from tests.helpers.chat import (
    CONVERSATION,
    CYCLE,
    SPLIT,
    TAGGED_TEMPLATE,
    UNTAGGED_TEMPLATE,
    make_fast_tokenizer,
)
from zephon._internal.utils.chat_template import _verify_chat_template
from zephon.validation import Issue, ValidationError


@pytest.fixture
def fast_tokenizer() -> Any:
    return make_fast_tokenizer()


def test_generation_sites_on_same_line_are_distinct(fast_tokenizer: Any) -> None:
    from zephon._internal.utils.chat_template import OwnedTemplate

    assert len(OwnedTemplate(SPLIT).sites) == 3
    report = _verify_chat_template(fast_tokenizer, SPLIT)
    warnings = [
        issue for issue in report.warnings if issue.code == "CHAT_UNCOVERED_GENERATION"
    ]
    assert len(warnings) == 1
    assert "(0, 1)" in warnings[0].message and "(1, 1)" in warnings[0].message


@pytest.mark.parametrize("error_type", [RuntimeError, TypeError])
def test_verifier_does_not_wrap_programming_errors(
    fast_tokenizer: Any, monkeypatch: pytest.MonkeyPatch, error_type: type[Exception]
) -> None:
    from zephon._internal.utils.chat_template import TemplateVerifier

    verifier = TemplateVerifier(fast_tokenizer, TAGGED_TEMPLATE, {})

    def broken(*args, **kwargs):
        raise error_type("programming error")

    monkeypatch.setattr(verifier.template, "render", broken)
    with pytest.raises(error_type, match="programming error"):
        verifier.verify(CONVERSATION, {})


def test_verification_cache_shared_and_lru(
    fast_tokenizer: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zephon._internal.utils import chat_template as module

    monkeypatch.setattr(module, "_VERIFICATION_CACHE_LIMIT", 2)
    module._VERIFIED_STRUCTURES.clear()
    verifier = module.TemplateVerifier(fast_tokenizer, TAGGED_TEMPLATE, {})
    calls = []
    verify = module.TemplateVerifier.verify

    def counted(self, *args, **kwargs):
        calls.append(1)
        return verify(self, *args, **kwargs)

    monkeypatch.setattr(module.TemplateVerifier, "verify", counted)
    verifier.verify_input_structure(CONVERSATION, {})
    other = module.TemplateVerifier(copy.deepcopy(fast_tokenizer), TAGGED_TEMPLATE, {})
    other.verify_input_structure(CONVERSATION, {})
    assert len(calls) == 1
    verifier.verify_input_structure(CONVERSATION, {"flag": True})
    verifier.verify_input_structure(CONVERSATION, {})  # refresh this entry
    verifier.verify_input_structure(CONVERSATION, {"flag": False})
    verifier.verify_input_structure(CONVERSATION, {})
    assert len(calls) == 3
    verifier.verify_input_structure(CONVERSATION, {"flag": True})
    assert len(calls) == 4
    module._VERIFIED_STRUCTURES.clear()


@pytest.mark.parametrize("target", ["message", "(other, message)"])
def test_with_cannot_rebind_message_owner(target: str) -> None:
    from zephon._internal.utils.chat_template import OwnedTemplate

    template = TAGGED_TEMPLATE.replace(
        "{% generation %}",
        "{% with " + target + " = messages[-1] %}{% generation %}",
    ).replace("{% endgeneration %}", "{% endgeneration %}{% endwith %}")
    with pytest.raises(ValueError, match="rebinding"):
        OwnedTemplate(template)


@pytest.mark.parametrize(
    ("target", "iterable", "index"),
    [
        ("message", "[messages[i - 2] if i >= 2 else message]", "i"),
        ("message, j", "[(messages[i - 2] if i >= 2 else message, i)]", "j"),
    ],
)
def test_nested_for_cannot_rebind_message_owner(
    target: str, iterable: str, index: str
) -> None:
    from zephon._internal.utils.chat_template import OwnedTemplate

    template = (
        "{% for message in messages %}{% if message.role == 'assistant' %}"
        "{% set i = loop.index0 %}"
        "{% for " + target + " in " + iterable + " %}"
        "<a>{% generation %}{{ messages["
        + index
        + "].content }}{% endgeneration %}</a>"
        "{% endfor %}{% else %}<u>{{ message.content }}</u>{% endif %}{% endfor %}"
    )
    with pytest.raises(ValueError, match="rebinding an enclosing loop variable"):
        OwnedTemplate(template)


def test_nested_tool_loop_with_distinct_target_retains_message_owner() -> None:
    from zephon._internal.utils.chat_template import OwnedTemplate

    template = (
        "{% for message in messages %}{% if message.role == 'assistant' %}"
        "{% for tool_call in message.tool_calls %}"
        "{% generation %}{{ tool_call.function.name }}{% endgeneration %}"
        "{% endfor %}{% endif %}{% endfor %}"
    )
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "tool_calls": [{"function": {"name": "lookup"}}]},
    ]
    text, spans = OwnedTemplate(template).render(messages, {})
    assert text == "lookup"
    assert spans == [(1, 0, len(text), 0)]


def test_nested_for_inside_generation_retains_captured_owner() -> None:
    from zephon._internal.utils.chat_template import OwnedTemplate

    template = (
        "{% for message in messages %}{% if message.role == 'assistant' %}"
        "{% generation %}{% for message in [message.content] %}{{ message }}"
        "{% endfor %}{% endgeneration %}{% endif %}{% endfor %}"
    )
    text, spans = OwnedTemplate(template).render(CONVERSATION, {})
    assert text == "good morningthe answer is 4"
    assert spans == [(1, 0, 12, 0), (3, 12, len(text), 0)]


def test_cached_sites_belong_only_to_the_cached_structure(fast_tokenizer: Any) -> None:
    from zephon._internal.utils.chat_template import TemplateVerifier

    template = TAGGED_TEMPLATE.replace(
        "{% generation %}{{ message['content'] }}<|end|>{% endgeneration %}",
        "{% if flag %}{% generation %}{{ message.content }}<|end|>{% endgeneration %}"
        "{% else %}{% generation %}{{ message.content }}<|end|>{% endgeneration %}{% endif %}",
    )
    first = TemplateVerifier(fast_tokenizer, template, {})
    first.verify_input_structure(CONVERSATION, {"flag": False})
    first.verify_input_structure(CONVERSATION, {"flag": True})
    assert first.executed_sites == {0, 1}
    second = TemplateVerifier(fast_tokenizer, template, {})
    second.verify_input_structure(CONVERSATION, {"flag": True})
    assert second.executed_sites == {0}


def test_unknown_tokenizer_cannot_share_verification_results() -> None:
    from zephon._internal.utils.chat_template import TemplateVerifier

    with pytest.raises(ValueError, match="fast Hugging Face tokenizer"):
        TemplateVerifier(object(), TAGGED_TEMPLATE, {})


def test_tokenizer_mutation_changes_verification_identity(fast_tokenizer: Any) -> None:
    from zephon._internal.utils.chat_template import TemplateVerifier

    before = TemplateVerifier(fast_tokenizer, TAGGED_TEMPLATE, {})._identity
    fast_tokenizer.add_tokens(["new-token-for-verifier"])
    after = TemplateVerifier(fast_tokenizer, TAGGED_TEMPLATE, {})._identity
    assert before != after


@pytest.fixture
def counted_verifier(fast_tokenizer: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    from zephon._internal.utils import chat_template as module

    monkeypatch.setattr(module, "_VERIFIED_STRUCTURES", OrderedDict())
    calls = []
    verify = module.TemplateVerifier.verify

    def counted(self, *args, **kwargs):
        calls.append(1)
        return verify(self, *args, **kwargs)

    monkeypatch.setattr(module.TemplateVerifier, "verify", counted)
    return module.TemplateVerifier(fast_tokenizer, SPLIT, {}), calls


def test_cache_reuses_tool_formats_across_payload_changes(
    counted_verifier: Any,
) -> None:
    verifier, calls = counted_verifier
    messages = copy.deepcopy(CYCLE)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Find an item",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            },
        }
    ]
    verifier.verify_input_structure(messages, {"tools": tools})
    messages[3]["tool_calls"][0]["function"] = {
        "name": "weather",
        "arguments": {"locations": ["Berlin", "Paris"], "units": "C"},
    }
    tools[0]["function"].update(
        name="weather",
        description="Forecast temperatures",
        parameters={"type": "object", "properties": {"locations": {"type": "array"}}},
    )
    verifier.verify_input_structure(messages, {"tools": tools})
    assert len(calls) == 1


def test_cache_distinguishes_flat_and_wrapped_calls(counted_verifier: Any) -> None:
    verifier, calls = counted_verifier
    messages = copy.deepcopy(CYCLE)
    verifier.verify_input_structure(messages, {})
    messages[3]["tool_calls"] = [messages[3]["tool_calls"][0]["function"]]
    verifier.verify_input_structure(messages, {})
    verifier.verify_input_structure(CYCLE, {})
    assert len(calls) == 2


@pytest.mark.parametrize("omitted", [False, True])
def test_flat_calls_require_content_and_supervision(
    fast_tokenizer: Any, omitted: bool
) -> None:
    from zephon._internal.utils.chat_template import TemplateVerifier

    messages = copy.deepcopy(CYCLE)
    messages[3]["tool_calls"] = [messages[3]["tool_calls"][0]["function"]]
    template = SPLIT.replace(
        "{% generation %}{{m.tool_calls|tojson}}{% endgeneration %}",
        "" if omitted else "{{m.tool_calls|tojson}}",
    )
    verifier = TemplateVerifier(fast_tokenizer, template, {})
    code = "CHAT_REQUIRED_CONTENT" if omitted else "CHAT_REQUIRED_SUPERVISION"
    with pytest.raises(ValidationError, match=code):
        verifier.verify_input_structure(messages, {})


@pytest.mark.parametrize("field", ["tool_calls", "tools"])
def test_cache_buckets_tool_counts(counted_verifier: Any, field: str) -> None:
    verifier, calls = counted_verifier
    for count in (0, 1, 2, 3, 4, 20):
        messages = copy.deepcopy(CYCLE)
        values = [copy.deepcopy(messages[3]["tool_calls"][0]) for _ in range(count)]
        kwargs = {}
        if field == "tools":
            kwargs["tools"] = values
        else:
            messages[3]["tool_calls"] = values
        verifier.verify_input_structure(messages, kwargs)
    assert len(calls) == 4


def test_cache_distinguishes_first_middle_and_last_call_formats(
    counted_verifier: Any,
) -> None:
    verifier, calls = counted_verifier
    messages = copy.deepcopy(CYCLE)
    call = messages[3]["tool_calls"][0]
    messages[3]["tool_calls"] = [copy.deepcopy(call) for _ in range(3)]
    verifier.verify_input_structure(messages, {})
    for index in range(3):
        changed = copy.deepcopy(messages)
        function = changed[3]["tool_calls"][index]["function"]
        function["arguments"] = json.dumps(function["arguments"])
        verifier.verify_input_structure(changed, {})
    assert len(calls) == 4
    # Repeating a middle format does not require a new probe.
    changed = copy.deepcopy(messages)
    changed[3]["tool_calls"] = [copy.deepcopy(call) for _ in range(5)]
    for index in (1, 2, 3):
        function = changed[3]["tool_calls"][index]["function"]
        function["arguments"] = json.dumps(function["arguments"])
    verifier.verify_input_structure(changed, {})
    assert len(calls) == 4


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("arguments", [object(), {1}, b"text"])
def test_cached_formats_still_validate_every_call(
    counted_verifier: Any, arguments: Any, wrapped: bool
) -> None:
    verifier, calls = counted_verifier
    messages = copy.deepcopy(CYCLE)
    call = messages[3]["tool_calls"][0]
    if not wrapped:
        call = call["function"]
    messages[3]["tool_calls"] = [copy.deepcopy(call) for _ in range(3)]
    verifier.verify_input_structure(messages, {})
    messages[3]["tool_calls"] = [copy.deepcopy(call) for _ in range(8)]
    changed = messages[3]["tool_calls"][5]
    function = changed["function"] if wrapped else changed
    function["arguments"] = arguments
    with pytest.raises(ValidationError, match="CHAT_TEMPLATE_INVALID"):
        verifier.verify_input_structure(messages, {})
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("omitted", "arguments"),
    [
        ("call", None),
        ("call", ""),
        ("generation", None),
        ("generation", ""),
        ("arguments", None),
        ("arguments", "{broken"),
    ],
)
def test_literal_arguments_require_supervised_output(
    fast_tokenizer: Any, omitted: str, arguments: Any
) -> None:
    from zephon._internal.utils.chat_template import TemplateVerifier

    messages = copy.deepcopy(CYCLE)
    messages[3]["tool_calls"][0]["function"]["arguments"] = arguments
    replacements = {
        "call": "",
        "generation": "{{m.tool_calls|tojson}}",
        "arguments": "{% generation %}{{m.tool_calls[0].function.name}}{% endgeneration %}",
    }
    template = SPLIT.replace(
        "{% generation %}{{m.tool_calls|tojson}}{% endgeneration %}",
        replacements[omitted],
    )
    code = (
        "CHAT_REQUIRED_SUPERVISION"
        if omitted == "generation"
        else "CHAT_TEMPLATE_INVALID"
    )
    with pytest.raises(ValidationError, match=code):
        TemplateVerifier(fast_tokenizer, template, {}).verify_input_structure(
            messages, {}
        )


def test_literal_argument_checks_keep_input_values_and_cache_exact_values(
    counted_verifier: Any,
) -> None:
    verifier, calls = counted_verifier
    messages = copy.deepcopy(CYCLE)
    function = messages[3]["tool_calls"][0]["function"]
    for arguments in (
        None,
        [],
        0,
        False,
        "",
        "null",
        "{broken",
        "{different",
        "{broken",
    ):
        function["arguments"] = arguments
        original = copy.deepcopy(messages)
        verifier.verify_input_structure(messages, {})
        assert messages == original
    assert len(calls) == 8


@pytest.mark.parametrize("arguments", [None, [], False, ""])
def test_literal_argument_branch_output_is_checked(
    fast_tokenizer: Any, arguments: Any
) -> None:
    from zephon._internal.utils.chat_template import TemplateVerifier

    messages = copy.deepcopy(CYCLE)
    messages[3]["tool_calls"][0]["function"]["arguments"] = arguments
    template = SPLIT.replace(
        "{% generation %}{{m.tool_calls|tojson}}{% endgeneration %}",
        "{% if m.tool_calls[0].function.arguments %}"
        "{% generation %}{{m.tool_calls|tojson}}{% endgeneration %}"
        "{% else %}{{m.tool_calls|tojson}}{% endif %}",
    )
    with pytest.raises(ValidationError, match="CHAT_REQUIRED_SUPERVISION"):
        TemplateVerifier(fast_tokenizer, template, {}).verify_input_structure(
            messages, {}
        )


def test_literal_argument_unrenderable_probe_fails_closed(fast_tokenizer: Any) -> None:
    from zephon._internal.utils.chat_template import TemplateVerifier

    messages = copy.deepcopy(CYCLE)
    messages[3]["tool_calls"][0]["function"]["arguments"] = None
    template = SPLIT.replace(
        "{{m.tool_calls|tojson}}",
        "{% if m.tool_calls[0].function.arguments is not none %}"
        "{{raise_exception('only null arguments supported')}}{% endif %}{{m.tool_calls|tojson}}",
    )
    with pytest.raises(ValidationError, match="only null arguments supported"):
        TemplateVerifier(fast_tokenizer, template, {}).verify_input_structure(
            messages, {}
        )


def test_literal_arguments_cannot_receive_another_assistants_loss(
    fast_tokenizer: Any,
) -> None:
    from zephon._internal.utils.chat_template import TemplateVerifier

    messages = copy.deepcopy(CYCLE)
    messages[3]["tool_calls"][0]["function"]["arguments"] = None
    template = SPLIT.replace(
        "{{m.content}}<|end|>",
        "{{m.content}}{% if loop.last %}{{messages[3].tool_calls|tojson}}{% endif %}<|end|>",
    )
    with pytest.raises(ValidationError, match="CHAT_CONTENT_OWNERSHIP"):
        TemplateVerifier(fast_tokenizer, template, {}).verify_input_structure(
            messages, {}
        )


@pytest.mark.parametrize("call", [None, "call", {}, {"function": None}])
def test_cached_formats_reject_malformed_calls(
    counted_verifier: Any, call: Any
) -> None:
    verifier, calls = counted_verifier
    verifier.verify_input_structure(CYCLE, {})
    messages = copy.deepcopy(CYCLE)
    messages[3]["tool_calls"] = [call]
    with pytest.raises(ValidationError, match="CHAT_TEMPLATE_INVALID"):
        verifier.verify_input_structure(messages, {})
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("content", 3),
        ("reasoning_content", []),
        ("content", [{"type": "image", "image": "x"}]),
        ("content", [{"type": "text", "text": 3}]),
    ],
)
def test_cached_formats_reject_invalid_message_content(
    counted_verifier: Any, field: str, value: Any
) -> None:
    verifier, calls = counted_verifier
    verifier.verify_input_structure(CYCLE, {})
    messages = copy.deepcopy(CYCLE)
    messages[3][field] = value
    with pytest.raises(ValidationError, match="CHAT_TEMPLATE_INVALID"):
        verifier.verify_input_structure(messages, {})
    assert len(calls) == 1


def test_custom_render_options_keep_exact_values(fast_tokenizer: Any) -> None:
    from zephon._internal.utils.chat_template import TemplateVerifier

    template = (
        "{% if mode == 'calls' %}"
        + SPLIT
        + "{% else %}"
        + TAGGED_TEMPLATE
        + "{% endif %}"
    )
    verifier = TemplateVerifier(fast_tokenizer, template, {})
    verifier.verify_input_structure(CYCLE, {"mode": "calls"})
    with pytest.raises(ValidationError, match="CHAT_REQUIRED_CONTENT"):
        verifier.verify_input_structure(CYCLE, {"mode": "text"})


def test_setup_result_survives_empty_process_caches(
    fast_tokenizer: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zephon._internal.utils import chat_template as module

    verifier = module.TemplateVerifier(
        copy.deepcopy(fast_tokenizer), TAGGED_TEMPLATE, {}
    )
    assert verifier.setup_report().ok
    saved = pickle.dumps(verifier.setup_result)
    module._VERIFIED_STRUCTURES.clear()
    module._owned_template.cache_clear()

    def unexpected_probe(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("verification probe ran")

    monkeypatch.setattr(module.TemplateVerifier, "verify", unexpected_probe)
    worker = module.TemplateVerifier(fast_tokenizer, TAGGED_TEMPLATE, {})
    assert worker.template is not verifier.template
    report = worker.setup_report(cached=pickle.loads(saved))
    assert report.ok
    assert worker.executed_sites == verifier.executed_sites
    assert worker.setup_result is not None
    report.issues.append(Issue("error", "TEST", "chat", "Caller-added issue"))
    assert worker.setup_report(cached=worker.setup_result).ok
    with pytest.raises(AssertionError, match="verification probe ran"):
        worker.verify_input_structure(CONVERSATION, {})


@pytest.mark.parametrize("changed", ["tokenizer", "template", "kwargs", "scope"])
def test_setup_result_requires_matching_configuration(
    fast_tokenizer: Any, monkeypatch: pytest.MonkeyPatch, changed: str
) -> None:
    from zephon._internal.utils import chat_template as module

    original = module.TemplateVerifier(
        copy.deepcopy(fast_tokenizer), TAGGED_TEMPLATE, {}
    )
    assert original.setup_report().ok
    tokenizer = copy.deepcopy(fast_tokenizer)
    template = TAGGED_TEMPLATE
    kwargs = {}
    scope = "after_last_user"
    if changed == "tokenizer":
        tokenizer.add_tokens(["different-vocabulary"])
    elif changed == "template":
        template += " "
    elif changed == "kwargs":
        kwargs["enable_thinking"] = True
    else:
        scope = "all_assistant"
    worker = module.TemplateVerifier(tokenizer, template, kwargs, scope)
    module._VERIFIED_STRUCTURES.clear()
    calls = []
    verify = module.TemplateVerifier.verify

    def counted(self: Any, *args: Any, **options: Any) -> Any:
        calls.append(1)
        return verify(self, *args, **options)

    monkeypatch.setattr(module.TemplateVerifier, "verify", counted)
    assert worker.setup_report(cached=original.setup_result).ok
    assert len(calls) == 2
    assert worker.setup_result != original.setup_result


def test_failed_setup_has_no_portable_result(fast_tokenizer: Any) -> None:
    from zephon._internal.utils.chat_template import TemplateVerifier

    template = TAGGED_TEMPLATE.replace("{{ message['content'] }}<|end|>", "<|end|>")
    verifier = TemplateVerifier(fast_tokenizer, template, {})
    assert not verifier.setup_report().ok
    assert verifier.setup_result is None


@pytest.mark.parametrize("in_message_loop", [False, True])
def test_tool_schema_loop_can_rebind_its_target(
    fast_tokenizer: Any, in_message_loop: bool
) -> None:
    schema = (
        "{% for tool in tools %}{% set tool = tool.function %}"
        "{{ tool.name }}{% endfor %}"
    )
    template = (
        TAGGED_TEMPLATE.replace(
            "{% for message in messages %}", "{% for message in messages %}" + schema
        )
        if in_message_loop
        else schema + TAGGED_TEMPLATE
    )
    report = _verify_chat_template(
        fast_tokenizer,
        template,
        chat_template_kwargs={"tools": [{"function": {"name": "lookup"}}]},
    )
    assert report.ok, report.format()


@pytest.mark.parametrize("field", ["thinking", "reasoning", "reasoning_content"])
def test_selected_reasoning_field_must_receive_loss(
    fast_tokenizer: Any, field: str
) -> None:
    messages = copy.deepcopy(CONVERSATION)
    messages[-1][field] = "private reasoning"
    report = _verify_chat_template(
        fast_tokenizer, TAGGED_TEMPLATE, probes=[{"messages": messages}]
    )
    assert any(issue.code == "CHAT_REQUIRED_CONTENT" for issue in report.errors)
    tagged = TAGGED_TEMPLATE.replace(
        "{% generation %}", "{% generation %}{{ message." + field + " | default('') }}"
    )
    report = _verify_chat_template(
        fast_tokenizer, tagged, probes=[{"messages": messages}]
    )
    assert report.ok, report.format()


@pytest.mark.parametrize(
    ("predicate", "normal", "trigger"),
    [
        ("message.content.startswith('SKIP:')", "answer SKIP:", "SKIP: answer"),
        ("message.content.endswith(':SKIP')", ":SKIP answer", "answer :SKIP"),
        ("'SKIP:' in message.content", "ordinary answer", "answer SKIP: answer"),
    ],
)
def test_content_controls_detect_unmasked_text_with_warm_cache(
    fast_tokenizer: Any, predicate: str, normal: str, trigger: str
) -> None:
    from zephon._internal.ops.tokenize_chat import TokenizeChat
    from zephon.validation import ValidationError

    template = TAGGED_TEMPLATE.replace(
        "{% generation %}{{ message['content'] }}<|end|>{% endgeneration %}",
        "{% if " + predicate + " %}{{ message.content }}{% endif %}"
        "{% generation %}{% if not (" + predicate + ") %}{{ message.content }}"
        "{% endif %}<|end|>{% endgeneration %}",
    )
    op = TokenizeChat(
        fast_tokenizer, chat_template=template, loss_scope="after_last_user"
    )
    messages = copy.deepcopy(CONVERSATION)
    messages[-1]["content"] = normal
    op._tokenize_delivery({"messages": messages}, "warmup")
    messages[-1]["content"] = trigger
    with pytest.raises(ValidationError, match="CHAT_REQUIRED_SUPERVISION"):
        op._tokenize_delivery({"messages": messages}, "trigger")


@pytest.mark.parametrize(
    "condition",
    [
        "message.content == 'COPY_PROMPT'",
        "message.content | length > 5",
        "message.content.startswith(prefix)",
        "message.content.endswith('SKIP', 2)",
    ],
)
def test_unpreservable_text_conditions_fail_setup(
    fast_tokenizer: Any, condition: str
) -> None:
    from zephon._internal.ops.tokenize_chat import TokenizeChat

    template = TAGGED_TEMPLATE.replace(
        "{% endgeneration %}",
        "{% if "
        + condition
        + " %}{{ messages[0].content }}{% endif %}{% endgeneration %}",
    )
    report = _verify_chat_template(fast_tokenizer, template)
    assert not report.ok
    op = TokenizeChat(
        fast_tokenizer, chat_template=template, loss_scope="after_last_user"
    )
    with pytest.raises(ValueError, match="content.*(comparison|prefix/suffix)"):
        op._tokenize_delivery({"messages": CONVERSATION}, "without-preflight")


def test_content_alias_comparison_in_assignment_fails_setup(
    fast_tokenizer: Any,
) -> None:
    template = TAGGED_TEMPLATE.replace(
        "{% generation %}",
        "{% set value = message.content %}{% set copy_prompt = value == 'COPY_PROMPT' %}"
        "{% generation %}{% if copy_prompt %}{{ messages[0].content }}{% endif %}",
    )
    report = _verify_chat_template(fast_tokenizer, template)
    assert not report.ok
    assert "content-dependent comparison" in report.format()


def test_verifier_requires_stock_identity_and_marker_coverage(
    fast_tokenizer: Any,
) -> None:
    report = _verify_chat_template(
        fast_tokenizer, TAGGED_TEMPLATE, stock_template=UNTAGGED_TEMPLATE
    )
    assert report.ok, report.format()
    report = _verify_chat_template(
        fast_tokenizer,
        TAGGED_TEMPLATE,
        stock_template=UNTAGGED_TEMPLATE + "extra",
    )
    assert not report.ok
    assert any(i.code == "CHAT_STOCK_IDENTITY" for i in report.errors)
    report = _verify_chat_template(
        fast_tokenizer, TAGGED_TEMPLATE, probes=[{"messages": CYCLE}]
    )
    assert any(i.code == "CHAT_REQUIRED_CONTENT" for i in report.errors)


def test_probe_coverage_uses_requested_scope(fast_tokenizer: Any) -> None:
    messages = copy.deepcopy(CYCLE)
    messages.extend(
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "yes"},
        ]
    )
    report = _verify_chat_template(
        fast_tokenizer,
        TAGGED_TEMPLATE,
        loss_scope="after_last_user",
        probes=[{"messages": messages}],
    )
    assert report.ok, report.format()
    messages[-1]["reasoning_content"] = "yes"
    report = _verify_chat_template(
        fast_tokenizer,
        TAGGED_TEMPLATE,
        loss_scope="after_last_user",
        probes=[{"messages": messages}],
    )
    assert any(issue.code == "CHAT_REQUIRED_CONTENT" for issue in report.errors)


def test_chat_verifier_rejects_invalid_scope(fast_tokenizer: Any) -> None:
    report = _verify_chat_template(
        fast_tokenizer, TAGGED_TEMPLATE, loss_scope="last_assistant"
    )
    assert not report.ok
    assert any("invalid loss_scope" in issue.message for issue in report.errors)
