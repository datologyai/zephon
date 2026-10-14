# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

import copy
from typing import Any

import pytest

from tests.helpers.chat import (
    CONVERSATION,
    SPLIT,
    TAGGED_TEMPLATE,
    make_fast_tokenizer,
)
from zephon.validation import verify_chat_template


@pytest.fixture
def fast_tokenizer() -> Any:
    return make_fast_tokenizer()


def test_generation_sites_on_same_line_are_distinct(fast_tokenizer: Any) -> None:
    from zephon._internal.utils.chat_template import OwnedTemplate

    assert len(OwnedTemplate(SPLIT).sites) == 3
    report = verify_chat_template(fast_tokenizer, SPLIT)
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
