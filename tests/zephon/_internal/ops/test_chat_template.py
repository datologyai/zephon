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


def test_endpoint_mapping_includes_hf_stopping_and_zero_index_rules() -> None:
    from zephon._internal.ops.chat_template import generation_mask

    class Encoding(dict):
        def char_to_token(self, batch: int, char: int) -> int | None:
            return {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2}.get(char)

    encoding = Encoding(input_ids=[10, 20, 30])
    assert generation_mask(encoding, [(1, 0, 1, 1)], {1}).tolist() == [1, 1, 1]
    assert generation_mask(encoding, [(0, 20, 21, 1), (1, 2, 4, 2)], {1}).tolist() == [
        0,
        0,
        0,
    ]
    assert generation_mask(encoding, [(1, 3, 3, 1)], {1}).tolist() == [0, 1, 0]


def test_generation_sites_on_same_line_are_distinct(fast_tokenizer: Any) -> None:
    from zephon._internal.ops.chat_template import OwnedTemplate

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
    from zephon._internal.ops.chat_template import TemplateVerifier

    verifier = TemplateVerifier(fast_tokenizer, TAGGED_TEMPLATE, {})

    def broken(*args, **kwargs):
        raise error_type("programming error")

    monkeypatch.setattr(verifier.template, "render", broken)
    with pytest.raises(error_type, match="programming error"):
        verifier.verify(CONVERSATION, {})


def test_verification_cache_shared_and_lru(
    fast_tokenizer: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zephon._internal.ops import chat_template as module

    monkeypatch.setattr(module, "_PROFILE_LIMIT", 2)
    module._PROFILES.clear()
    verifier = module.TemplateVerifier(fast_tokenizer, TAGGED_TEMPLATE, {})
    calls = []
    verify = module.TemplateVerifier.verify

    def counted(self, *args, **kwargs):
        calls.append(1)
        return verify(self, *args, **kwargs)

    monkeypatch.setattr(module.TemplateVerifier, "verify", counted)
    verifier.require_profile(CONVERSATION, {})
    other = module.TemplateVerifier(copy.deepcopy(fast_tokenizer), TAGGED_TEMPLATE, {})
    other.require_profile(CONVERSATION, {})
    assert len(calls) == 1
    verifier.require_profile(CONVERSATION, {"flag": True})
    verifier.require_profile(CONVERSATION, {})  # refresh this entry
    verifier.require_profile(CONVERSATION, {"flag": False})
    verifier.require_profile(CONVERSATION, {})
    assert len(calls) == 3
    verifier.require_profile(CONVERSATION, {"flag": True})
    assert len(calls) == 4
    module._PROFILES.clear()


@pytest.mark.parametrize("target", ["message", "(other, message)"])
def test_with_cannot_rebind_message_owner(target: str) -> None:
    from zephon._internal.ops.chat_template import OwnedTemplate

    template = TAGGED_TEMPLATE.replace(
        "{% generation %}",
        "{% with " + target + " = messages[-1] %}{% generation %}",
    ).replace("{% endgeneration %}", "{% endgeneration %}{% endwith %}")
    with pytest.raises(ValueError, match="rebinding"):
        OwnedTemplate(template)


def test_cached_sites_belong_only_to_the_cached_profile(fast_tokenizer: Any) -> None:
    from zephon._internal.ops.chat_template import TemplateVerifier

    template = TAGGED_TEMPLATE.replace(
        "{% generation %}{{ message['content'] }}<|end|>{% endgeneration %}",
        "{% if flag %}{% generation %}{{ message.content }}<|end|>{% endgeneration %}"
        "{% else %}{% generation %}{{ message.content }}<|end|>{% endgeneration %}{% endif %}",
    )
    first = TemplateVerifier(fast_tokenizer, template, {})
    first.require_profile(CONVERSATION, {"flag": False})
    first.require_profile(CONVERSATION, {"flag": True})
    assert first.executed_sites == {0, 1}
    second = TemplateVerifier(fast_tokenizer, template, {})
    second.require_profile(CONVERSATION, {"flag": True})
    assert second.executed_sites == {0}


def test_unknown_tokenizer_cannot_share_verification_results() -> None:
    from zephon._internal.ops.chat_template import TemplateVerifier

    with pytest.raises(ValueError, match="fast Hugging Face tokenizer"):
        TemplateVerifier(object(), TAGGED_TEMPLATE, {})


def test_tokenizer_mutation_changes_verification_identity(fast_tokenizer: Any) -> None:
    from zephon._internal.ops.chat_template import TemplateVerifier

    before = TemplateVerifier(fast_tokenizer, TAGGED_TEMPLATE, {})._identity
    fast_tokenizer.add_tokens(["new-token-for-verifier"])
    after = TemplateVerifier(fast_tokenizer, TAGGED_TEMPLATE, {})._identity
    assert before != after
