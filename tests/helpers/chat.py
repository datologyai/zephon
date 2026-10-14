# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Offline chat tokenizer, templates and conversation fixtures."""

from typing import Any

import pytest

transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")


_WORDS = (
    "hello hi there how are you good morning the answer is what thanks "
    "yes no 1 2 4 12 + ? ! . w"
).split()

#: Both templates render to the identical string; only span provenance differs.
UNTAGGED_TEMPLATE = (
    "{% for message in messages %}<|{{ message['role'] }}|>{{ message['content'] }}"
    "<|end|>{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}"
)
TAGGED_TEMPLATE = (
    "{% for message in messages %}{% if message['role'] == 'assistant' %}"
    "<|assistant|>{% generation %}{{ message['content'] }}<|end|>{% endgeneration %}"
    "{% else %}<|{{ message['role'] }}|>{{ message['content'] }}<|end|>{% endif %}"
    "{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}"
)
#: Assistant turns end without EOS — exercises the conditional EOS finalizer.
TAGGED_TEMPLATE_NO_EOS = (
    "{% for message in messages %}{% if message['role'] == 'assistant' %}"
    "<|assistant|>{% generation %}{{ message['content'] }}{% endgeneration %}"
    "{% else %}<|{{ message['role'] }}|>{{ message['content'] }}<|end|>{% endif %}"
    "{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}"
)
#: The trailing marker moves with the conversation length, so partial renders
#: are not prefixes of the full render.
POSITION_DEPENDENT_TEMPLATE = (
    "{% for message in messages %}<|{{ message['role'] }}|>{{ message['content'] }}"
    "{% if loop.last %}<|sep|>{% endif %}<|end|>{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)
TOOLS_TEMPLATE = (
    "{% if tools %}<|sep|>{% endif %}{% if enable_thinking %}<|sep|><|sep|>{% endif %}"
    + UNTAGGED_TEMPLATE
)


def _messages(*items: tuple[str, Any]) -> list[dict[str, Any]]:
    return [{"role": role, "content": content} for role, content in items]


CONVERSATION = _messages(
    ("user", "hi there"),
    ("assistant", "good morning"),
    ("user", "what is 2 + 2 ?"),
    ("assistant", "the answer is 4"),
)

GOLDEN_TOKENS = (
    "<|user|> hi there <|end|> <|assistant|> good morning <|end|> "
    "<|user|> what is 2 + 2 ? <|end|> <|assistant|> the answer is 4 <|end|>"
).split()
GOLDEN_MASK = [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]


def make_fast_tokenizer() -> Any:
    vocab = {"[UNK]": 0}
    for word in _WORDS:
        vocab[word] = len(vocab)
    tok = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    fast = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="[UNK]",
        eos_token="<|end|>",
        pad_token="<|pad|>",
    )
    fast.add_special_tokens(
        {
            "additional_special_tokens": [
                "<|user|>",
                "<|assistant|>",
                "<|system|>",
                "<|sep|>",
            ]
        }
    )
    return fast


SPLIT = (
    "{% for m in messages %}<|{{m.role}}|>{% if m.role == 'assistant' %}"
    "{% if m.reasoning_content %}{% generation %}{{m.reasoning_content}}{% endgeneration %}<|sep|>{% endif %}"
    "{% if m.tool_calls %}{% generation %}{{m.tool_calls|tojson}}{% endgeneration %}<|sep|>{% endif %}"
    "{% if m.content %}{% generation %}{{m.content}}<|end|>{% endgeneration %}{% endif %}"
    "{% else %}{{m.content}}<|end|>{% endif %}{% endfor %}"
)
CYCLE = [
    {"role": "user", "content": "hello"},
    {"role": "assistant", "content": "hi"},
    {"role": "user", "content": "what"},
    {
        "role": "assistant",
        "content": "",
        "reasoning_content": "how",
        "tool_calls": [
            {
                "type": "function",
                "function": {"name": "lookup", "arguments": {"question": "what"}},
            }
        ],
    },
    {"role": "tool", "content": "12"},
    {"role": "assistant", "content": "the answer", "reasoning_content": "yes"},
]
