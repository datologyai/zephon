# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Argument types for ``Pipeline`` operator-configuration methods."""

from typing import Literal, TypeAlias

SpecialTokensMode: TypeAlias = Literal[
    "bos_eos", "bos", "eos", "none", "tokenizer_default"
]

MissingFieldMode: TypeAlias = Literal["error", "empty"]

SpanSource: TypeAlias = Literal["auto", "generation_tags", "prefix_diff"]

PackingAlgorithm: TypeAlias = Literal["first_fit", "best_fit", "wrap", "best_fit_wrap"]

__all__ = ["MissingFieldMode", "PackingAlgorithm", "SpanSource", "SpecialTokensMode"]
