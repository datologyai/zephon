"""Rename a field, and drop short records, before tokenizing."""

from typing import Any

from zephon import Pipeline


def to_text(payload: Any) -> dict[str, Any] | None:
    """Keep the body under a standard name; returning None drops the record."""
    body = payload.get("body", "")
    return {"text": body} if len(body) >= 32 else None


# The function has to be deterministic -- no globals, no unseeded randomness --
# or replay after a checkpoint will not reproduce the same stream.
pipeline = (
    Pipeline(work_source)
    .map_transform(to_text)
    .tokenize(tokenizer_id="gpt2", field="text")
)
