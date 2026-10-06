"""Strip boilerplate from a document, and drop what is left if it is too short."""

from typing import Any

from zephon import Pipeline


def clean_document(payload: Any) -> dict[str, Any] | None:
    """Remove navigation boilerplate; return None to drop the document."""
    text = payload["text"].replace("Home | About | Contact", "").strip()
    return {**payload, "text": text} if len(text) >= 200 else None


pipeline = Pipeline(work_source).decode_text().map_transform(clean_document)
