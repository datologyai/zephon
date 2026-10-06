"""Give each operator the compute it needs."""

from typing import Any

from zephon import Pipeline


def clean_text(payload: Any) -> dict[str, Any]:
    """Strip the surrounding whitespace the crawler left behind."""
    return {"text": payload["text"].strip()}


# Every operator but batch takes parallelism. More workers than the training
# loop can consume costs resources without buying throughput.
pipeline = (
    Pipeline(work_source)
    .map_transform(clean_text, parallelism=4)
    .tokenize(tokenizer_id="gpt2", field="text", parallelism=16)
    .batch(microbatch_size=32)
)
