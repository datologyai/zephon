"""Decode byte payloads to str before tokenizing them."""

from zephon import Pipeline

pipeline = (
    Pipeline(work_source)
    .decode_text()
    .tokenize(tokenizer_id="gpt2", field="text")
    .batch(microbatch_size=32)
)
