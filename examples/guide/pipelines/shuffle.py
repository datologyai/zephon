"""Shuffle the record stream itself, across datasets."""

from zephon import Pipeline

# A larger buffer is a better shuffle for more memory; the whole buffer is
# emitted at every flush boundary.
pipeline = (
    Pipeline(work_source)
    .tokenize(tokenizer_id="gpt2", field="text")
    .shuffle(buffer_size=16384, seed=42)
    .batch(microbatch_size=32)
)
