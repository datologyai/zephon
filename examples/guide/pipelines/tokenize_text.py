"""Turn a text field into token ids."""

from zephon import Pipeline

pipeline = (
    Pipeline(work_source)
    .tokenize(tokenizer_id="gpt2", field="text", parallelism=8)
    .batch(microbatch_size=32)
)
