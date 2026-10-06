"""Buffer finished batches ahead of the training loop."""

from zephon import Pipeline

# Absorbs short variations in preparation time; it cannot help when the
# pipeline is simply slower than the model.
pipeline = (
    Pipeline(work_source)
    .tokenize(tokenizer_id="gpt2", field="text")
    .batch(microbatch_size=32)
    .options(prefetch_batches=8)
)
