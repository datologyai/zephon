"""Resume a run from a saved checkpoint."""

from zephon import Pipeline

# The same datasets, work source and operators as the original run, down to
# the tokenizer and every transform parameter.
pipeline = (
    Pipeline(work_source)
    .tokenize(tokenizer_id="gpt2", field="text")
    .pack_flat(max_length=4097, algorithm="wrap", pad_token_id=0)
    .batch(microbatch_size=32)
)

pipeline.restore(checkpoint["data"])  # before the first iteration, not after

for sample_batch in pipeline:
    train_step(model, sample_batch.to_training(return_labels=True))
