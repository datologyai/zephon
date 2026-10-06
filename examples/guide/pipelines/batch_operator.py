"""Group records into SampleBatch objects."""

from zephon import Pipeline

# The default drops incomplete batches. drop_last=False keeps them, which in a
# pipeline that flushes means a short batch at every flush, not just at the end.
pipeline = (
    Pipeline(work_source)
    .tokenize(tokenizer_id="gpt2", field="text")
    .batch(microbatch_size=32, drop_last=True)
)

for sample_batch in pipeline:
    print(len(sample_batch.records), sample_batch.ids)
