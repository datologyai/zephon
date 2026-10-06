"""Print the execution plan Zephon compiled for a Pipeline."""

from zephon import Pipeline

pipeline = (
    Pipeline(work_source)
    .tokenize(tokenizer_id="gpt2", field="text", parallelism=4)
    .batch(microbatch_size=32)
)

# Each operator with its parallelism (tokenize@p4), the queue sizes between
# operators and stages, and how far ahead batches are prepared.
print(pipeline.explain())
