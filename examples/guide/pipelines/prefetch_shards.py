"""Download shards before the fetch operator needs to open them."""

from zephon import Pipeline

# buffer_size counts samples to look ahead by. Looking too far ahead pressures
# the shard cache and can evict shards that are still needed.
pipeline = (
    Pipeline(work_source)
    .prefetch(buffer_size=4096, parallelism=8)
    .tokenize(tokenizer_id="gpt2", field="text")
    .batch(microbatch_size=32)
)
