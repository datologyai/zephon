"""Collect runtime metrics while a Pipeline runs."""

from zephon import Pipeline

# Tracking costs throughput, especially under the process runner, so turn it
# on while tuning and leave it off in production.
pipeline = Pipeline(work_source).batch(microbatch_size=32).enable_observability()

for step, sample_batch in enumerate(pipeline):
    if step % 100 == 0:
        print(pipeline.metrics_snapshot())
        print(pipeline.fetch_timing_snapshot())
