"""Attach an operator class by passing an instance."""

from zephon import Pipeline

pipeline = (
    Pipeline(work_source)
    .decode_text()
    .add_op(QualityScoreOp("quality-classifier-v2"))
    .batch(microbatch_size=32)
)
