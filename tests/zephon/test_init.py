from zephon import Pipeline
from zephon.pipeline import Pipeline as PipelineImpl


def test_api_alias_resolves_pipeline() -> None:
    assert Pipeline is PipelineImpl
