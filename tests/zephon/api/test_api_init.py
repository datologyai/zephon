from zephon.api import Pipeline
from zephon.api.pipeline import Pipeline as PipelineImpl


def test_api_alias_resolves_pipeline() -> None:
    assert Pipeline is PipelineImpl
