# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Tests that verify examples in the examples/ directory run successfully.

These tests import and run the main() functions from each example to ensure
the documentation examples stay in sync with the codebase.
"""


def _assert_valid_batch(batch):
    """Verify batch has expected structure to catch API drift."""
    # Batch should have expected SampleBatch attributes
    assert hasattr(batch, "records"), "Batch missing 'records' attribute"
    assert hasattr(batch, "ids"), "Batch missing 'ids' property"
    assert len(batch) > 0, "Batch should not be empty"
    assert isinstance(batch.records, tuple), "Batch.records should be a tuple"
    assert len(batch.ids) == len(batch), "Batch.ids length should match batch length"


def test_run_basic():
    """Test that run_basic.py example runs without error."""
    from examples.run_basic import build_pipeline

    # Build and iterate the pipeline
    pipe = build_pipeline()
    count = 0
    for batch in pipe:
        count += 1
        _assert_valid_batch(batch)
    assert count > 0


def test_run_jsonl():
    """Test that run_jsonl.py example runs without error."""
    from examples.run_jsonl import build_pipeline

    pipe = build_pipeline()
    count = 0
    for batch in pipe:
        count += 1
        _assert_valid_batch(batch)
    assert count > 0


def test_run_mixture():
    """Test that run_mixture.py example runs without error."""
    from examples.run_mixture import build_pipeline

    pipe = build_pipeline()
    count = 0
    for batch in pipe:
        count += 1
        _assert_valid_batch(batch)
    assert count > 0


def test_run_with_prefetch():
    """Test that run_with_prefetch.py example runs without error."""
    from examples.run_with_prefetch import build_pipeline_with_prefetch

    pipe = build_pipeline_with_prefetch()
    count = 0
    for batch in pipe:
        count += 1
        _assert_valid_batch(batch)
    assert count > 0
