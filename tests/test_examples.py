# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Tests that verify examples in the examples/ directory run successfully.

These tests import and run the main() functions from each example to ensure
the documentation examples stay in sync with the codebase.
"""


def test_run_basic(capsys):
    """Test that run_basic.py example runs without error."""
    from examples.run_basic import main

    main()
    captured = capsys.readouterr().out
    assert "PLAN:" in captured
    assert "SUMMARY:" in captured


def test_run_jsonl(capsys):
    """Test that run_jsonl.py example runs without error."""
    from examples.run_jsonl import main

    main()
    captured = capsys.readouterr().out
    assert "PLAN:" in captured
    assert "SUMMARY:" in captured


def test_run_mixture(capsys):
    """Test that run_mixture.py example runs without error."""
    from examples.run_mixture import main

    main()
    captured = capsys.readouterr().out
    assert "PLAN:" in captured
    assert "SUMMARY:" in captured


def test_run_with_prefetch(capsys):
    """Test that run_with_prefetch.py example runs without error."""
    from examples.run_with_prefetch import main

    main()
    captured = capsys.readouterr().out
    assert "Zephon Prefetch Example" in captured
    assert "COMPARISON" in captured


def test_run_your_first_pipeline(capsys):
    """Test that your_first_pipeline.py example runs without error."""
    from examples.your_first_pipeline import main

    main()
    captured = capsys.readouterr().out
    assert "Your First Zephon Pipeline" in captured
    assert "Done! You processed 3 samples in 2 batch(es)." in captured
