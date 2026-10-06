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
    """Test the real-tokenizer example without requiring network access."""
    from examples.your_first_pipeline import main

    main(tokenizer_id="__fallback__")
    captured = capsys.readouterr().out
    assert "Your First Zephon Pipeline" in captured
    assert "tokenize@p1" in captured
    assert "batch@p1" in captured
    assert "Batch 0: 2 samples, shape=(2," in captured
    assert "texts=" in captured


def test_first_pipeline_warns_and_uses_fallback_without_transformers(
    monkeypatch, recwarn
):
    """Keep the base-install path runnable without Transformers."""
    from examples import your_first_pipeline

    monkeypatch.setattr(your_first_pipeline, "find_spec", lambda _name: None)

    assert your_first_pipeline.default_tokenizer_id() == "__fallback__"
    assert "Transformers is not installed" in str(recwarn.pop(RuntimeWarning).message)
