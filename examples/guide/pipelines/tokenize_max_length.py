"""Cap token sequence length by splitting rather than truncating."""

from zephon import Pipeline

# truncation=True would discard the tail instead; split_long_samples keeps it
# as further records, each at most max_length tokens.
pipeline = Pipeline(work_source).tokenize(
    tokenizer_id="gpt2",
    field="text",
    max_length=2048,
    split_long_samples=True,
)
