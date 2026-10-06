"""Pack records while keeping each document reachable."""

from zephon import Pipeline

pipeline = (
    Pipeline(work_source)
    .tokenize(tokenizer_id="gpt2", field="text")
    .pack_sequences(max_length=4097, num_bins=8, algorithm="first_fit")
)

for record in pipeline:
    # packed_samples, not an assembled sequence: to_training cannot convert
    # this, which is the trade for reaching the individual documents.
    print(len(record.payload["packed_samples"]))
