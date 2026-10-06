"""Concatenate tokenized records into fixed-length sequences."""

from zephon import Pipeline

# max_length is the model's context length plus one, so to_training can shift
# the labels by a position and still fill the context.
pipeline = (
    Pipeline(work_source)
    .tokenize(tokenizer_id="gpt2", field="text")
    .pack_flat(max_length=4097, algorithm="wrap", pad_token_id=0)
)

for record in pipeline:
    # input_ids, attention_mask, and positions, which resets to zero at each
    # document the sequence contains.
    print(sorted(record.payload))
