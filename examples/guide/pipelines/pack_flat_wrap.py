"""Fill every sequence completely, splitting records where needed."""

from zephon import Pipeline

# A FIFO stream of tokens cut at max_length. Tokens left over at a flush that
# cannot fill a sequence are dropped, and Zephon logs how many.
pipeline = (
    Pipeline(work_source)
    .tokenize(tokenizer_id="gpt2", field="text")
    .pack_flat(max_length=4097, algorithm="wrap", pad_token_id=0)
    .batch(microbatch_size=8)
)
