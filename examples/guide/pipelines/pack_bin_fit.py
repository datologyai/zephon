"""Pack without ever splitting a record."""

from zephon import Pipeline

# num_bins partially filled sequences stay open; first_fit takes the first bin
# with room, best_fit the one left with the least. Records longer than
# max_length fit nowhere and are dropped, and each flush emits every open bin.
pipeline = (
    Pipeline(work_source)
    .tokenize(tokenizer_id="gpt2", field="text")
    .pack_flat(max_length=4097, algorithm="first_fit", num_bins=8, pad_token_id=0)
    .batch(microbatch_size=8)
)
