"""Tokenize web documents and chat traces together in one stream."""

from zephon import Pipeline

# Records with a "messages" field are tokenized like tokenize_chat and supervise
# the assistant's turns; all other records are tokenized from "text" like
# tokenize and supervise every token. Each path keeps its operator's defaults,
# except that Qwen tokenizers have no BOS, so text documents only get an EOS.
pipeline = (
    Pipeline(work_source)
    .tokenize_mixed(
        tokenizer_id="Qwen/Qwen3-0.6B",
        max_length=4097,
        text_special_tokens="eos",
        split_long_samples=True,
        loss_scope="all_assistant",
    )
    .pack_flat(max_length=4097, algorithm="wrap")
    .batch(microbatch_size=16)
)
