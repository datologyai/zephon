"""Apply a chat template and mark which tokens the loss sees."""

from zephon import Pipeline

# loss_mask marks the assistant turns, so the model learns to answer rather
# than to predict the user's messages.
pipeline = (
    Pipeline(work_source)
    .tokenize_chat(
        tokenizer_id="Qwen/Qwen3-0.6B",
        field="messages",
        mask_field_out="loss_mask",
    )
    .batch(microbatch_size=16)
)
