"""Save the Pipeline's progress with the rest of the training state."""

from zephon import Pipeline

pipeline = (
    Pipeline(work_source)
    .tokenize(tokenizer_id="gpt2", field="text")
    .pack_flat(max_length=4097, algorithm="wrap", pad_token_id=0)
    .batch(microbatch_size=32)
)

for step, sample_batch in enumerate(pipeline):
    train_step(model, sample_batch.to_training(return_labels=True))
    if step % 1000 == 0:
        # One object, written together: a pipeline state that describes a
        # different step than the model's will silently skip or repeat data.
        save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "data": pipeline.checkpoint(),
            }
        )
