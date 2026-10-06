# Training Integrations

When we are initially developing or testing out our data pipeline, it's easiest to treat a
Zephon `Pipeline` as a regular Python iterable that produces batches of data that we can
manually inspect and debug. But as we discussed in
[Distributed Training](pipelines/distributed_training.md), integrating our data pipeline
into a distributed model training framework requires us to align the two at a few specific
points to ensure that the model receives the data in the right form and that the data
loader and the model always agree on where the training run is. Here we will discuss these
integration points in the context of our reference integrations for
[TorchTitan](https://github.com/pytorch/torchtitan) and
[Megatron-LM](https://github.com/NVIDIA/Megatron-LM). Each integration's repository
([torchtitan-zephon](https://github.com/datologyai/torchtitan-zephon/tree/main/examples/zephon) and
[Megatron-LM-zephon](https://github.com/datologyai/Megatron-LM-zephon/tree/main/examples/zephon)) has a
README with installation instructions, launch commands, and smoke tests to
get you up and running quickly, so the purpose of this page is to explain the decisions we
had to make in order to make the integrations work so that you can adapt them to your own
setup or build a similar integration for your own framework.

## Where Zephon Meets the Training Loop

Here is the idealized training loop from earlier in the guide, with annotations for the
places where the data loader and the training framework need to agree with each other:

```python
pipeline = build_pipeline(recipe, seq_len, tokenizer, topology)    # settings the framework owns
if resuming:
    pipeline.restore(checkpoint["data"])                           # restore with the model

for step, batch in enumerate(pipeline):
    loss = train_step(model, batch.to_training(...))               # the batch the model expects
    if step % save_every == 0:
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "data": pipeline.checkpoint(),                         # save with the model
        }
```

Let's discuss each of the integration points here in turn.

**Building the Pipeline.** The first place the two sides meet is when the `Pipeline` is
built, because some of the settings that we need for configuring the `Pipeline` are also
required for the model training loop, and so we need to pull them from the model training
framework. The other settings, like the datasets, the mixture weights, and the knobs that
we expect to experiment with are all part of the data recipe that do not depend on the
training framework directly, and so we can configure them in a TOML file that is identical
across both our TorchTitan and Megatron-LM integrations:

```{literalinclude} ../../examples/guide/integrations/data_recipe.toml
:language: toml
:caption: examples/guide/integrations/data_recipe.toml
```

The data recipe does not include which tokenizer we're using, the sequence length, or any
information about the topology because these things are already configured for the
training framework itself, and we do not want to risk introducing additional configuration
options for Zephon that could be allowed to drift from the ones the training loop uses.
Instead, in each of the integrations, the configuration code reads a combination of the
data recipe from the TOML file and the subset of the model framework's configuration
settings that it needs for the other things that the `Pipeline` needs to know. The
configuration code is also the natural place for any and all validation checks that we
want to do to ensure that the `Pipeline` will be constructed correctly so that a
misconfigured run fails immediately instead of silently failing or inadvertently losing
its elastic guarantees many hours later. For example, both of our reference integrations
have checks that will fail immediately if the `canonical_replicas` setting is not
divisible by the data-parallel size, or if an optimizer step would consume something other
than a multiple of `canonical_replicas` batches.

In the TorchTitan integration, the Zephon data loader lives in the `torchtitan_recipes.zephon`
package, and a TorchTitan config recipe opts into it by setting its `dataloader` field (and,
if it runs validation, its validator's `dataloader` field) to a `ZephonDataLoader.Config`
built from the data recipe with `ZephonDataLoader.Config.from_toml`. The rest of the config
recipe is ordinary TorchTitan, and every rank in the job builds its own `Pipeline` instance.
In Megatron, we build the Zephon `Pipeline` via a replacement for the
"dataset provider" function in the `pretrain_gpt_zephon.py` entry point, and only the
subset of ranks that consume the batch need to build one, since Megatron broadcasts each
batch to the other tensor-parallel ranks.

**Shaping Each Batch.** The second integration point is in the shape of the batch itself,
provided by the `SampleBatch.to_training` method. Every framework expects its batches in a
particular form with the field names and types that match what the model is expecting.
[Building Training Batches](pipelines/building_training_batches.md) explains how to map a
packed sequence of tokens into the format that the model expects, and the job of the
integration code is to wire up the settings from the training framework that are necessary
for this to work into the `to_training` method's arguments.

TorchTitan expects one flat sequence of tokens in each microbatch, with per-document
positions, a padding mask, and a count of the number of tokens in the batch that count
toward the loss. Megatron expects `[micro_batch_size, sequence_length]` shaped tensors and
a float loss mask, and then it has a number of different training options that can change
the form of the positions and attention masks depending on the model that is being
trained.

**Saving and Restoring.** The last place where we need to ensure alignment between the
training framework and Zephon is in checkpointing and restoring: we need to guarantee that
the data loader's state describes exactly the same point in training as the model's state.
If the two inadvertently drift apart, a resumed run will skip or repeat data without any
error information to tell you that this happened.

The simplest way to guarantee that the two are in sync is to store the output of the
`Pipeline.checkpoint()` method in the same checkpoint object that has the state of the
model and the optimizer, so that they are always written and read together. This is what
the TorchTitan integration does: the TorchTitan data loader abstraction supports the same
`state_dict` and `load_state_dict` methods as the model, and then the framework stores the
Zephon checkpoint payload inside of TorchTitan's checkpoint object as an opaque value.

If your framework keeps the data loader checkpoint separate from the model checkpoint, as
Megatron does, make sure that a resumed run can only ever pair the model state with the
data loader state that was saved at the same step. Megatron writes the Zephon checkpoint
into a per-iteration directory alongside the model checkpoint, and on resume our
integration looks for the data loader state only in the directory for the iteration that
the model checkpoint came from, and refuses to start if it isn't there, rather than
quietly starting the data over from the beginning.

**Building Your Own Integration.** If you are integrating Zephon with a different training
framework, you should start from whichever reference integration's approach to ranks and
data loaders is closer to yours, and then work through configuring the `Pipeline`, shaping
the batch that gets handed off to the training loop, and ensuring consistent checkpoint
save/restore semantics. We recommend iterating over a `Pipeline` directly rather than
wrapping it inside of a PyTorch `DataLoader` unless your framework absolutely requires
one.
