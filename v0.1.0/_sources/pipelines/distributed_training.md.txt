# Distributed Training

Most of the examples in this user guide (well, probably all of them, now that I think
about it) have used a single `Pipeline` feeding a single training loop. Real training runs
are spread across many GPUs, and the training job needs to divide the model and the data
across those GPUs using some combination of data parallelism (DP), tensor parallelism
(TP), pipeline parallelism (PP, and no, not the same as a `Pipeline`), and context
parallelism (CP) (people love inventing new ways to parallelize so in context of MoEs, we
also have expert parallelism — EP). This page explains what Zephon needs to know about
that division, how to derive this information from your training run's topology, and how
to plan a run so that it can later be resumed on a different number of GPUs.

**One Pipeline Per Rank.** Of all of the different ways that a training job can be
parallelized, the one that we care about the most from a Zephon perspective is the
data-parallelism dimension. Each data-parallel group needs its own unique, non-overlapping
slice of the training curriculum, while all of the ranks within the same data-parallel
group need to see exactly the same data at exactly the same time.

Zephon handles both of these requirements without any communication between ranks during
data loading. Each rank that needs data builds its own `Pipeline` from the same `Dataset`,
`WorkSource`, and operator definitions, and tells that `Pipeline` instance which
data-parallel group it belongs to. Because each part of the `Pipeline` is deterministic,
each instance can work out for itself which samples belong to its group, and peers within
the same group process identical batches by construction instead of by broadcasting them
to each other. The only time that the `Pipeline` instances need to explicitly coordinate
with each other is when they take a checkpoint. This means that every `Pipeline` instance
in a distributed training job needs to know two pieces of information: its *data identity*
so it can determine which samples to produce, and its *coordination identity* so it can
determine how to participate in the checkpointing process.

**Data Identity: Which Samples To Produce.** The data identity of a `Pipeline` is
defined via two options:

1. the `dp_degree`, which is the number of independent data streams in the job, equivalent
   to the training frameworks' data-parallel dimension.
2. The `dp_group_id` identifies which of those streams this particular `Pipeline` instance
   produces, from `0` to `dp_degree - 1`.

Both of these values should come from the data-parallel sub-mesh of your training
framework's topology, not from the global rank. Here is an example for a job on 8 GPUs
that uses tensor parallelism of degree 2, leaving 4 data-parallel groups:

```{literalinclude} ../../../examples/guide/pipelines/data_identity_device_mesh.py
:language: python
:caption: examples/guide/pipelines/data_identity_device_mesh.py
```

Here, GPUs 0 and 1 are tensor parallel peers that share `dp_group_id=0`, so each of them
runs its own `Pipeline` and both produce identical batches, while GPUs 2 and 3 share
`dp_group_id=1`, and so on. Zephon does not need to know anything about the tensor-,
pipeline-, or context-parallel degrees of the job since those do not influence which ranks
should see the same data.

From the perspective of Zephon, different flavors of data parallelism like DDP-style
replication or FSDP-style sharding, are both data-parallel dimensions that need different
data on every rank. The differences between them are really about how the model's
parameters and gradients are managed, and not about what data each rank reads. If your
framework exposes these as separate mesh dimensions (e.g., `dp_replicate` and `dp_shard`
for hybrid sharded data parallelism), flatten them into a single data-parallel dimension
and use its size and local rank as `dp_degree` and `dp_group_id`:

```{literalinclude} ../../../examples/guide/pipelines/data_identity_hsdp_mesh.py
:language: python
:caption: examples/guide/pipelines/data_identity_hsdp_mesh.py
```

Note that if you do not explicitly set `dp_degree` and `dp_group_id`, they will default to
`world_size` and `global_rank`, which is correct for a job that only uses data parallelism.

**Coordination Identity: Which Pipelines Checkpoint Together.** The coordination
identity of a `Pipeline` is defined by the `world_size` and `global_rank` options, and
it describes the set of `Pipeline` instances that take checkpoints together. The
`world_size` is simply the number of `Pipeline` instances that participate in each
checkpoint, and `global_rank` is the unique identifier for a particular `Pipeline`
instance in that set from `0` to `world_size - 1`. The instance that has `global_rank=0`
acts as the leader that combines the progress of the other instances into a single
checkpoint.

Whenever `world_size` is greater than 1, every `Pipeline` instance must also be configured
with 1) a common `aggregate_dir` option that all of the participating instances can read
and write to, either as a shared filesystem directory or as a prefix in object storage and
2) a `run_id` that is unique to this training run so that different jobs that share the
same `aggregate_dir` do not collide with each other.

```{literalinclude} ../../../examples/guide/pipelines/coordination_identity.py
:language: python
:caption: examples/guide/pipelines/coordination_identity.py
```

Every `Pipeline` instance that is part of the same coordination group must call
`Pipeline.checkpoint()` at the same training step. If one of them does not, the leader
will wait for the missing instance until the aggregation timeout (`aggregate_timeout_s`,
which is set to three minutes by default) expires, and then the checkpoint operation will
fail. The details of when to take a checkpoint and how to restore it are covered in
[Checkpointing and Resuming Pipelines](checkpointing_and_resuming.md).

In many jobs, the coordination identity is simply the global rank and the world size of
the training job itself. But it doesn't have to be set this way, and up next we explain
why it might not be.

**Choosing Which Ranks Build A Pipeline.** Because the peer ranks within a single
data-parallel group all produce the same batches, a training job has a choice to make
about which of these ranks need to build a `Pipeline` at all. There are two common
strategies for doing this with different tradeoffs.

The simplest approach is to have every rank build its own `Pipeline`. This means that
every one of these ranks will need to take part in checkpointing, and the coordination
identity is the training job's global `world_size` and `rank`. The tensor- and
pipeline-parallel peers within each data-parallel group do redundant work to produce
identical batches, but they never have to exchange data with each other. In practice, this
redundant work is rarely a bottleneck, since the model training loop (including the
forward and backward passes and any collective communication steps) tends to dominate the
time of each training step. This is the approach that our TorchTitan integration takes, as
discussed in [Training Integrations](../training_integrations.md).

It is also possible to configure things such that only some of the ranks need to build a
`Pipeline`. Some frameworks build each batch on one rank of a model-parallel group and
then broadcast it to the others, so only the ranks that actually load data need to build a
`Pipeline`. In this situation, the coordination identity should describe only the
instances that actually load data, since those are the only ones that need to participate
in checkpoints. Our Megatron integration (see
[Training Integrations](../training_integrations.md)) builds a `Pipeline` on
tensor-parallel rank 0 of the model pipeline stage that consumes the batch, and only the
first of these stages saves the data loader state, so exactly one `Pipeline` per
data-parallel group participates in each checkpoint. It therefore uses the data-parallel
size and rank for both the data identity *and* the coordination identity:

```{literalinclude} ../../../examples/guide/pipelines/coordination_identity_per_dp_group.py
:language: python
:caption: examples/guide/pipelines/coordination_identity_per_dp_group.py
```

No matter which strategy you use, it's important to keep two rules in mind. First, every
`Pipeline` that shares a `dp_group_id` must be built from identical definitions and
options, since they can only produce the same batches if they are running the same
pipeline. Second, the instances counted in the `world_size` must be exactly the instances
that call `checkpoint()` — no more, no less.

(planning-for-elasticity)=

**Planning For Elasticity.** Training runs don't always finish on the hardware that they
started on. A job that starts on 64 GPUs may need to be resumed on 32 after a
preemption, or scaled up to 128 GPUs if more capacity is needed. Zephon supports this by
dividing the training curriculum into a fixed number of logical data streams called
**lanes** which are controlled by the `canonical_replicas` option. The lanes are then
partitioned among the data-parallel groups the current run has:

```{literalinclude} ../../../examples/guide/pipelines/canonical_replicas.py
:language: python
:caption: examples/guide/pipelines/canonical_replicas.py
```

With `canonical_replicas=8`, a run with 8 data-parallel groups assigns one lane to each
group, while a run with 4 data-parallel groups assigns two lanes to each of them. As long
as `canonical_replicas` stays the same, the run can be restored on a different `dp_degree`
and each global training step will contain the same samples.
{ref}`Elastic Resumption <elastic-resumption>` covers the restore
process itself; here we will only focus on the settings that you need to choose at the
start of a run because they cannot be changed later.

1. **Choose `canonical_replicas` for the largest run you expect.** The `dp_degree` can
   never exceed `canonical_replicas`, so set it to the largest number of data-parallel
   groups you expect during the run. If it is left unset, it defaults to the initial
   `dp_degree` value, which allows the run to be resumed on fewer data-parallel groups,
   but not more.
2. **Keep `canonical_replicas` divisible by each `dp_degree` you use.** Otherwise some
   data-parallel groups own more streams than others, which unbalances the load between
   them. Zephon warns when this happens, and both of our reference integrations check for
   this condition and refuse to start if it isn't satisfied.
   [Highly composite numbers](https://en.wikipedia.org/wiki/Highly_composite_number) are
   your friend here.
3. **Consume a multiple of `canonical_replicas` batches per optimizer step.** Each lane
   produces its own fixed sequence of batches, and each data-parallel group takes turns
   delivering batches from the lanes that it owns. If every optimizer step consumes a
   multiple of `canonical_replicas` batches across all of the data-parallel groups, then
   every lane contributes the same number of batches to every step, and each step contains
   the same batches no matter how the lanes are assigned to the groups. For example, with
   `canonical_replicas=8`, a step that consumes 16 batches always contains the next two
   batches from each of the eight lanes. A step that consumes only four batches contains
   batches from just four of the lanes, and which four lanes those are depends on the
   current topology, so a run that is resumed on a different topology would group its
   samples into steps differently than the original run would have. Our reference
   integrations check for this condition at startup and will raise an error if this
   doesn't hold.

**Sharing a Node.** Because every rank that loads data will need to run its own
`Pipeline`, keep in mind that a node that has 8 GPUs will be running 8 independent
`Pipeline` instances, each with its own memory buffers and workers for fetching and
transforming data. These instances may end up competing with each other (and with the
training process itself) for CPU cores, memory, and network bandwidth, so be sure that
when you are sizing the `parallelism` and other settings described in [Inspecting and
Tuning Pipelines](inspecting_and_tuning_pipelines.md) that you are scaling the
per-`Pipeline` values down as the number of `Pipeline` instances on each node goes up,
and double-check `Pipeline.explain()` to verify the total number of threads and
processes that will be running on each node before you launch at scale.
