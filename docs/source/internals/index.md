# Under the Hood

The rest of this guide describes Zephon from the outside: what you configure, what you
call, and what comes back out. This section describes it from the inside. You will want it
when you are writing an operator that buffers or reorders records, or when you are
debugging a `Pipeline` that will not evict a chunk or will not restore a checkpoint. You
may also just want to check for yourself that the guarantees the rest of the guide makes
really are guarantees.

None of this is required reading in order to use Zephon. If you only want to know which
knobs exist and what they do, [How Pipelines Run](../pipelines/how_pipelines_run.md) and
[Inspecting and Tuning Pipelines](../pipelines/inspecting_and_tuning_pipelines.md) are
where you want to be.

**A warning about stability.** These pages describe behaviour rather than implementation.
Most of the machinery involved lives in private modules whose layout and signatures we
change without notice, so we have tried to write down the contract it implements and to
name things the way you would reach them: the operator you called, the option you set, the
class you would subclass. The contract is the part that is stable. If we rework a
mechanism described here, the guarantees should survive it even though the code will not
look the same. The one exception is the `WorkSource` interface itself, which is still
settling and may change between releases.

The five pages build on each other, so they are worth reading in order. [WorkSources and
Work Chunks](worksources_and_work_chunks.md) explains where the data order comes from in
the first place, and what a curriculum of your own would have to implement. [Elastic
Determinism](elastic_determinism.md) explains the logical partitioning that makes that
order independent of the number of GPUs. [Operators and Runners](operators_and_runners.md)
explains how an operator can run across sixteen workers and still make the same decisions
it would have made on one. [Records and Sample
Lifecycles](records_and_sample_lifecycles.md) explains how an individual record is
identified, how it splits and merges, and how the engine knows when it is finished with
it. [Checkpoint Replay and Bounded Recovery](checkpoint_replay.md) puts the four together
and explains how a checkpoint stays small and a restore stays fast.

```{toctree}
:maxdepth: 2

worksources_and_work_chunks
elastic_determinism
operators_and_runners
records_and_sample_lifecycles
checkpoint_replay
```
