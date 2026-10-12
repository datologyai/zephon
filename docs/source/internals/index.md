# Under the Hood

The rest of this guide describes Zephon from the outside: what you configure, what you
call, and what comes back out. This section describes it from the inside. You need it when
you are writing an operator that buffers or reorders records, or when you are debugging a
pipeline that will not evict a chunk or will not restore a checkpoint. You may also want
it simply to check that the guarantees the rest of the guide makes are actually
guarantees, and not just hopeful adjectives.

Nothing here is required reading in order to use Zephon. If you only want to know which
knobs exist and what they do, [How Pipelines Run](../pipelines/how_pipelines_run.md) and
[Inspecting and Tuning Pipelines](../pipelines/inspecting_and_tuning_pipelines.md) are
where you want to be.

**A Warning About Stability.** These pages describe machinery that lives in
`zephon/_internal/` , which is explicitly not part of the public API. The layout and the
signatures in there change without notice, and we will not keep them stable for you. What
*is* stable is the contract that the machinery implements, and that is what we have tried
to write down here. Where a page names a class or a function inside `_internal` , treat it
as a pointer to the code rather than as something to import. The exception is [WorkSources
and Work Chunks](worksources_and_work_chunks.md), which documents a public interface,
because `zephon.work` is something you are meant to implement against.

The five pages build on each other in order. [WorkSources and Work
Chunks](worksources_and_work_chunks.md) explains where the data order comes from in the
first place, and what a curriculum of your own would have to implement. [Elastic
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
