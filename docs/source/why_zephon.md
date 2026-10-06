# Why Zephon?

**Your data loader is a hidden variable.** If you resume a training run on 8 GPUs instead
of the 16 GPUs it started on, you would expect to be training on the same data in the same
order. That is not the case. The data loader quietly changes what each GPU sees.
[Mosaic Streaming](https://docs.mosaicml.com/projects/streaming/en/latest/distributed_training/elastic_determinism.html)
calls the property you might have been expecting *elastic determinism*: yielding the same
global batches no matter how many GPUs you train on. Most data loaders do not even offer
it.

Isolating variables is the basis of experimental research, as
[early systems work](https://multicians.org/InstrumentationPaper.html) stressed. If two
runs differ in one parameter and everything else behaves deterministically, the difference
in outcome belongs to that parameter. This is not the case when the loader silently
changes the data order, making it a hidden variable in your experiments.

<figure class="zephon-post zephon-embed zephon-framed">
  <iframe src="_static/embeds/fault-resume.html" height="560" loading="lazy"
          title="Compare Zephon and standard loader loss curves after a fault"></iframe>
  <figcaption>
    <strong>Resuming a run on a different GPU count.</strong> If we continue training
    after a fault or cluster resize event, without an elastically deterministic data
    loader, the loss curve depends on the number of GPUs we train with. Drag the divider to
    compare the two loaders; each is shown against its own uninterrupted reference.
  </figcaption>
</figure>

*Any* variation in the input data sequence may confound the results of ablations. The
comparison above shows how the loss curve of a training run behaves when continuing
training on fewer nodes after an interruption. The change in GPU topology
causes a standard data loader to change the data order. However, with Zephon, every
topology receives the same global batches, resulting in overlapping loss curves. This is
even measurable in downstream evaluations.

<section class="zephon-post figure-carousel" aria-label="Downstream evaluation across GPU counts"
         aria-roledescription="carousel" tabindex="-1">
  <div class="figure-carousel-header">
    <span class="figure-carousel-title">Downstream evaluation across GPU counts</span>
    <span class="figure-carousel-count" aria-hidden="true">1 / 2</span>
  </div>
  <div class="figure-carousel-viewport">
    <div class="figure-carousel-item" data-active="true" role="group" aria-roledescription="slide"
         aria-label="1 of 2: DCLM Core v1">
      <iframe src="_static/embeds/eval-dclm.html" height="440" loading="lazy"
              title="DCLM Core v1 score against training topology"></iframe>
      <p class="figure-caption">
        <strong>DCLM Core v1.</strong> Scores after training on 8–64 GPUs, with
        non-elastically deterministic (standard) data loading and with Zephon. We train 1B
        models for 20B tokens; the model seed checkpoints and RNG seeding are identical, so
        the runs differ only in data order due to topology differences. The band marks the
        noise floor from running the identical evaluation suite 4 times. Hover a point for
        its score.
      </p>
    </div>
    <div class="figure-carousel-item" data-active="false" role="group" aria-roledescription="slide"
         aria-label="2 of 2: FineWeb" aria-hidden="true" inert>
      <iframe src="_static/embeds/eval-fineweb.html" height="440" loading="lazy"
              title="FineWeb score against training topology"></iframe>
      <p class="figure-caption">
        <strong>FineWeb.</strong> The same comparison on the other suite, with an identical
        setup. Models trained with Zephon have flat, consistent evaluation scores, while a
        non-deterministic loader lands on a different score for each GPU count — the spread
        here is wider than the evaluation's own noise floor. Hover a point for its score.
      </p>
    </div>
  </div>
  <div class="figure-carousel-navigation" role="group"
       aria-label="Downstream evaluation across GPU counts navigation">
    <button type="button" class="figure-carousel-arrow" data-carousel-step="previous"
            aria-label="Previous item"><svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m15 18-6-6 6-6"/></svg></button>
    <div class="figure-carousel-choices">
      <button type="button" aria-pressed="true"><span class="figure-carousel-dot"
              aria-hidden="true"></span>DCLM Core v1</button>
      <button type="button" aria-pressed="false"><span class="figure-carousel-dot"
              aria-hidden="true"></span>FineWeb</button>
    </div>
    <button type="button" class="figure-carousel-arrow" data-carousel-step="next"
            aria-label="Next item"><svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m9 18 6-6-6-6"/></svg></button>
  </div>
  <span class="figure-carousel-live sr-only" aria-live="polite" aria-atomic="true"></span>
</section>

In the downstream evaluation above, the swings in loss correspond to a variation of eval
scores across GPU counts that is above relevant thresholds of public evaluation suites like DCLM Core v1 or
FineWeb, as well as our internal evaluation suites here at Datology. We observe a maximum
swing of ~1pp for Core v1 and ~0.56pp for FineWeb.

**Why this is hard.** Preserving the data order would be straightforward if the pipeline
hands samples through one at a time. Modern pipelines, however, often don't do this.
Splitting documents into subsequences, packing subsequences into bins, and reordering
samples are common things for a stateful n-to-m pipeline. Such a pipeline has no sample
index. Progress is not a number you can just write down in this case; it is spread across
in-flight operator state, partially filled bins, and records that have been reordered. So
how does Zephon manage writing checkpoints, and what does it do on resumption?

Zephon handles this by checkpointing how far the run has progressed through the data
rather than the internal state of the pipeline, and by periodically bringing its operators
back to a clean point that a resumed run can restart from. This is what lets Zephon bound
the state of an otherwise potentially infinitely unbounded pipeline.
[How Pipelines Run](pipelines/how_pipelines_run.md) describes this mechanism in detail,
and [Distributed Training](pipelines/distributed_training.md) covers what you need to
configure for a multi-rank run.

**Why we released it.** When we went looking for something that already implemented
elastic determinism, came with useful operations, and was decoupled from underlying file
formats, we could not find anything. So we built Zephon for ourselves and for the teams we
work with. Once we had it running, it was clear it could help anyone training a model, so
we decided to release it to everyone.

We are doing this because of where we think the AI landscape is headed. We want to help
create a world where any team can build its own foundation model. Building models requires
reliable experiments, and non-deterministic data loading makes experiments unreliable. If
everyone has to rebuild their data loading infrastructure from scratch before they can
even run a reliable experiment, this is not very efficient.
