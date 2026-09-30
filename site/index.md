---
layout: page
permalink: /
title: null
description: >-
  HIP EP is an ONNX Runtime Execution Provider for AMD GPUs. It compiles ONNX
  graphs through an MLIR pipeline and runs them with hipBLASLt and custom HIP
  kernels.
---

<section class="hero">
  <div class="section__inner hero__grid">
    <div class="hero__main">
      <p class="kicker">ONNX Runtime Execution Provider</p>
      <h1 class="headline-lg">The fastest, most efficient LLM inference backend on AMD iGPU</h1>
      {%- comment -%}
      The lede used to name the three MLIR stages in an em-dash aside -- ONNX
      dialect, to a HIP dialect, to LLVM IR -- and that aside, and only that
      aside, is gone: a hip-ep developer's review called it too technical for
      a first screen and said it makes the design look complicated rather than
      capable. The stages are still on /docs/, where a reader who wants them is
      already looking.

      The rest of the sentence is the original and should stay that way. "an
      MLIR pipeline" is the claim the aside was decorating, not part of the
      decoration. "custom HIP kernels" is the project's own term -- it is what
      _config.yml's description says, what this page's own front matter says,
      and what the shipped artifact is called (custom_kernels_gfx*, from
      docs/design/custom_kernel_design.md). Do not paraphrase it.
      {%- endcomment -%}
      <p class="lede">
        HIP EP compiles your ONNX graph through an MLIR pipeline and executes
        it on AMD GPUs with hipBLASLt and custom HIP kernels.
      </p>
      <div class="btn-row">
        <a class="btn btn--primary" href="{{ '/docs/get-started/' | relative_url }}">Get started</a>
        <a class="btn btn--ghost" href="{{ '/docs/models/' | relative_url }}">Models</a>
        <a class="btn btn--ghost" href="{{ '/docs/benchmarks/' | relative_url }}">Benchmarks</a>
        <a class="btn btn--ghost" href="{{ site.repo_url }}/releases/tag/{{ site.hip_ep_version }}">Download {{ site.hip_ep_version }}</a>
      </div>
    </div>

    {%- comment -%}
    The feature deck. Without JavaScript every panel is simply stacked and the
    controls stay hidden, which is why the slides are ordinary sections rather
    than an off-screen track: the fallback has to be readable, not merely
    present. initDeck() in script.js takes over from there.
    {%- endcomment -%}
    <aside class="deck" data-deck aria-label="What HIP EP does">
      <div class="deck__viewport" data-deck-viewport>
        {%- comment -%}
        Five claims, each led by the one figure that carries it. This panel used
        to step through the README Highlights, which describe how the thing is
        built; a visitor deciding whether to spend an afternoon on it is asking
        what it covers and what it costs them. The Highlights are in the README
        and in the design docs, where a reader who has decided to care will go
        looking for them.
        {%- endcomment -%}
        <article class="deck__slide" data-deck-slide>
          <p class="deck__kicker">Models</p>
          <p class="deck__stat">20+</p>
          <p class="deck__title">Model architectures supported</p>
        </article>
        <article class="deck__slide" data-deck-slide>
          <p class="deck__kicker">Context</p>
          <p class="deck__stat">128K</p>
          <p class="deck__title">Maximum supported context, in tokens</p>
        </article>
        <article class="deck__slide" data-deck-slide>
          <p class="deck__kicker">Hardware</p>
          <p class="deck__stat">3</p>
          <p class="deck__title">Ryzen AI processor series</p>
          <dl class="deck__spec">
            <dt>Ryzen AI Max series</dt>
            <dd><span class="deck__spec-hint">incl.</span> Strix Halo</dd>
            <dt>Ryzen AI 400 series</dt>
            <dd><span class="deck__spec-hint">incl.</span> Gorgon Point</dd>
            <dt>Ryzen AI 300 series</dt>
            <dd><span class="deck__spec-hint">incl.</span> Strix Point, Krackan Point</dd>
          </dl>
        </article>
        <article class="deck__slide" data-deck-slide>
          <p class="deck__kicker">License</p>
          <p class="deck__stat deck__stat--word">Open source</p>
          <p class="deck__title">Compiler, runtime and kernels in one repository</p>
        </article>
      </div>

      <div class="deck__controls" data-deck-controls hidden>
        <button class="deck__nav" type="button" data-deck-prev aria-label="Previous feature">&#8249;</button>
        <div class="deck__dots" data-deck-dots></div>
        <button class="deck__nav" type="button" data-deck-next aria-label="Next feature">&#8250;</button>
      </div>
    </aside>
  </div>
</section>

{%- comment -%}
The wall is grouped by family and filtered by the pills above it, the same
grouping /docs/models/ uses -- so a reader who follows a card through to the
docs finds the models in the order they just saw them.

Every family's cards are in the HTML. Without JavaScript the pills are jump
links and the seven groups are simply stacked under their own headings, which
is long but complete; initModelTabs() collapses that to one family at a time
and takes the headings off, because at that point the selected pill is the
heading. Same arrangement as the deck above: ship the readable fallback, then
upgrade it.

The prose that used to sit under the heading is gone on instruction. What it
said -- that this is a validation suite rather than a compatibility list -- is
the first paragraph of /docs/models/, which every card now links into.
{%- endcomment -%}
<section class="section">
  <div class="section__inner">
    <p class="kicker">Model showcase</p>
    {%- comment -%}
    The release tag is read from the benchmark data, not from
    site.hip_ep_version, and the difference matters: what is being versioned
    here is the numbers on the cards below, so the tag has to move when they do
    and stay put when they do not. The two happen to agree today.
    {%- endcomment -%}
    <h2 class="headline-md">Models that already run on your GPU
      <span class="headline-badge"><span class="visually-hidden">Measured on release </span>{{ site.data.benchmarks.snapshot.release }}</span>
    </h2>

    <div class="model-tabs" data-model-tabs>
      <div class="pill-list" data-model-tabs-list>
        {%- for f in site.data.model_families -%}
        {%- assign fam_llm = site.data.models.llm | where: "family", f.slug -%}
        {%- assign fam_vlm = site.data.models.vlm | where: "family", f.slug -%}
        {%- assign fam_n = fam_llm.size | plus: fam_vlm.size -%}
        <a class="pill-list__item" href="#models-{{ f.slug }}" data-model-tab="{{ f.slug }}">{{ f.name }}
          <span class="pill-list__meta">{{ fam_n }}</span>
        </a>
        {%- endfor -%}
      </div>

      {%- for f in site.data.model_families -%}
      {%- assign fam_llm = site.data.models.llm | where: "family", f.slug -%}
      {%- assign fam_vlm = site.data.models.vlm | where: "family", f.slug -%}
      <div class="model-tabs__panel" id="models-{{ f.slug }}" data-model-panel="{{ f.slug }}">
        <h3 class="model-tabs__heading">{{ f.name }}</h3>
        <div class="model-grid">
          {%- for m in fam_llm %}
          {% include model-card.html m=m kind="LLM" %}
          {%- endfor %}
          {%- for m in fam_vlm %}
          {% include model-card.html m=m kind="VLM" %}
          {%- endfor %}
        </div>
      </div>
      {%- endfor -%}
    </div>
  </div>
</section>

{%- comment -%}
A "Supported hardware" table (the three gfx targets) and an "Install" section
(the one-command deploy script, plus the manual two-command route) used to sit
between the model wall and the cards below. Both were removed from the landing
page by decision, not by accident: the same content is on /docs/ and under
/docs/get-started/, which the Start here cards point at. Do not reinstate
either here without asking -- the omission is the intent.
{%- endcomment -%}

<section class="section">
  <div class="section__inner">
    <h2 class="headline-md">Start here</h2>
    {%- comment -%}
    Four cards, one per documentation section, and that is the whole rule. A
    fifth pointed at /docs/get-started/source-build/ and was removed by
    decision: it sat beside Get Started pointing into Get Started, so the
    landing page offered the reader two doors to the same section and made the
    rarest install path look like a peer of the section that contains it. That
    page has since been removed from the site altogether -- source builds are
    documented in the repository now -- so the card has nothing to point at
    either way. Do not reinstate a card here for a page that lives under one of
    the four.
    {%- endcomment -%}
    <div class="card-grid">
      <a class="card card--link" href="{{ '/docs/get-started/' | relative_url }}">
        <p class="card__title">Get Started</p>
        <p class="card__body">From an empty machine to a model on the GPU, one
          page per install path, each ending in a check that catches the silent
          CPU fallback.</p>
      </a>
      <a class="card card--link" href="{{ '/docs/' | relative_url }}">
        <p class="card__title">Overview</p>
        <p class="card__body">What HIP EP is, how a graph reaches the GPU, which
          hardware is covered, and what it is pinned to.</p>
      </a>
      <a class="card card--link" href="{{ '/docs/models/' | relative_url }}">
        <p class="card__title">Models</p>
        <p class="card__body">The language and vision-language models validated
          on every release, and what each one is built out of.</p>
      </a>
      <a class="card card--link" href="{{ '/docs/benchmarks/' | relative_url }}">
        <p class="card__title">Benchmarks</p>
        <p class="card__body">One snapshot per release — throughput and latency,
          with the machine, the commands and the warm-up rules behind them.</p>
      </a>
    </div>
    <p>
      Something here wrong, missing, or contradicted by your own machine? The
      compiler, the runtime, the kernels and this site are all in
      <a href="{{ site.repo_url }}">one open-source repository</a> — open an
      <a href="{{ site.repo_url }}/issues">issue</a>, including the case where
      the documentation is what is broken.
    </p>
  </div>
</section>
