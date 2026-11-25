---
layout: page
permalink: /
title: null
description: >-
  HIP EP is an ONNX Runtime Execution Provider for AMD GPUs. It compiles ONNX
  graphs through an MLIR pipeline and runs them with hipBLASLt and custom HIP
  kernels.
---

{%- comment -%}
The hero is one left-aligned column. It used to be a two-column split with a
stat carousel on the right, which is the layout, the control cluster and very
nearly the headline sentence of a competing project's landing page; being
mistaken for someone else's site is worse than any amount of empty space on
the right. Everything the carousel carried is still here -- it is the spec
strip at the bottom of the section, where four claims are readable at once
instead of one every seven seconds.

Nothing replaced the right-hand column and nothing should. The section is
four things -- headline, lede, buttons, strip -- on the background field that
_includes/backdrop.html draws, and the field running full-bleed behind them
is what makes the empty right read as a surface rather than as a track that
failed to fill. A drawing lived here for a while and was removed; it is in
the history if it is wanted back, but it is not a hole waiting for something.

There is no eyebrow above the headline either. "ONNX Runtime Execution
Provider" is what the lede's first clause says in a sentence, and a chip of
small caps was the first thing between the header and the words that matter.
{%- endcomment -%}
<section class="hero">
  <div class="section__inner">
    {%- comment -%}
    This used to read "The fastest, most efficient ...", which was the competing
    site's headline with the hardware word swapped. The superlatives are gone on
    instruction, and dropping them fixes both of that sentence's problems at
    once: the phrasing is no longer borrowed, and the page no longer makes the
    one claim above the fold that nothing behind it supports --
    /docs/benchmarks/ publishes throughput for this backend, not a ranking
    against other backends.

    "fast and efficient" is a description and can stand on measurements that
    exist. A superlative is a comparison, and restoring one here puts the page
    back in the position of asserting a result the site does not publish. If a
    ranking is ever wanted, it needs numbers against named alternatives on
    /docs/benchmarks/ first, and the headline follows from those -- not the
    other way round.

    "A", not "The", and the article is doing the same work as the rest of the
    edit. "The fast and efficient backend" is a definite description of an
    indefinite quality: it reads as a claim to be the only one, which is the
    superlative coming back in through the article. "A" says this is one such
    backend and lets the benchmarks say how good.
    {%- endcomment -%}
    <h1 class="headline-lg hero__headline">A fast and efficient LLM inference backend on AMD iGPU</h1>
    {%- comment -%}
    The lede used to name the three MLIR stages in an em-dash aside -- ONNX
    dialect, to a HIP dialect, to LLVM IR -- and that aside, and only that
    aside, is gone: a hip-ep developer's review called it too technical for
    a first screen and said it makes the design look complicated rather than
    capable. The stages are still on /docs/, where a reader who wants them is
    already looking. Nothing on this screen names them now, and that is the
    intended state: the first sentence a visitor reads should say what the
    thing does, not how it is staged internally.

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

    {%- comment -%}
    The four claims the stat carousel used to step through, with its own
    labels and its own wording -- all four readable at once, with the accent
    moving between them on its own. A carousel hides three of four behind a
    control a visitor has no reason to press; this hides nothing and has no
    control, so there is no state a reader can lose by ignoring it.

    `data-specs` and `data-spec` are initSpecStrip()'s hooks. Without them, or
    without script.js, the strip is a static four-column list with every
    figure at full accent -- see the note on .specs in styles.css, which is
    where that fallback is actually enforced.

    The hardware entry lost the codename list that fitted inside a 400px
    slide and does not fit inside a quarter-width cell. The full mapping --
    series, marketing name and gfx target -- is the table on /docs/ and on
    /docs/get-started/, which is where someone checking whether their own
    machine qualifies is going anyway.
    {%- endcomment -%}
    <dl class="specs" data-specs>
      <div class="specs__item" data-spec>
        <dt class="specs__label">Models</dt>
        <dd class="specs__value">20+</dd>
        <dd class="specs__note">model architectures supported</dd>
      </div>
      <div class="specs__item" data-spec>
        <dt class="specs__label">Context</dt>
        <dd class="specs__value">128K</dd>
        <dd class="specs__note">maximum supported context, in tokens</dd>
      </div>
      <div class="specs__item" data-spec>
        <dt class="specs__label">Hardware</dt>
        <dd class="specs__value">3</dd>
        <dd class="specs__note">Ryzen AI processor series: Max, 400, 300</dd>
      </div>
      <div class="specs__item" data-spec>
        <dt class="specs__label">License</dt>
        {%- comment -%}
        "Open source", not "MIT". It was briefly the licence name and was put
        back on instruction: the other three figures answer "how much do I
        get", and at a glance "MIT" reads as a fourth specification rather than
        as the reassurance it is meant to be. A visitor who wants the licence
        identifier finds it in the repository. This is the one value in the
        strip that is a word rather than a figure -- see .specs__value--word.
        {%- endcomment -%}
        <dd class="specs__value specs__value--word">Open source</dd>
        <dd class="specs__note">compiler, runtime and kernels in one repository</dd>
      </div>
    </dl>
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
heading. Same arrangement as the model matrix on /docs/models/: ship the
readable fallback, then upgrade it. The hero above no longer needs the
arrangement at all -- it is markup and a stylesheet now.

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
