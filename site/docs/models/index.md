---
title: Model Matrix
description: The models included in the HIP EP %VERSION% benchmark snapshot.
---

{%- assign all_llm = site.data.models.llm -%}
{%- assign all_vlm = site.data.models.vlm -%}
{%- assign total = all_llm.size | plus: all_vlm.size -%}

{% comment %}
Keep the distinction between the benchmark snapshot and broader validation:
models in this table have published measurements, while the wider validated set
does not necessarily have performance data on this site.
{% endcomment %}
This page lists the models with published quantized exports and benchmark
results in this release. It is not a complete list of models supported by HIP
EP. <span class="text-accent">More than 50 models are validated across a
release cycle</span>, while this matrix covers only the models included in this
benchmark snapshot.

A model not listed here may still run successfully. Use the same ONNX Runtime or
OGA path, verify GPU execution, and benchmark it on your target hardware.

{% comment %}
This section used to be seven sub-sections, one per family, each with a blurb
and a run-on line of model names -- the thing a hip-ep developer's review
called a list under a heading that says matrix. It is a table now, and the
thing the old layout did that a table cannot is kept below it: the links into
the family pages. (The one-line family blurbs were kept at that point too, and
dropped later, by the page owner, in the rewrite of this page.)

The family pages were NOT deleted, though the same review suggested it. Each
one carries two paragraphs of architectural reading that does not fit in a
cell -- the Qwen page's point, that dense, MoE and Gated DeltaNet all compile
through one pipeline without an operator library apiece, is the compiler's
selling point and there is nowhere else on the site that says it. What changed
is that they stopped being the navigation path: the table's Family column
links to them, so depth is one click away and comparison needs none.

The sort and the filter are progressive enhancement -- see the no-JS note in
_includes/model-matrix.html. Do not make the table depend on them.
{% endcomment %}
## Model matrix

There are {{ total }} models across {{ site.data.model_families.size }} families:
{{ all_llm.size }} text models and {{ all_vlm.size }} vision-language models. The
table can be sorted or filtered by model, family, architecture, or quantization.

{% include model-matrix.html %}

<p class="table-note">
Prefill and decode rates use fixed input and output lengths across the matrix: 2K input tokens for prefill and 128 generated tokens for decode. Prefill is derived from prompt length and time-to-first-token.
</p>

<p class="table-note">
Prefill results are not directly comparable between text and vision-language models because VLM prefill also includes image processing through the vision encoder. Filter by model type before comparing results. Results for all three prompt lengths, together with the underlying TTFT measurements, are available on the
<a href="{{ '/docs/benchmarks/' | relative_url }}">Benchmarks</a> page.
</p>

### By family

{% comment %}
The rows are generated from _data/model_families.yml and _data/models.yml
rather than typed out, so the counts cannot drift from the matrix above them
and a new family cannot be forgotten here. The order is the order of the data
file, which is hand-kept largest-first -- see the header comment there.
{% endcomment %}
| Family | Organization | Models |
|---|---|---|
{% for f in site.data.model_families -%}
{%- assign fam_llm = all_llm | where: "family", f.slug -%}
{%- assign fam_vlm = all_vlm | where: "family", f.slug -%}
{%- assign fam_all = fam_llm | concat: fam_vlm -%}
{%- assign fam_url = '/docs/models/' | append: f.slug | append: '/' -%}
| [{{ f.name }}]({{ fam_url | relative_url }}) | {{ f.vendor }} | {{ fam_all.size }} |
{% endfor %}
See the family pages for model-specific architecture, quantization, and runtime
details.

The same models are also listed on the
[landing page]({{ '/' | relative_url }}) as filterable cards.

## What the snapshot records

| Checked | What it means | What to check locally |
|---|---|---|
| Function | The model compiles and produces output in the release run | Missing operators, unresolved shapes, or crashes on your exact export |
| Performance | Time-to-first-token and tokens-per-second for the published prompt lengths | Driver version, power profile, and thermal conditions can affect absolute numbers |
| Accuracy | Perplexity and task scores do not degrade against the reference implementation | Validate with your own prompts, tasks, or reference outputs |

Performance and accuracy are separate measurements. A throughput result
describes the performance of the published quantized export on the reference
system; it does not establish whether the model is suitable for a particular
application.

## Dense and sparse

The matrix distinguishes between dense models and sparse mixture-of-experts
(MoE) models.

A dense model uses all of its parameters for each token. A sparse MoE model
activates only a subset of its parameters for each token. For models with an
`A3B` or `A4B` suffix, the suffix indicates the approximate number of active
parameters per token.

For sparse models, total parameters are primarily relevant to memory
requirements, while active parameters are more relevant to compute and
throughput. Neither value should be used on its own to compare model
performance.

Qwen3.6 also combines sparse routing with Gated DeltaNet rather than standard
attention. It uses the same HIP EP compilation pipeline.

## What "int4" means here

All models in the matrix use quantized weights. The weights are represented as
4-bit integers grouped along the input dimension, with a scale for each group
and, for asymmetric schemes, a zero point. Activations remain in fp16.

HIP EP consumes the standard ONNX `MatMulNBits` representation used by the
broader ONNX ecosystem; the format is not specific to HIP EP.

For vision-language models, the vision encoder remains in fp16 while the text
decoder is quantized. The encoder runs once per image, while the decoder runs
for each generated token.

The matrix includes several quantization schemes, including RTN, AWQ, k-quant,
symmetric and asymmetric variants, with group sizes from 32 to 128. HIP EP can
consume these different formats.

Quantization is part of the model artifact rather than a runtime option. HIP EP
does not quantize models at runtime. An fp16 model therefore runs in fp16 and
retains the corresponding memory footprint.

## Running a model

These models use the same ONNX Runtime execution path as other ONNX models. See
[Get Started]({{ '/docs/get-started/' | relative_url }}) for installation and
setup.

For generative models,
[ONNX Runtime GenAI](https://github.com/microsoft/onnxruntime-genai) (OGA)
provides the surrounding inference flow, including tokenization, KV-cache
management, and the decode loop. HIP EP executes the ONNX graph used by that
flow.

Calling `session.run()` directly on a decoder graph produces the logits for a
single token; it does not provide the tokenizer, KV-cache, or decode loop needed
for autoregressive generation.

## Where the numbers are

{% comment %}
Placeholder mode supports builds where benchmark figures are not published.
Keep both branches so those builds do not render zeroes as measurements.
{% endcomment %}
{% if site.data.benchmarks.snapshot.placeholder -%}
The release snapshot exists, but those figures are not cleared for publication
yet, so the family pages carry what a model *is* rather than how fast it ran.
The [Benchmarks]({{ '/docs/benchmarks/' | relative_url }}) page explains the
withholding in one place rather than repeating it on each card.
{%- else -%}
Family pages show a representative prompt length for each model. The
[Benchmarks]({{ '/docs/benchmarks/' | relative_url }}) page contains the full
benchmark snapshot, including all three prompt lengths and the test conditions.

The benchmark data represents a single release snapshot and is intended for
comparison within the conditions documented on that page.
{%- endif %}
