---
title: Benchmarks
description: Benchmark results for the HIP EP %VERSION% release.
---

{% assign snap = site.data.benchmarks.snapshot -%}
{% assign lens = site.data.benchmarks.prompt_lengths -%}
{% assign catalog = site.data.models.llm | concat: site.data.models.vlm -%}

{% if snap.placeholder %}
<div class="note note--warn" markdown="1">
**Numbers withheld — every value below is a zero placeholder.** The measurements
exist and the run described here happened; the results are simply not cleared for
publication yet. Zero means "not published", not "measured as zero", and nothing
on this page should be cited or compared against anything.

The page is here so the layout, units, rounding and per-model coverage can be
reviewed ahead of that clearance. Publishing is a single edit to
`site/_data/benchmarks.yml`: drop in the numbers and set `placeholder: false`.
</div>
{% endif %}

{%- comment -%}
All three values come from _data/benchmarks.yml rather than being typed here,
so the page cannot claim a release, an OS or a machine that the numbers below
it did not come from. Replacing the snapshot is one edit to that file.

Written as raw HTML rather than as a Markdown table because this one has no
header row: it is three label/value pairs, so the label is a row header
(`<th scope="row">`) and Markdown has no syntax for that. The site's table
styling applies either way -- `.docs-content table` matches on the element.
{%- endcomment -%}
<table>
  <tbody>
    <tr><th scope="row">Release</th><td>{{ snap.release }}</td></tr>
    <tr><th scope="row">OS</th><td>{{ snap.os }}</td></tr>
    <tr><th scope="row">Hardware</th><td>{{ snap.gpu }}</td></tr>
  </tbody>
</table>

These results were collected on a single machine. Results from a different GPU,
driver, power configuration, or thermal state are not directly comparable.

## What this page covers

This page reports a single benchmark snapshot for {{ snap.release }}. It is not
a release history or a comparison with other runtimes.

The results apply to {{ snap.release }} on the hardware listed above. If you are
running a different release or configuration, treat these numbers as reference
data rather than measurements of your setup.

## Metrics

| Metric | Unit | Direction | Description |
|---|---|---|---|
| TTFT | seconds | Lower is better | Time from submitting the prompt to the first generated token. |
| TPS | tokens/second | Higher is better | Steady-state generation rate after the first token. |

TTFT is primarily affected by prefill and therefore increases with prompt
length. TPS measures decode throughput after the first token.

{% comment %}
Version-bound claim, re-checked at each release bump. Carried from v0.4.0 to
v0.5.1 on a static check of the shipped EP: it exposes the same caching-related
provider options it did before and gained no on-disk artifact store, so the
default has not moved. The same claim appears three times on
docs/get-started/binary-package.md, which carries the fuller note and the
re-test to run. If it falls there, it falls here too.
{% endcomment %}
Compile time is excluded from both metrics. HIP EP compiles the graph on first
use; subsequent calls use the compiled graph. In {{ site.hip_ep_version }},
compiled artifacts are kept for the lifetime of the session and are not cached
on disk, so a new process recompiles the graph.

See the [Overview]({{ '/docs/' | relative_url }}) for details on the compilation
flow.

## Results

The benchmarks use prompt lengths of {{ lens | join: ", " }} tokens.

{% include bench-bars.html %}

{% unless snap.placeholder %}
TPS at a {{ lens[1] }}-token prompt on {{ snap.gpu }}, using HIP EP
{{ snap.release }}. Higher is better. The chart shows one prompt length; the
table below contains results for all three.
{% endunless %}

<div class="table-scroll" markdown="1">

| Model | Parameters | {% for l in lens %}TTFT {{ l }} (s) | {% endfor %}{% for l in lens %}TPS {{ l }} | {% endfor %}
|---|---|{% for l in lens %}---:|{% endfor %}{% for l in lens %}---:|{% endfor %}
{% for row in site.data.benchmarks.lm -%}
{% assign m = catalog | where: "id", row.id | first -%}
| {{ m.name | default: row.id }} | {{ m.params }} | {% for v in row.ttft %}{{ v }} | {% endfor %}{% for v in row.tps %}{{ v }} | {% endfor %}
{% endfor %}

</div>

TTFT generally increases with prompt length because prefill processes the prompt
tokens. TPS generally decreases as the prompt grows because each generated token
attends over a larger KV cache.

For sparse MoE models, the parameter count is shown as total / active
parameters. Decode performance is more closely related to the active parameter
count than the total parameter count.

For vision-language models, TTFT includes the vision encoder. This contributes
to the higher TTFT often seen at short prompt lengths.

{% comment %}
This section opened with "Sixteen generative models, and nothing else", i.e.
with what IS on the page, under a heading promising what is not. Reordered on a
hip-ep developer's review. Same content, the other way round. The heading it
was reordered under, "What is not on this page", was later renamed to "Scope"
by the page owner; the ordering is the part that came from the review.
{% endcomment %}
## Scope

This page covers generative model benchmarks using TTFT and TPS. Other workload
classes are validated separately and are not included here.

{% comment %}
Re-check this statement at each release bump rather than carrying it forward;
the paragraph should be removed once the suite is ready for published results.
{% endcomment %}
The Procyon AI Inference Benchmark is also part of the test suite. For
{{ snap.release }}, its workloads are functionality-verified, with performance
optimization still in progress. Results are therefore not published here.

## Measurement conditions

The benchmark harness uses the following conditions:

- **Serial execution.** Only one benchmark runs on the GPU at a time.
- **Debug and tracing disabled.** `HIPDNN_EP_PERF=1` and `HIPDNN_EP_DEBUG=1` are
  not set.
- **Autotune caches primed.** Cold autotune runs are excluded.
- **First inference excluded.** The first inference includes one-time graph
  compilation.

Thermal state is not controlled. Sustained workloads can reduce clock or memory
performance as the system heats up, particularly on thin systems.

Different drivers, power profiles, memory configurations, and thermal conditions
can also affect absolute results. Reproduced results should therefore be
compared only under similar conditions.

## Reproducing the results

The release package includes `model_benchmark` for Windows. It runs models
through OGA and reports TTFT and TPS directly. See the
[binary package]({{ '/docs/get-started/binary-package/' | relative_url }}) page
for usage and benchmark options.

<div class="note note--warn" markdown="1">
Before comparing results, verify that the model is actually running on the GPU.
ONNX Runtime can fall back to CPU execution. Set:

```powershell
$env:HIPDNN_EP_STRICT=1
```

to make unsupported GPU execution fail instead of silently falling back to CPU.
</div>
