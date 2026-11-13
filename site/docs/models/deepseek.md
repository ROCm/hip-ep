---
title: DeepSeek
description: A reasoning distillation onto a 70B Llama-shaped decoder — the largest dense graph in the matrix.
family: deepseek
---

DeepSeek-R1-Distill-Llama 70B is a reasoning model distilled onto the
[Llama]({{ '/docs/models/llama/' | relative_url }}) architecture, so it brings
nothing architecturally new to the matrix. What it brings is size: it is the
largest *dense* graph in the {{ site.data.benchmarks.snapshot.release }} matrix,
and a dense model reads every parameter it holds on every token.

That makes it the memory-planning case. Where a sparse model of similar total
size gets away with touching a fraction of itself per token, this one does not,
and it is included for exactly that reason.

{% include model-family.html family=page.family %}
