---
title: Llama
description: The dense decoder most of the ecosystem's tooling assumes.
family: llama
---

Llama is the baseline shape. It is a plain dense transformer decoder, it is
what most exporters, quantizers and inference harnesses were written against
first, and it is the model to try when you want to know whether a problem is
your model or your setup.

The {{ site.data.benchmarks.snapshot.release }} matrix also includes a
Llama-shaped decoder under a different name:
[DeepSeek-R1-Distill-Llama 70B]({{ '/docs/models/deepseek/' | relative_url }})
is a reasoning distillation onto this architecture at roughly nine times the
size.

{% include model-family.html family=page.family %}
