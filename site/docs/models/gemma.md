---
title: Gemma
description: Three vision-language models, from a 4B dense decoder to a 26B sparse mixture of experts.
family: gemma
---

All three Gemma entries are vision-language: an image goes in, text comes out.
They are also the family that covers the size range of the matrix most
evenly — a 4B dense decoder at the small end, a 12B dense one in the middle,
and a 26B sparse mixture of experts that reads about 4B parameters per token.

Comparing the three is the clearest illustration of why parameter count alone
does not predict speed. The dense models read everything they hold on every
token; the sparse one does not, so it sits at a different point on the
size-versus-rate curve than its total suggests. The
[benchmarks page]({{ '/docs/benchmarks/' | relative_url }}) has the figures.

{% include model-family.html family=page.family %}
