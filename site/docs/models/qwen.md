---
title: Qwen
description: Two Qwen2.5 text models and seven Qwen3.x vision-language models in the release matrix.
family: qwen
---

Qwen is the widest family in the matrix and the one that covers the most
architectural ground. Qwen2.5 is a pair of conventional dense text decoders,
one of them specialised for code. Qwen3.x is vision-language, and between its
seven entries it spans dense decoders, sparse mixtures of experts, Gated
DeltaNet, and — in Qwen3.8 Flash-Next — Qwen Sparse Attention with an n-gram
embedding table.

That range is why this page is worth reading rather than skimming: none of
those shapes needed an operator library of its own. They compile through
the same pipeline as everything else on the
[overview]({{ '/docs/models/' | relative_url }}), and the benchmark snapshot
keeps their differences in separate rows rather than turning them into separate
install paths.

{% include model-family.html family=page.family %}
