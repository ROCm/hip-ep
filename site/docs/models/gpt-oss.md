---
title: GPT-OSS
description: Two sparse mixture-of-experts text models, including the largest model in the matrix.
family: gpt-oss
---

Both GPT-OSS entries are sparse mixtures of experts, which is why the matrix
has a 120B model in it at all. A sparse model holds far more parameters than it
reads: only a few billion of them take part in any one token, so the rate it
generates at tracks that active figure rather than its total size. The total
still has to fit in memory — which is what makes these a workload for unified
memory rather than a workload for a small discrete card.

Sparse routing is also the case that breaks naive per-operator dispatch, since
which experts run is decided at runtime. Both models compile through the
standard pipeline regardless; see the
[overview]({{ '/docs/models/' | relative_url }}) for what the release checks.

{% include model-family.html family=page.family %}
