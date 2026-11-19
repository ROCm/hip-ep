---
title: Nemotron
description: Two hybrid MoE text models from NVIDIA's Nemotron 3 line.
family: nemotron
---

Nemotron in this matrix is a pair of hybrid text models: Lightning at 30B
total / 3B active, and Super at 120B total / 12B active. Both interleave
Mamba-2, MoE, and attention, and both are validated on the same Strix Halo
snapshot as the dense and vision-language entries.

{% include model-family.html family=page.family %}
