---
title: Mistral
description: A 7B dense instruction model and a 119B multimodal MoE.
family: mistral
---

Mistral covers both ends of the matrix. Mistral 7B Instruct is the small end
of the text side: a plain dense decoder that leaves the most headroom for
everything else on the machine. Mistral Small 4 is a 119B multimodal MoE with
about 6.5B parameters active per token, added in v0.5.3, and it sits with the
other large vision-language entries on the same snapshot.

{% include model-family.html family=page.family %}
