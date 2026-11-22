---
title: Laguna
description: Two sparse MoE coding models from Poolside — Laguna XS 2.1 and Laguna S 2.1.
family: laguna
---

Laguna is the coding-focused MoE family in the matrix. XS is a 33B total /
3B active model sized for a single local machine; S is a 118B total / 8B
active model in the same recipe. Both use sliding-window and global attention
together with a large expert pool, and both compile through the same HIP EP
pipeline as the rest of the matrix.

{% include model-family.html family=page.family %}
