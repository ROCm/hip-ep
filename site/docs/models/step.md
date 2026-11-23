---
title: Step
description: Step 3.5 Flash — a 196B sparse MoE with about 11B parameters active per token.
family: step
---

Step 3.5 Flash is the largest sparse MoE in the matrix by total parameter
count. It keeps roughly 11B parameters active per token, so decode cost tracks
the active set rather than the 196B total. It was added in v0.5.3 and uses the
same OGA + HIP EP path as the other text models.

{% include model-family.html family=page.family %}
