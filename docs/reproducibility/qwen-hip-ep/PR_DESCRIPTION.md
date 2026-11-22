<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Add frozen native QSA specializations and reproducible experiment evidence

Replace four selected full-attention nodes with native stateful QSA operations
for the pinned OLMo and Qwen models. Selector projection, features, stable ranking
and selected-address attention run on the EP stream inside the existing full-model
session. Dense GQA remains available and provides Qwen's full-support path. Add
BF16 selector import, compiler/runtime validation and matched evaluation tools.

The Qwen W4A16 comparison measures batch1 at4K/16K/32K: decode throughput changes
by+0.50%/+1.23%/+1.60%, while prefill decreases4.13%/4.47%/8.83%. PPL changes
5.96868807 to5.97127044; GSM8K is111/128 for both, with identical short-context
generations. Full KV storage is retained and QSA adds workspace. These measurements
belong to the preserved pre-merge binaries, not the new merge build.

Include a portable JSON/CSV paper packet, compressed raw quality rows, raw paired
timings, checksum manifest and local raw archive descriptor. Explicitly record
memory coverage, the failed/canceled64K attempt, limited accuracy coverage and
precision/generation/statistics differences from the MI355X source campaign.

Validation: see [AUDIT.md](AUDIT.md),`candidate-validation.json` and the hashed
historical receipts. The implementation is qualified for gfx1151/wave32; shape
guards do not establish full-model performance or memory support at all bounds.
The local PR branch resolves conflicts against local`origin/main` while preserving
upstream GQA host-length/bias handling, FP8 import and ROCTx tracing. No branch,
pull request or results have been pushed or published by this work.
