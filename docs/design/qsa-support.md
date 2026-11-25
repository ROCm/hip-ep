<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# QSA scope for the combined HIP-EP PR

All QSA compiler, runtime, kernel, export, validation and experiment work is
consolidated on [`users/rajeevp/qsa`](https://github.com/ROCm/hip-ep/tree/users/rajeevp/qsa).
The repository requires an approved prefix, so the requested `rajeevp/qsa` name
uses the compliant `users/rajeevp/qsa` namespace. The older local branches
preserve historical work and are not separate PRs.

All three configurations use [published Hugging Face selectors](qsa-selector-assets.md)
with pinned revisions and file/per-tensor verification. Native exports do not
require private selector checkpoint files.

| Model | Frozen selector configuration | Native operation | Export entry point |
|---|---|---|---|
| `allenai/Olmo-3-1025-7B` | Four: zero-based 7, 11, 15, 23 | `com.amd::QsaGroupQueryAttention` | `scripts/qsa/export_native_gqa.py` |
| `Qwen/Qwen3.8-27B` | Four: zero-based 27, 31, 35, 39 | `com.amd::QwenQsaGroupQueryAttention` | `scripts/qwen_qsa/export_native_gqa.py` |
| `Qwen/Qwen3.8-27B` | Eight: zero-based 3, 7, 23, 27, 31, 35, 39, 63 | Same Qwen operation at all eight selected nodes | `scripts/qwen_qsa/export_eight.py` |

OLMo eight-selector export and weights are not implemented. The Qwen Eight
export uses all eight updated checkpoint-640 weights, including the overlapping
four positions; it does not combine the two checkpoints. Selector count is a
graph-export choice, not a compiler flag or a runtime switch. Both prefill and
decode use the selected model's QSA contract inside the original full-model
session, with one EP dispatch per forward. There is no training or calibration.

These are pinned model specializations qualified on gfx1151/wave32, with native
W4A16 group128 backbones, FP16 activations/KV and BF16 selectors. Qwen retains
FP32 recurrent state and all 48 Gated DeltaNet blocks. OLMo's pooling, rotary,
geometry and support rules remain separate from Qwen's. Full KV storage remains
allocated; sparse attention does not imply KV-memory savings.

OLMo's native contract is batch one. Qwen's measured full-model performance is
batch one at 4K, 16K and 32K; primitive/batched checks do not qualify other
full-model workloads. No valid Qwen64K timing is available. Cache history must be
contiguous, with prefill followed by single-token decode and in-place KV;
continuous batching, rollback and arbitrary model geometries are not covered.

See the [OLMo contract](olmo-qsa-native-gqa.md), [OLMo native results](../experiments/olmo-qsa-native-gqa-results.md),
[Qwen contract](qwen-qsa-native-gqa.md), [four-selector evidence](../reproducibility/qwen-hip-ep/README.md)
and [four/eight data coverage](../reproducibility/qwen-eight-hip-ep/README.md).
The Eight BFCL50 pilot is complete; Eight GSM8K, native PPL and application-memory
comparisons are unmeasured. The source Eight checkpoint still misses its fixed
32K PPL gate. Neither support nor the task pilot implies accepted long-context
quality. Historical measurements identify their measured binaries separately
from the final PR source.
