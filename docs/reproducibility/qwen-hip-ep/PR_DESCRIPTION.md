<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Add frozen native QSA specializations and reproducible experiment evidence

<!-- QWEN_EIGHT_PERFORMANCE_BEGIN -->
## Latest eight-selector performance

This performance addendum belongs to branch `rajeevp/qsa`; the historical four-selector merge-candidate validation below remains separately identified.

**Decode throughput improved over four selectors at every measured length.** Prefill is slower at 4K/16K and slightly faster at 32K. All eight weights, including the overlapping four positions, come from checkpoint 640; this is not a layer-count-only change.

| Context | Eight vs four: prefill | Eight vs four: decode |
|---|---:|---:|
| 4K | -5.21% | +0.71% |
| 16K | -6.31% | +1.94% |
| 32K | +0.92% | +5.31% |

Eight uses one warmup and three measured requests at batch one, with 64 common teacher-forced decode inputs. Default/four controls are reused across runs (two warmups, six requests, two resident sessions sharing cache buffers); Eight has one resident session. No paired confidence interval or isolated layer-count causal claim applies. Both prefill and decode use QSA at all eight selected layers.

The fixed 50-case BFCL component pilot scores Eight 29/50 versus reused Original 26/50; no new Original generation was run. This is not the official overall BFCL score. Eight GSM8K and application-memory comparisons remain unmeasured. The source checkpoint still misses its fixed 32K PPL gate; this pilot does not establish accepted long-context quality. See the [eight-selector report](../../experiments/qwen-qsa-eight-selector-results.md) and [raw-data paper handoff](../qwen-eight-hip-ep/README.md).
<!-- QWEN_EIGHT_PERFORMANCE_END -->

Frozen QSA now executes inside the existing full-model HIP-EP session instead
of crossing Python/Torch or extra ONNX-session boundaries at selected attention
layers. This one PR contains both native model specializations, BF16 selector
import, compiler lowering/verifiers, stateful runtime/cache handling, GPU kernels,
export tools, validation and reproducible performance/quality evidence.

| Model | Selector configurations in this PR |
|---|---|
| OLMo-3-1025-7B | Four: layers 7/11/15/23 |
| Qwen3.8-27B | Four: 27/31/35/39; Eight: 3/7/23/27/31/35/39/63 |

Layer indices are zero-based. OLMo Eight is not implemented. Both Qwen exports
use the same Qwen native operation; the Eight export loads every updated
checkpoint-640 selector, including the overlapping four positions. Selector
count is chosen by graph export. Both prefill and decode use QSA, with one EP
dispatch per forward; full KV storage remains allocated.

The qualified target is gfx1151/wave32 with the pinned native W4A16 group128
models, FP16 activations/KV, BF16 selectors and Qwen FP32 recurrent state. OLMo
is batch one; measured Qwen throughput is batch one at4K/16K/32K. Broader hardware,
continuous batching and arbitrary model geometries are not qualified. See the
[combined support scope](../../design/qsa-support.md).

The linked evidence retains Original/Four paired timing and the later Eight
across-run comparison, independent prefill/decode timers, raw rows and hashes.
The fixed BFCL50 component pilot scores Original26/50, Four24/50 and Eight29/50;
it is not the official overall BFCLv4 score. Eight GSM8K, native PPL and
application-memory comparisons remain unmeasured. Checkpoint640 still misses
the source fixed32K PPL gate; this PR does not claim accepted long-context quality.

The historical merged candidate passed736 compiler lit tests (nine unsupported),
eight CPU unit tests, six export tests and expanded GPU kernel/operator/full-model
controls; [AUDIT.md](AUDIT.md) and [candidate-validation.json](candidate-validation.json)
bind those checks to their exact source/binaries. The branch includes current
upstream main and the previously audited integration fixes. The final cleanup
checks are recorded separately in the [PR validation receipt](../qwen-eight-hip-ep/pr-validation.json).
Historical performance/accuracy remains attached to the preserved measured build.
Large model weights, compiled binaries/caches and local ZIP archives are not
part of the branch; the portable Markdown/JSON/CSV/raw evidence is included.
