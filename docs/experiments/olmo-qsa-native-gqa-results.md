<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# OLMo QSA: native GQA integration

Both variants run one complete ONNX graph in one session. The native QSA specialization replaces only attention middles 7/11/15/23. Layer 19 stays dense. All 224 backbone projection nodes and packed quantized weights remain unchanged. Batch is one; selectors are unchanged BF16; training updates are zero.

## Execution and correctness

| Check | Result |
|---|---|
| Dense Original full graph versus dense boundary reference | Passed |
| Native QSA full graph versus frozen boundary reference | Passed |
| Prefixes 17/517/1025/4092 plus four cached calls | Finite; exact native QSA replay |
| Native QSA full 4K versus cached 4092+4 | Passed |
| ORT dispatches per forward | Original: 1; Changed: 1 |
| Frozen selector and persistent cache checks | Passed |

GPU validation uses the same new EP build for both complete models and their reference backbone. The selector math remains BF16 input/parameters/output with FP32 accumulation. Decode uses a native GEMV; prefill uses native BF16 hipBLASLt. There are no Torch/ctypes/session handoffs at attention boundaries. Internal staged HIP kernels still perform the additional selector and sparse-attention work.

## Paired Dolma quality

| Original PPL | Changed PPL | Change | Paired 95% interval | 5% gate |
|---:|---:|---:|---|---|
| 6.434833 | 6.523952 | +1.385% | [+1.166, +1.632]% | Pass |

Same 64 held-out 4096-token windows, 4095 prefill targets and 64 cached targets per document. These windows are reused from the source project. Original/Changed order alternates by document. The gate is relative to native Original in this build. This does not establish broad long-context quality.

## Decode, separate timer

| Prefix tokens | Original tokens/s | Changed tokens/s | Throughput change | Paired 95% interval | Pairs |
|---:|---:|---:|---:|---|---:|
| 4,096 | 25.9814 | 26.6066 | +2.406% | [+1.401, +3.372]% | 6 |
| 16,384 | 21.9431 | 23.6208 | +7.645% | [+6.932, +8.203]% | 6 |
| 32,768 | 17.7470 | 20.5210 | +15.631% | [+15.140, +16.090]% | 6 |
| 65,472 | 12.5241 | 15.8495 | +26.552% | [+26.219, +26.941]% | 6 |

## Prefill, separate timer

| Prefix tokens | Original tokens/s | Changed tokens/s | Throughput change | Paired 95% interval | Pairs |
|---:|---:|---:|---:|---|---:|
| 4,096 | 1728.4559 | 1553.9591 | -10.096% | [-10.435, -9.748]% | 6 |
| 16,384 | 1515.3892 | 1363.1075 | -10.049% | [-10.296, -9.801]% | 6 |
| 32,768 | 1409.7014 | 1269.9645 | -9.913% | [-10.085, -9.764]% | 6 |
| 65,472 | 1212.3842 | 1104.6074 | -8.890% | [-8.999, -8.766]% | 6 |

Rates use balanced repeated Original/Changed pairs on the same physical GPU with identical token inputs, packed backbone weights, head work and persistent KV buffers. Two exact-shape warmups per variant verify finite outputs and exact replay. Timers are HIP-synchronized CPU wall time. Decode covers 64 fixed teacher-forced calls and excludes prefill, sampling, model loading and tensor/binding allocation. Prefill includes the final-position output head and excludes model loading and preallocation. Intervals use 10,000 paired bootstrap resamples with seed 42 and describe within-run variation. Aggregate and per-request rates are equal at batch one.

GSM8K is diagnostic and no longer an acceptance gate, as requested. The accepted 94/128 versus 91/128 pilot and earlier multi-session timings remain in the [boundary prototype report](olmo-qsa-results.md); those are not measurements of this native implementation. Native GSM8K has not been remeasured here. Only completed native timing suites appear above.

The ONNX frontend originally lacked BF16 constant type mapping. The native integration adds that mapping and a raw-byte dtype regression check, preserving selector storage instead of converting parameters. Compiler tests also cover native lowering and unsupported dtype/geometry rejection.

## Evidence

Hardware is the same Radeon 8060S gfx1151 / Ryzen AI MAX+ PRO 395 Windows machine used for the boundary prototype. The native runtime is built from this branch into a separate installation. Build commands, compiler/runtime versions, immutable export receipts, binary/source hashes, profiles and raw rows are retained under `_local/qsa/native-gqa/` and `_local/qsa/stages/`.

Validation passed 608 compiler LIT cases (9 unsupported), three runtime tests, three constant/dtype CPU tests and five Python export-contract tests. The final audit verifies the exact built source and installed binary hashes from `build-receipt.json`, as well as all 128 native Dolma rows and every completed timing pair.

See the [native operator design](../design/olmo-qsa-native-gqa.md) for its fixed geometry, cache/state contract and reproduction entry points. No support is claimed beyond the validated batch-one geometry or measured contexts.
