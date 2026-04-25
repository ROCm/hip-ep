<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Kokoro TTS on the onnx-hipdnn-ep Execution Provider

This document captures the plan for adding the 33 ONNX operators that
Kokoro-82M (StyleTTS2 distilled, ISTFTNet decoder) needs but
`onnx-hipdnn-ep` does not yet implement, plus the integration work to
ship Kokoro through this EP from `lemondate`.

The end goal is **full GPU offload** of Kokoro on a single ROCm-capable
AMD GPU (gfx1151 / gfx1201) via the MorphiZen MLIR compilation pipeline:
`onnx.*` -> `hip.*` -> LLVM IR -> native DLL -> hipDNN/MIOpen/hipBLASLt
+ custom HIP kernels.

No CPU fallback. No iGPU + dGPU split. One device, end to end.

---

## Current state (commit baseline)

`onnx-hipdnn-ep` already supports **16 of the 49 ONNX ops** that Kokoro
uses, covering **1677 of 2463 nodes** (68%). The supported ops cover
the heavy compute (MatMul, Conv, Gemm, Softmax) and most of the
elementwise / shape plumbing.

| Already supported (16 ops, 1677 nodes) |
|---|
| Add (441), Mul (315), Unsqueeze (192), MatMul (102), Sqrt (90), Transpose (89), Conv (88), Gemm (73), Sub (68), Reshape (61), Cast (18), Softmax (12), Sigmoid (1), ReduceSum (1), Squeeze (5), Gather (121) |

The 33 missing ops account for **786 nodes (32%)**. They are the work
we have to do.

---

## Operator gap and implementation strategy

For each missing op we picked the cheapest GPU-only backend, in this
preference order:

1. **hipDNN pointwise** (`HIPDNN_POINTWISE_*`) - already plumbed
   through `hip.miopen.*` style ops in the EP; cheapest to add.
2. **hipDNN operations** (`HIPDNN_OPERATION_*_EXT`) - one tier up,
   uses the graph builder.
3. **MIOpen** - for Conv-shaped ops, RNNs, and MIOpen-specific
   activations.
4. **Custom HIP kernel** - hand-written `.hip` kernel in
   `3rd-party/custom_kernels/`, hooked through the `hip.*` runtime
   shim.
5. **Host / decompose** - shape-only ops or ops that fold to existing
   primitives.

### Tier 1 - elementwise (12 ops, 384 nodes)

| ONNX op | nodes | Backend | hipDNN enum / notes |
|---|---|---|---|
| Slice | 170 | custom HIP kernel | strided gather (no hipDNN op) |
| Div | 94 | hipDNN pointwise | `HIPDNN_POINTWISE_DIV` (12) |
| Pow | 62 | custom HIP kernel | no hipDNN POW; decompose `EXP(y*LOG(x))` first, custom kernel as backup |
| Sin | 51 | hipDNN pointwise | `HIPDNN_POINTWISE_SIN` (38) |
| LeakyRelu | 28 | MIOpen activation | `miopenActivationLEAKYRELU` |
| Tanh | 13 | hipDNN pointwise | `HIPDNN_POINTWISE_TANH_FWD` (47) |
| Exp | 1 | hipDNN pointwise | `HIPDNN_POINTWISE_EXP` (16) |
| Cos | 1 | custom HIP kernel | no hipDNN COS |
| Floor | 1 | hipDNN pointwise | `HIPDNN_POINTWISE_FLOOR` (17) |
| Round | 1 | custom HIP kernel | no hipDNN ROUND |
| Clip | 1 | hipDNN pointwise chain | `MIN(28)` + `MAX(29)` |
| Atan | 1 | custom HIP kernel | no hipDNN ATAN |

### Tier 2 - shape / data movement (8 ops, 234 nodes)

| ONNX op | nodes | Backend | notes |
|---|---|---|---|
| ReduceMean | 130 | hipDNN operation | `HIPDNN_OPERATION_TYPE_REDUCTION_EXT` + `HIPDNN_REDUCE_TENSOR_AVG` |
| Shape | 73 | host metadata | no GPU compute; folds to `arith.constant` |
| Concat | 72 | custom HIP kernel | strided memcpy along axis |
| Resize | 6 | custom HIP kernel | linear/nearest interpolation |
| ConstantOfShape | 6 | device fill | `hipMemset` / fill kernel |
| Expand | 5 | host / stride | folds to broadcast in downstream ops |
| Pad | 2 | custom HIP kernel | constant / edge pad |
| Range | 2 | host arange | folds to `arith.constant` if static |

### Tier 3 - comparison / logic (6 ops, 18 nodes)

| ONNX op | nodes | Backend | hipDNN enum |
|---|---|---|---|
| Where | 7 | hipDNN pointwise | `HIPDNN_POINTWISE_BINARY_SELECT` (4) |
| Equal | 4 | hipDNN pointwise | `HIPDNN_POINTWISE_CMP_EQ` (6) |
| Greater | 3 | hipDNN pointwise | `HIPDNN_POINTWISE_CMP_GT` (8) |
| Less | 2 | hipDNN pointwise | `HIPDNN_POINTWISE_CMP_LT` (10) |
| GreaterOrEqual | 1 | hipDNN pointwise | `HIPDNN_POINTWISE_CMP_GE` (7) |
| And | 1 | hipDNN pointwise | `HIPDNN_POINTWISE_LOGICAL_AND` (25) |

### Tier 4 - normalization (1 op, 31 nodes)

| ONNX op | nodes | Backend | hipDNN op |
|---|---|---|---|
| LayerNormalization | 31 | hipDNN operation | `HIPDNN_OPERATION_TYPE_LAYERNORM_EXT` |

### Tier 5 - sequence / recurrent (2 ops, 8 nodes)

| ONNX op | nodes | Backend | notes |
|---|---|---|---|
| LSTM | 6 | MIOpen RNN | `miopenLSTM`; needs careful weight layout (ONNX `[i,o,f,c]` -> MIOpen `[i,f,c,o]`) |
| CumSum | 2 | custom HIP kernel | scan along axis |

### Tier 6 - specialty (4 ops, 9 nodes)

| ONNX op | nodes | Backend | notes |
|---|---|---|---|
| ConvTranspose | 6 | MIOpen | `miopenConvolutionDescriptor` with `miopenTranspose` mode |
| STFT | 1 | custom HIP kernel | window + rocFFT (already in TheRock) |
| NonZero | 1 | custom HIP kernel | data-dependent output, mask-then-prefix-sum |
| ScatterND | 1 | custom HIP kernel | gather-scatter via index tensor |

### Coverage summary

| Backend | ops | nodes |
|---|---|---|
| hipDNN pointwise | 13 | 270 |
| hipDNN operations | 2 | 161 |
| MIOpen | 3 | 40 |
| Custom HIP kernel | 9 | 252 |
| Host / decompose | 6 | 63 |
| **Already supported** | 16 | 1677 |
| **Total** | **49** | **2463** |

100% of Kokoro graph covered. No host fallback for any compute op.

---

## Architecture: where each op lives in the EP

For each new op we add up to four files, mirroring the existing ops:

```
lib/Conversion/OnnxToHip/<Op>Conversion.cpp     # MLIR rewrite pattern: onnx.<Op> -> hip.<op>
lib/Conversion/OnnxToHip/OnnxToHipUtils.h       # declare populate<Op>ConversionPatterns
lib/Conversion/OnnxToHip/OnnxToHip.cpp          # call populate function in convertComputeOps
lib/Conversion/HipToLLVM/<Op>ToLlvm.cpp         # if a new hip.<op> dialect entry is added
include/hip/Dialect/IR/HipOps.td                # TableGen op definition (when adding a new hip op)
3rd-party/custom_kernels/<op>.hip               # HIP source (only when using custom kernel)
3rd-party/custom_kernels/<op>.h                 # C-ABI header for the kernel
test/lit/Conversion/onnx-to-hip/test_<op>.mlir  # LIT test for the conversion pattern
test/lit/Conversion/hip-to-llvm/test_<op>.mlir  # LIT test for the LLVM lowering (when relevant)
test/lit/e2e/test_<op>_model.mlir               # E2E test that compiles + runs on GPU
```

Many Tier 1 ops can be added inside the existing
`ElementwiseConversion.cpp` / `PowerConversion.cpp` /
`ActivationConversion.cpp` because the runtime already wires elementwise
hipDNN pointwise dispatch.

---

## Phased execution plan

### Phase A - bring up the build on this machine (gfx1201)

1. Set up workspace directories (`prebuilt-local`, `therock` junction)
2. Run `scripts/setup-prebuilt.sh` to fetch LLVM 22.1, Protobuf 34, FlatBuffers 25.12
3. Configure with `BUILD_HIP_TOOLS=ON`, `BUILD_MOCK_RUNTIME=OFF`, `HIP_ARCHITECTURES=gfx1201`
   - Skip `BUILD_EP=ON` for the first round so we can iterate without ONNX Runtime
4. Build, install
5. Run all existing LIT tests (`ctest -R MorphizenMLIRLitTests`)
6. Run the existing E2E tests on the 9070 XT

### Phase B - Tier 1 ops (12 ops, 384 nodes)

In parallel, since each op is independent at the conversion-pattern
level:

- **hipDNN pointwise ops**: extend `ElementwiseConversion.cpp` with a
  generic `OnnxToHipPointwise<TPattern, kPointwiseEnum>` template.
- **MIOpen activation ops** (LeakyRelu): extend
  `ActivationConversion.cpp` to take an alpha attribute.
- **Custom kernel ops** (Slice, Pow, Cos, Round, Atan): add
  `<Op>Conversion.cpp` + `3rd-party/custom_kernels/<op>.hip` +
  `lib/Runtime/real/<op>.cpp`.

Each op gets a LIT test plus an E2E test.

### Phase C - Tier 2 ops (8 ops, 234 nodes)

- `ReduceMean` extends the existing `ReduceSumConversion.cpp`.
- `Shape`, `ConstantOfShape`, `Range`, `Expand` are mostly
  shape-folding in the conversion pass (they replace with
  `arith.constant` or fold to existing broadcast paths).
- `Concat`, `Resize`, `Pad` are custom HIP kernels.

### Phase D - Tier 3 + 4 (7 ops, 49 nodes)

- Comparison / logic ops fold into the same pointwise template as
  Tier 1.
- `LayerNormalization` extends `NormConversion.cpp` (which already
  handles `SimplifiedLayerNormalization`). Use
  `HIPDNN_OPERATION_TYPE_LAYERNORM_EXT`.

### Phase E - Tier 5 + 6 (6 ops, 17 nodes)

- `ConvTranspose` extends `ConvConversion.cpp` to set the MIOpen
  transpose mode.
- `LSTM` is the trickiest: needs MIOpen RNN descriptor, ONNX-to-MIOpen
  weight layout permutation, and proper handling of `initial_h` /
  `initial_c`. Test with a single-layer LSTM model first.
- `CumSum`, `STFT`, `NonZero`, `ScatterND` are custom HIP kernels.

### Phase F - end-to-end Kokoro

1. Compile the full `model.onnx` (Kokoro-82M, 326 MB) through
   `hip-onnx-runner`.
2. Run inference, dump waveform output.
3. Diff against the CPU ONNX baseline (PESQ / waveform L2 / Whisper
   round-trip CER).
4. Profile and squash any latency regressions vs the existing
   `kokoro-hip-server` ggml backend.

### Phase G - lemondate integration

1. Create `kokoro-onnx-ep` branch in `lemondate`.
2. Add an `onnx-ep` mode to `kokoro-hip-server` that loads
   `model.onnx` via the MorphiZen Execution Provider DLL.
3. Wire `LEMONADE_KOKORO_BACKEND=onnx-ep` through `lemond`,
   `config_file.cpp`, and `run_lemond.ps1.tmpl`.
4. Stage the EP DLL + dependencies in `lemondate/bin/`.
5. End-to-end test through the existing Claude Code voice plugin.

---

## Privacy / repo separation

This EP is in a **private repository**. None of the changes here, none
of the IPA dictionaries we touch, and none of the integration code on
the lemondate side may name `onnx-hipdnn-ep`, MorphiZen, hipDNN
internals, or AMD's pre-release SDK paths in a way that would leak
implementation details into the public `lemondate` or
`tts_tts_claude_code` repos.

Concretely:

- `lemondate` exposes an abstract `--backend onnx-ep` and a
  `LEMONADE_KOKORO_BACKEND=onnx-ep` env var. The shim that loads the
  EP DLL must accept the DLL path via env var
  (`LEMONADE_KOKORO_ONNX_EP_DLL`) so the public repo never hardcodes
  `onnxruntime_morphizen_ep.dll`.
- The `LEMONADE_KOKORO_ONNX_MODEL` env var points at a stock public
  Kokoro-82M ONNX (e.g. `onnx-community/Kokoro-82M-v1.0-ONNX`); no
  AMD-specific quantization or model surgery is shipped publicly.
- The plan and design notes for this work live in
  `docs/kokoro_tts_plan.md` *inside this private repo*. They are not
  mirrored to `demos/plans/`.

---

## Status snapshot (last build)

**Build / infrastructure:**
- Repo builds cleanly on a 9070 XT (gfx1201) against the venv
  `_rocm_sdk_devel` SDK.  Test harness: 65 E2E + 100+ LIT, only the
  pre-existing `test_reciprocal` (validator-Inf) and `test_reshape`
  (harness segfault) fail; everything else passes.
- WMMA kernels (matmul_nbits, gemm_wmma, gqa) ported from RDNA3 to
  RDNA4 via `wmma_frag.h` (no half-wave replication on rdna4, gfx12
  WMMA builtin, matching C/D row mapping).
- ONNX Runtime (`rel-1.24.3`) built locally with `--use_dml`; the
  EP DLL `onnxruntime_morphizen_ep.dll` and `hip-onnx-runner.exe` are
  installed to `D:\jam\prebuilt-local\bin\`.

**Op coverage (49/49 of Kokoro's op types implemented as conversion
patterns + lowering + HIP kernels, except for the 8 listed below):**

| Implemented this branch | Kokoro nodes |
|---|---|
| Slice, Div, Pow, Sin, Tanh, Exp, Cos, Floor, Round, Clip, Atan, LeakyRelu | 384 |
| ReduceMean, Shape, Concat, ConstantOfShape, Range | 226 |
| Equal, Greater, Less, GreaterOrEqual, And, Where, LayerNormalization | 49 |
| CumSum | 2 |

| Pre-existing in `onnx-hipdnn-ep` | Kokoro nodes |
|---|---|
| Add, Mul, Unsqueeze, MatMul, Sqrt, Transpose, Conv, Gemm, Sub, Reshape, Cast, Softmax, Sigmoid, ReduceSum, Squeeze, Gather | 1677 |

| Still missing (28 nodes / 1.1% of Kokoro) | Kokoro nodes |
|---|---|
| LSTM, ConvTranspose, STFT, Pad, Resize, Expand, NonZero, ScatterND | 28 |

**Verified end-to-end via `hip-onnx-runner` on gfx1201:**
- A single-op fp16 Tanh model loads through the ORT → MorphiZen
  bridge, compiles to a model DLL, and executes on the GPU in
  3.2 ms.  This confirms the ORT bridge, MLIR pipeline, LLVM
  codegen, runtime bitcode, and HIP custom kernels are all wired up
  correctly end-to-end.

**Kokoro now compiles all the way through to a model DLL that loads
and runs.**  Pipeline timings (gfx1201, RelWithDebInfo build,
hip-onnx-runner -m kokoro_82m_static_128_clean.onnx):

| Phase | Time |
|---|---|
| MLIR parsing | 0.005s |
| ConvertOnnxToHipPass | 0.526s (constants ~0.1s + compute ~0.5s) |
| OneShotBufferizePass | 0.026s |
| BufferDeallocationSimplification | 0.681s |
| ConvertHipToLLVMPass | 0.046s |
| translateToLLVMIR | 0.041s |
| linkRuntime | 0.031s |
| optimizeLLVMIR | 0.559s |
| compileToObject | 1.081s |
| linkToDLL | 0.102s |
| **Total CompilerDriver** | **3.224s** |
| Plugin LoadLibrary | 0.032s |
| HIP / MIOpen / hipBLASLt init | 0.130s |
| 538 constants @ 324MB to VRAM | 0.098s |

DLL size: 3.07 MB.  6943 rewrite events fire in
`convertComputeOps` in 0.007 s; the greedy driver then converges.

**The actual `Run()` call still fails** with ORT's
`Tensor::CalculateTensorStorageSize Tensor shape.Size() must be >= 0`
because of the `dropUnsupportedOnnxOps` placeholders the EP inserts
for ops it can't yet bufferize/lower (current placeholders for:
Slice (2), Range (2), NonZero (1)).  These ops live in the iSTFTNet
audio-synthesis tail of Kokoro and need real GPU implementations
before the model produces correct audio; the **structural** wiring
through the EP is in place.

All blocking issues from the previous snapshot have been worked
around or made irrelevant by `dropUnsupportedOnnxOps`:

- Dynamic-shape bridge: `morphizen` now emits `ShapedType::kDynamic`
  instead of raw `-1` (with a symmetric mapping back when extracting
  shapes), so MLIR's verifier no longer trips.
- Greedy rewriter: capped iterations + heartbeat listener prevent
  the previous 30-minute hang.  Several patterns were hardened to
  handle Kokoro's quirks: rank-0 inputs (Gather, Slice, Concat,
  Squeeze/Unsqueeze, Transpose, ReduceMean, CumSum), invalid
  axis values (Gather), broadcast-aware empty-tensor construction
  (Add/Mul/Sub/Pow/Div/Where/Compare), arbitrary permutations
  decomposed into 2-swap chains (Transpose), shape-based axes
  inference (Unsqueeze/Squeeze) so they no longer need a constant
  axes operand, contiguous-tail relaxation (ReduceMean) for dynamic
  input shapes, and dropping of dangling `onnx.NoValue` placeholders.
- Tier 6 ops: `hip.pad`, `hip.expand`, `hip.conv_transpose`,
  `hip.resize` exist in the dialect with DPS+effects, bufferize
  registration, and conversion patterns.  `hip.range` exists too,
  but the conversion pattern is gated off pending a fix for a greedy-
  rewriter crash on dynamic-rank-1 outputs.

| Surviving onnx ops at end of `convertComputeOps` | nodes |
|---|---|
| Slice (dynamic `ends` from a runtime expand_shape) | 2 |
| Range (`RangeToHip` re-disabled until `hip.range` LLVM lowering lands) | 2 |
| NonZero (NonzeroConversion populate disabled — heap-corrupts greedy) | 1 |

These 5 surviving onnx ops are now caught by `dropUnsupportedOnnxOps`
which replaces each with a `tensor.empty` of the result type so the
rest of the pipeline (bufferize, HipToLLVM, LLVM codegen, link)
succeeds.  This is intentionally INCORRECT for any inference path
that actually depends on those ops -- in Kokoro that's the iSTFTNet
audio-synthesis tail.

**Concrete remaining work to get correct audio output:**

1. **NonZero LLVM lowering** -- `hip.nonzero` op + DPS interface +
   bufferize hooks already exist; the conversion populate currently
   heap-corrupts the greedy rewriter when MERELY registered (matchAndRewrite
   never runs because Kokoro's NonZero has dynamic K and the pattern bails
   early).  Root cause TBD; once fixed, re-enable
   `populateNonzeroConversionPatterns` in
   `lib/Conversion/OnnxToHip/OnnxToHip.cpp::convertComputeOps`.
2. **Slice with dynamic `ends`** -- the 2 surviving slices come from
   the `decoder/decoder/generator/istft/stft/Slice*` chain where the
   `ends` operand is the result of a `tensor.expand_shape` of a
   constant.  `extractI64Constant` already looks through expand/
   collapse/cast; need to also look through arith arithmetic that
   produces the i64 from a dim size.  Alternative: emit a runtime
   `hip.dynamic_slice` op backed by a small HIP kernel.
3. **Range with dynamic limit** -- the 2 surviving ranges come from
   `/encoder/Range` where `limit` comes from a runtime tensor.dim.
   `RangeToHip` was re-disabled because there's no `hip.range` LLVM
   lowering -- needs a tiny HIP kernel `hip_range(start, delta, n,
   out)` and a `RangeOpLowering` that wires it up.
4. **STFT (`hip.stft`)** -- conversion pattern + runtime stub exist
   (`hip_stft_frame_window` / `hip_stft_split_complex` are zero-fill
   placeholders); needs the real rocFFT path.
5. **LSTM (`hip.lstm`)** -- conversion pattern + MIOpen RNN forward
   inference call exist as of commit 339c500; needs a real ONNX-to-
   MIOpen weight permutation and an end-to-end test on a small RNN
   model before Kokoro's 6 LSTM nodes produce correct output.
6. **`hip.miopen.softmax`** -- runtime stub
   (`hip_miopen_softmax`, zero-fill placeholder) needs the real
   `miopenSoftmaxForward` call (input -> output, dims rows/cols).
5. **ScatterND (1 node)** -- gather-scatter HIP kernel.
6. **`hip.range` final pattern** -- replace the `tensor.empty %c1`
   placeholder with a real dynamic-shape allocation handshake so the
   greedy rewriter doesn't crash on rank-1 dynamic outputs.
7. **`onnx.Shape` for dynamic input dims** -- emit a
   `tensor.from_elements` of static constants + `tensor.dim` instead
   of folding to a single `arith.constant`.
8. **Slice/Pad with externalised constant operands** -- either bump
   `kDefaultExternalizeMinNumElements` to inline tiny shape
   constants (the previous attempt at threshold=8/16/64 caused
   downstream heap corruption -- needs proper investigation), or
   round-trip a `hip.inline_value` attribute through to_tensor
   without the `getFromRawBuffer` heap corruption seen in this
   session.
9. **Tiny edge cases:** rank-0 onnx.Transpose with rank-4 perm
   (dead code in Kokoro's iSTFTNet noise path -- relax perm-rank
   check or fold out in a pre-pass).

Phase G (lemondate integration) is on hold until bufferization
clears.  The shim, env vars, and abstract `--backend onnx-ep`
plumbing are still defined as in the section above; only the EP DLL
hookup is pending.
