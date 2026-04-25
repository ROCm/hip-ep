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

**Kokoro itself does not yet load** through `hip-onnx-runner` due to
two issues that go beyond op coverage:

1. **Dynamic shapes.**  The MorphiZen ORT bridge serialises
   `unk__N` dims as raw negative literal sizes in MLIR types, which
   trips MLIR's `RankedTensorType::verify` (`invalid tensor
   dimension size`) before any of our passes run.  After
   `polygraphy surgeon sanitize --override-input-shapes
   input_ids:[1,128] style:[1,256] speed:[1] --fold-constants` the
   model still has 269 dynamic dims because Kokoro's iSTFTNet
   decoder predicts durations at runtime and uses them as shape
   inputs (`Range`, `ConstantOfShape` of `Cast` of a predicted
   length).  Possible fixes:
   - **(EP-side)** Patch the ORT-to-MLIR bridge in
     `morphizen-graph` / `onnx-ir-imp` to emit `?` (kDynamic)
     instead of the raw negative `dim_value`.  Most of our
     conversion patterns already require `hasStaticShape()` and
     would still bail, but the MLIR verifier would stop tripping
     and we'd fall back through ORT to CPU per-op.
   - **(Model-side)** Split Kokoro at the duration predictor and
     compile two static-shape subgraphs that meet at a fixed-size
     boundary (encoder → predictor on the input side; decoder
     starting from a precomputed maximum frame count on the audio
     side).  Effort: ~1 week of model surgery.

2. **Missing ops** (LSTM, ConvTranspose, STFT, NonZero, ScatterND,
   Pad, Resize, Expand) — even with dynamic shapes solved, these
   eight op types account for 28 nodes that need conversion +
   lowering + kernel implementations.  LSTM (MIOpen RNN with
   ONNX-to-MIOpen weight permutation) and STFT (rocFFT-backed) are
   the largest pieces; the others mirror the kernels we already have.

The next concrete step is whichever of {dynamic-shape bridge fix,
model surgery} the team prefers, followed by the LSTM / STFT
implementations.  Everything below the bridge is verified working.
