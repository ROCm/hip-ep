<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
---
name: add-hip-ep-op
description: Add a new operator end-to-end across the hip-ep MLIR compiler stack (HIP dialect, ONNX-to-HIP conversion, bufferization, HIP-to-LLVM lowering, runtime wrapper, HIP device kernel, tests). Use when adding, implementing, or wiring up support for an ONNX op (native onnx.<Op> or com.microsoft onnx.Custom) in the hip-ep / hipgpu execution provider, or when an op fails with "op was not bufferized" or falls back to CPU.
---

# Add hip-ep Op End-to-End

Add a new op across the full hip-ep MLIR compiler stack. First gather the op
details if not provided: op name, ONNX source (native `onnx.<Op>` or
`com.microsoft` `onnx.Custom`), math formula, attributes (name/type/default),
and supported dtypes.

Model each layer on an existing similar op:
- Unary custom-kernel op: GELU
- Float `alpha` attribute plumbing: LeakyRelu
- `onnx.Custom` + `com.microsoft` domain matching: Gqa

Missing any layer causes silent CPU fallback; a missing bufferization
registration specifically produces `error: op was not bufferized`.

Implement every layer below, then stop for the user to build.

## 1. HIP Dialect
- `hip-ep/include/hip/Dialect/IR/HipOps.td`: `def Hip_<Name>Op : Hip_DpsOp<"snake_name">`
  with `Hip_ContextType:$ctx`, `Hip_TensorOrMemRef:$input/$output`, plus any attrs
  (e.g. `DefaultValuedAttr<F64Attr, "1.702">:$alpha`).
- `hip-ep/lib/Dialect/IR/HipDialect.cpp`: define `getDpsInitsMutable` + `getEffects`.

## 2. ONNX -> HIP conversion
- New `hip-ep/lib/Conversion/OnnxToHip/<Name>Conversion.cpp` with
  `populate<Name>ConversionPatterns`. For `com.microsoft` ops, match `"onnx.Custom"`
  and check `function_name` + `domain_name`. Build the op with `createEmptyTensor`
  init + `getContextArg`.
- Declare the populate fn in `OnnxToHipUtils.h`, call it in `OnnxToHip.cpp`
  `convertComputeOps`, add the source to `OnnxToHip/CMakeLists.txt`.

## 3. HIP -> memref (bufferization)
- `hip-ep/include/hip/Dialect/IR/HipBufferize.h`: add
  `<Name>Op::attachInterface<HipDstBufferizableModel<<Name>Op>>(*ctx);` in
  `registerHipBufferizableOpInterfaceModels`.

## 4. HIP -> LLVM lowering
- `hip-ep/lib/Conversion/HipToLLVM/HipToLLVMUtils.h`: add `kWrap<Name> = "wrap_<name>"`.
- `ActivationLowering.cpp` (or the relevant lowering file): add `<Name>OpLowering`
  emitting the `wrap_<name>` call; register it in the `populate*LoweringPatterns` list.

## 5. LLVM wrapper / host runtime
- Declare `wrap_<name>` in `hip-ep/lib/Runtime/hipdnn_ep_runtime.h`.
- Implement it in `hip-ep/lib/Runtime/real/<area>.cpp` dispatching to the kernel.
- Add a mock stub in `hip-ep/lib/Runtime/mock/mock_gpu.cpp`.

## 6. HIP device kernel
- New `hip-ep/lib/Runtime/Kernels/hip/<name>_kernel.hip` (traits for f32/f16/bf16/f64;
  `extern "C" int hip_<name>(...)` with dtype switch).
- Declare `HIP_KERNEL_API int hip_<name>(...)` in `Kernels/include/hip_custom_kernels.h`.
- Add the `.hip` to `Kernels/CMakeLists.txt` `_kernel_sources`.

## 7. Tests
- Lit: `test/lit/Conversion/onnx-to-hip/test_<name>.mlir` + e2e in `test/lit/e2e/`.
- Numeric: extend the matching `test/numeric/tests/test_*.py`.

## Build & run notes (report these; do not build unless asked)
Rebuild via `build/hip-ep`; the kernel needs `custom_kernels_gfx1151` rebuilt too.
Run on gfx1151 with the NATIVE artifact path (avoids the LLVM_IR JIT crash), TheRock
DLLs on PATH, and a cleared model cache:

```powershell
$env:THEROCK_DIST="C:\Users\anilm\workspace\repos\rocm\therock-dist-windows-gfx1151-7.11.0"
$env:PATH="$env:THEROCK_DIST\bin;$env:PATH"
del $env:TEMP\morphizen_mlir_*
./onnxruntime_perf_test.exe --plugin_ep_libs "hipgpu|hipgpu.dll" --plugin_eps "hipgpu" `
  --plugin_ep_options "artifact_format|NATIVE" -t 1 -c 1 -s -I <MODEL_PATH>
```
