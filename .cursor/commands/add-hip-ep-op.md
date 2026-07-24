<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Add hip-ep Op

Add a new operator end-to-end across the hip-ep MLIR compiler stack by reading
and following `.cursor/skills/add-hip-ep-op/SKILL.md`.

Op to add: $ARGUMENTS

Steps:

1. Read `.cursor/skills/add-hip-ep-op/SKILL.md` and follow it exactly.
2. If the op details are missing, first gather: op name, ONNX source (native
   `onnx.<Op>` vs `com.microsoft` `onnx.Custom`), math formula, attributes
   (name/type/default), and supported dtypes.
3. Model each layer on the closest existing op (GELU for a unary custom-kernel
   op, LeakyRelu for float-`alpha` plumbing, Gqa for `onnx.Custom` +
   `com.microsoft` matching).
4. Implement every layer: HIP dialect op, ONNX->HIP conversion (+ register in
   `OnnxToHipUtils.h` / `OnnxToHip.cpp` / `CMakeLists.txt`), bufferization
   registration, HIP->LLVM lowering (+ `kWrap<Name>`), runtime wrapper (real +
   mock + decl), HIP device kernel (+ header decl + `Kernels/CMakeLists.txt`),
   and tests (lit conversion + e2e, numeric).
5. Run the recently-edited files through the linter and fix issues.
6. Do NOT build unless asked — report the build/run notes from the skill
   (rebuild `build/hip-ep`, rebuild `custom_kernels_gfx1151`, clear
   `%TEMP%\morphizen_mlir_*`, run via the NATIVE artifact path).

A missing layer causes silent CPU fallback; a missing bufferization
registration specifically produces `error: op was not bufferized`.
