<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->

# CPU fallback ONNX assets

## `cpugate_shell.onnx` (active)

Minimal valid ONNX shell graph with a single custom op `com.amd.morphizen.cpu.CpuGate`
(`i` → `o`, `uint8` placeholders). Used only by the debug CPU-fallback CPUGate
(Quark-style): one inner CPU `Ort::Session` borrows `OrtKernelContext*`; real ops
run via `Ort::Op::Create` + `Ort::Op::Invoke(kernel_ctx_, …)`.

Regenerate:

```bash
python backend-mlir-compiler/custom-op-mlir/scripts/gen_cpugate_shell_onnx.py \
  backend-mlir-compiler/custom-op-mlir/onnx/cpugate_shell.onnx
```

CMake also runs this script at build time and embeds bytes into
`cpugate_shell_onnx_data.inc` for `cpu_fallback_cpugate.cpp`.

Gather CPU fallback (`cpu_fallback_bridge.cpp`): large tensors use a host reference
implementation (dense row-major, matches the GPU `gather_kernel` layout); tensors at or
below 64 MiB may fall back to CPUGate + `Ort::Op::Invoke("Gather", …)` with optional
fp32 promotion for fp16/bf16 storage types.
