<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->

# CPU fallback ONNX assets

## `cpugate_shell.onnx` (active)

Minimal valid ONNX shell graph with a single custom op `com.amd.morphizen.cpu.CpuGate`
(`i` → `o`, fixed `UINT8[1]` placeholders). Used only by the debug CPU-fallback CPUGate
(Quark-style): one inner CPU `Ort::Session` borrows `OrtKernelContext*`; real ops
run via `Ort::Op::Create` + `Ort::Op::Invoke(kernel_ctx_, …)`.

Regenerate:

```bash
python backend-mlir-compiler/custom-op-mlir/scripts/gen_cpugate_shell_onnx.py \
  backend-mlir-compiler/custom-op-mlir/onnx/cpugate_shell.onnx
```

CMake also runs this script at build time and embeds bytes into
`cpugate_shell_onnx_data.inc` for `cpu_fallback_cpugate.cpp`.

Gather CPU fallback uses the same generic path as other ops: `hipdnn_cpu_fb_try_gather`
→ `HipdnnCpuFbGenericDesc` → CPUGate `Ort::Op::Invoke("Gather", …)` on the gate
thread. `invoke_generic_cpu` handles int32→int64 index widen and ONNX output-rank
correction for Gather only.
