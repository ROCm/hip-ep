<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Design Documents

Architecture and design decisions for the ONNX HIP/DNN Execution Provider.

All documents in this directory (including subdirectories) have **Document Type: Design**.

## Document Status

| Document | Status | Date | Description |
|----------|--------|------|-------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Self-Reviewed | 2026-02-12 | Entry point with 7 major design decisions |
| [MLIR-COMPILATION-OVERVIEW.md](MLIR-COMPILATION-OVERVIEW.md) | Self-Reviewed | 2026-02-14 | Compilation pipeline overview |
| [RUNTIME-ARCHITECTURE.md](RUNTIME-ARCHITECTURE.md) | Self-Reviewed | 2026-02-12 | Runtime state and context design |
| [CONSTANT-HANDLING-DESIGN.md](CONSTANT-HANDLING-DESIGN.md) | Self-Reviewed | 2026-02-13 | Model constants (weights, biases) |
| [MEMORY-MANAGEMENT.md](MEMORY-MANAGEMENT.md) | Draft | 2026-02-14 | GPU memory allocation strategy |
| [BUFFER-LIFETIME-DESIGN.md](BUFFER-LIFETIME-DESIGN.md) | Draft | 2026-02-14 | Buffer deallocation using MLIR standard pipeline |
| [DYNAMIC-SHAPE-DESIGN.md](DYNAMIC-SHAPE-DESIGN.md) | Draft | 2026-02-10 | Runtime-determined tensor dimensions |
| [EPCONTEXT-MEMORY-OPTIMIZATION.md](EPCONTEXT-MEMORY-OPTIMIZATION.md) | Draft | 2026-02-13 | EP context memory optimization |
| [alternatives/NATIVE-VS-IR-COMPARISON.md](alternatives/NATIVE-VS-IR-COMPARISON.md) | Self-Reviewed | 2026-02-13 | Native DLL vs LLVM IR storage |
| [mlir/LOWERING-PIPELINE.md](mlir/LOWERING-PIPELINE.md) | Self-Reviewed | 2026-02-14 | Transformation stages |
| [mlir/INTERFACE-DESIGN.md](mlir/INTERFACE-DESIGN.md) | Draft | - | C interface and prerequisites |
| [mlir/HIP-DIALECT-DESIGN.md](mlir/HIP-DIALECT-DESIGN.md) | Draft | - | HIP dialect and wrappers |
| [mlir/passes/01-OnnxToHip.md](mlir/passes/01-OnnxToHip.md) | Self-Reviewed | 2026-02-13 | ONNX → HIP dialect lowering |
| [mlir/passes/05-HipToLLVM.md](mlir/passes/05-HipToLLVM.md) | Self-Reviewed | 2026-02-13 | HIP → LLVM lowering and wrappers |
| [mlir/passes/04-MemoryPooling.md](mlir/passes/04-MemoryPooling.md) | Draft | 2026-02-14 | Memory pooling optimization (60% savings) |
| [mlir/passes/06-GenerateInterfacePass.md](mlir/passes/06-GenerateInterfacePass.md) | Self-Reviewed | 2026-02-12 | C interface generation pass |
| [mlir/passes/WHY-HIP-WRAPPERS.md](mlir/passes/WHY-HIP-WRAPPERS.md) | Draft | - | Justification for wrapper approach |
