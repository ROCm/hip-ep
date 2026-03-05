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
| [MLIR-COMPILATION-OVERVIEW.md](MLIR-COMPILATION-OVERVIEW.md) | Draft | 2026-03-02 | Compilation pipeline overview |
| [RUNTIME-ARCHITECTURE.md](RUNTIME-ARCHITECTURE.md) | Self-Reviewed | 2026-02-12 | Runtime state and context design |
| [COMPILATION-OPTIONS.md](COMPILATION-OPTIONS.md) | Draft | 2026-03-03 | Compiler options (opt_level, output_mode, constants_file) |
| [CONSTANT-HANDLING-DESIGN.md](CONSTANT-HANDLING-DESIGN.md) | Self-Reviewed | 2026-02-13 | Model constants (weights, biases) |
| [MEMORY-MANAGEMENT.md](MEMORY-MANAGEMENT.md) | Draft | 2026-03-02 | GPU memory allocation strategy |
| [02b-OneShotBufferize.md](mlir/passes/02b-OneShotBufferize.md) | Draft | 2026-03-03 | Bufferization pipeline (one-shot-bufferize) |
| [DYNAMIC-SHAPE-DESIGN.md](DYNAMIC-SHAPE-DESIGN.md) | Draft | 2026-02-10 | Runtime-determined tensor dimensions |
| [EPCONTEXT-MEMORY-OPTIMIZATION.md](EPCONTEXT-MEMORY-OPTIMIZATION.md) | Draft | 2026-02-13 | EP context memory optimization |
| [alternatives/NATIVE-VS-IR-COMPARISON.md](alternatives/NATIVE-VS-IR-COMPARISON.md) | Self-Reviewed | 2026-02-13 | Native DLL vs LLVM IR storage |
| [mlir/LOWERING-PIPELINE.md](mlir/LOWERING-PIPELINE.md) | Draft | 2026-03-02 | Transformation stages |
| [mlir/INTERFACE-DESIGN.md](mlir/INTERFACE-DESIGN.md) | Draft | - | C interface and prerequisites |
| [mlir/HIP-DIALECT-DESIGN.md](mlir/HIP-DIALECT-DESIGN.md) | Draft | 2026-03-02 | HIP dialect and operations |
| [mlir/passes/01-HipAddContextArg.md](mlir/passes/01-HipAddContextArg.md) | Draft | 2026-03-03 | hip-add-context-arg pass |
| [mlir/passes/02-OnnxToHip.md](mlir/passes/02-OnnxToHip.md) | Draft | 2026-03-02 | ONNX → HIP dialect lowering (tensor-first) |
| [mlir/passes/03-Canonicalization.md](mlir/passes/03-Canonicalization.md) | Draft | 2026-03-02 | Canonicalization after bufferization |
| [mlir/passes/04-MemoryPooling.md](mlir/passes/04-MemoryPooling.md) | Draft | 2026-03-02 | Memory pooling (strip packing) |
| [mlir/passes/04a-MemoryPoolingAlgorithm.md](mlir/passes/04a-MemoryPoolingAlgorithm.md) | Draft | 2026-03-02 | Strip packing algorithm detail |
| [mlir/passes/05-HipToLLVM.md](mlir/passes/05-HipToLLVM.md) | Draft | 2026-03-02 | HIP → LLVM lowering |
| [mlir/passes/06-GenerateInterfacePass.md](mlir/passes/06-GenerateInterfacePass.md) | Self-Reviewed | 2026-02-12 | C interface generation pass |
| [mlir/passes/WHY-HIP-WRAPPERS.md](mlir/passes/WHY-HIP-WRAPPERS.md) | Draft | - | Justification for wrapper approach |
