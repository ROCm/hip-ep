<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Architecture Design

**MLIR-based AOT Compilation with EPContext for AMD ROCm**

**Date:** 2026-02-12
**Document Type:** Architecture Design
**Review Status:** Self-Reviewed
**Branch:** `mlir-integration`

---

## Table of Contents

- [Problem](#problem)
- [System Architecture](#system-architecture)
- [Key Design Decisions](#key-design-decisions)
  - [1. Pattern-Based Lowering vs Manual Transformation](#1-pattern-based-lowering-vs-manual-transformation)
  - [2. Stateful Interface (init/compute/cleanup)](#2-stateful-interface-initcomputecleanup)
  - [3. Embedded Constants vs External Files](#3-embedded-constants-vs-external-files)
  - [4. Synchronous Execution vs Async](#4-synchronous-execution-vs-async)
  - [5. Full Model Fusion vs Per-Op Execution](#5-full-model-fusion-vs-per-op-execution)
  - [6. Standalone Resources vs Shared Context](#6-standalone-resources-vs-shared-context)
  - [7. LLVM IR Merging for Zero-Cost Runtime Abstraction](#7-llvm-ir-merging-for-zero-cost-runtime-abstraction)
- [Design Principles](#design-principles)
- [ONNX-MLIR Integration](#onnx-mlir-integration)
- [Open Architectural Questions](#open-architectural-questions)
  - [Native DLL vs LLVM IR Storage](#native-dll-vs-llvm-ir-storage)
- [References](#references)

---

## Problem

[ONNX Runtime](https://onnxruntime.ai/) execution with JIT compilation incurs significant overhead:
- Compilation delay on first inference startup
- Large runtime dependencies (LLVM/MLIR libraries)
- Repeated compilation: Same model recompiled on every process start

This architecture explores ahead-of-time (AOT) compilation approaches to reduce or eliminate JIT overhead by pre-compiling models and storing compiled artifacts in [EPContext](https://onnxruntime.ai/docs/execution-providers/EP-Context-Design.html).

---

## System Architecture

### Two-Stage Architecture

The system separates compilation from execution into two distinct stages:

```
┌─────────────────────────────────────────────────────────────┐
│               COMPILE-TIME (Level-1 Pass)                    │
├─────────────────────────────────────────────────────────────┤
│  Dependencies: LLVM, MLIR, onnx-mlir, HIP headers, MIOpen,  │
│                clang (for Runtime bitcode generation)       │
│                                                              │
│  ONNX Model                                                 │
│      ↓                                                       │
│  ONNX → MLIR (onnx-mlir)                                    │
│      ↓                                                       │
│  ONNX dialect → HIP dialect (OnnxToHip Pass)                │
│      ↓                                                       │
│  HIP dialect → LLVM dialect (HipToLLVM Pass)                │
│      ↓                                                       │
│  Generate C interface (GenerateInterfacePass)               │
│      ↓                                                       │
│  LLVM IR → Merge with Runtime bitcode → Optimize (inline)  │
│      ↓                                                       │
│  Optimized IR → Native DLL                                  │
│      ↓                                                       │
│  Embed in EPContext → ONNX model with EPContext             │
│                                                              │
│  Runs once: At model conversion or first load               │
└─────────────────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│                  RUNTIME (Custom Op)                         │
├─────────────────────────────────────────────────────────────┤
│  Dependencies: HIP runtime, MIOpen (+ optional: MemoryModule│
│  or LLVM JIT depending on artifact format)                  │
│                                                              │
│  Load EPContext → Extract artifact → Load and execute       │
│      ↓                                                       │
│  Resolve entry points (inference_init/compute/cleanup)      │
│      ↓                                                       │
│  Execute GPU inference → Return results                     │
│                                                              │
│  Runs repeatedly: Every inference session                   │
└─────────────────────────────────────────────────────────────┘
```

### Architectural Rationale

**Why two stages?**
- Reduce or eliminate JIT compilation overhead
- Potentially remove heavyweight dependencies from deployment
- Enable aggressive compile-time optimizations
- Support WebNN no-disk-access requirements

**Why separate compilation artifacts?**
- Compilation tools (LLVM/MLIR) potentially not needed at inference time
- Potentially smaller deployment footprint
- Potentially faster startup (load vs compile)

### Interface Contract

The compiled DLL exports exactly 3 C functions:
- `int inference_init(void** out_state)` - Allocate GPU resources once
- `int inference_compute(void* state, span_t* inputs, span_t* outputs)` - Execute inference
- `int inference_cleanup(void* state)` - Free GPU resources

See [Design Decision #2](#2-stateful-interface-initcomputecleanup) for rationale.

### Implementation Details

For compilation pipeline details:
- **Overview:** [MLIR-COMPILATION-OVERVIEW.md](MLIR-COMPILATION-OVERVIEW.md) - High-level compilation flow
- **Transformations:** [mlir/LOWERING-PIPELINE.md](mlir/LOWERING-PIPELINE.md) - Detailed pass-by-pass transformations
- **Interface specification:** [mlir/INTERFACE-DESIGN.md](mlir/INTERFACE-DESIGN.md) - Complete C interface design

---

## Key Design Decisions

### 1. Pattern-Based Lowering vs Manual Transformation

**Decision:** Use [MLIR's pattern rewriting framework](https://mlir.llvm.org/docs/DialectConversion/) with typed operations from [onnx-mlir](https://github.com/onnx/onnx-mlir).

**Rationale:**
- MLIR best practice: Standard way to implement dialect conversions
- Type safety: Compile-time checked operation types vs runtime string matching
- Extensible: Adding operations = adding pattern classes
- Productivity: Semantic operand access (getX(), getW()) vs manual operand indexing
- Maintainability: ONNX spec updates handled via onnx-mlir submodule updates

**Trade-offs:**

| Aspect | Pattern-Based | Manual Transform |
|--------|--------------|------------------|
| **Type safety** | Compile-time checked | Runtime crashes possible |
| **Maintainability** | ONNX spec updates via submodule | Manual updates per spec change |
| **Learning curve** | Must learn [MLIR patterns](https://mlir.llvm.org/docs/PatternRewriter/) | Straightforward imperative code |
| **Attribute access** | Semantic getters | Manual string lookup + cast |

---

### 2. Stateful Interface (init/compute/cleanup)

**Decision:** Three-function lifecycle vs single stateless function.

**Rationale:**
- GPU memory allocation is expensive: ~35ms per GB ([HIP Issue #3809](https://github.com/ROCm/hip/issues/3809))
- Allocating per inference would dominate compute time
- Industry best practice: "Allocate once, reuse" ([CUDA](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)/[HIP optimization guides](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html))

**Trade-offs:**

| Aspect | Stateful (init/compute/cleanup) | Stateless (single function) |
|--------|--------------------------------|----------------------------|
| **Performance** | Allocate once, reuse | Allocate per inference |
| **Complexity** | 3 functions, state management | 1 function, simple |
| **Multi-model** | Each model manages own state | Simple resource sharing |
| **Error handling** | Separate init errors from compute errors | Single error path |

**Interface signature:**
- `int inference_init(void** out_state)` - Create GPU resources once
- `int inference_compute(void* state, span_t* inputs, span_t* outputs)` - Execute inference
- `int inference_cleanup(void* state)` - Free GPU resources

See [INTERFACE-DESIGN.md](mlir/INTERFACE-DESIGN.md) for complete specification.

---

### 3. Embedded Constants vs External Files

**Decision:** Embed weights/biases directly in DLL, not separate files.

**Rationale:**
- Single artifact: [EPContext](https://onnxruntime.ai/docs/execution-providers/EP-Context-Design.html) is final compiled model (recompile to change weights)
- Deployment simplicity: No external dependencies or file paths
- Compiled code controls layout: Can optimize constant organization
- Semantic consistency: Constants are part of compiled code, not configuration
- Testing simplicity: No need to manage external file paths during test setup

**Trade-offs:**

| Aspect | Embedded Constants | External Files |
|--------|-------------------|---------------|
| **DLL size** | Larger (includes weights) | Smaller (code only) |
| **Deployment** | Single ONNX file | ONNX + weight files |
| **Weight updates** | Requires recompilation | Can swap files |
| **Loading** | Constants ready at init | Extra I/O on startup |
| **Testing** | Simple (no file path management) | Requires path configuration |

See [CONSTANT-HANDLING-DESIGN.md](CONSTANT-HANDLING-DESIGN.md) for implementation details.

---

### 4. Synchronous Execution vs Async

**Decision:** `inference_compute()` blocks until GPU work completes.

**Rationale:**
- Matches [ONNX Runtime CustomOp](https://onnxruntime.ai/docs/reference/operators/add-custom-op.html) semantics (synchronous `Compute()`)
- Simpler interface (no separate sync function needed)
- Sufficient for single fused node = entire model execution
- Can add async variant later without breaking existing interface

**Trade-offs:**

| Aspect | Synchronous | Asynchronous |
|--------|------------|--------------|
| **Interface** | 3 functions | 4+ functions (compute + wait/poll) |
| **Complexity** | Simple blocking | Requires state management |
| **ORT integration** | Matches CustomOp semantics | Requires ORT async support |
| **Future-proof** | Can extend later | Hard to simplify later |

---

### 5. Full Model Fusion vs Per-Op Execution

**Decision:** Entire ONNX model graph fused into single [CustomOp](https://onnxruntime.ai/docs/reference/operators/add-custom-op.html) node.

**Rationale:**
- Keep intermediates on GPU: No CPU round-trips between operations
- Simplifies interface: Only model inputs/outputs cross CPU-GPU boundary
- Enables global optimizations: Can fuse across operation boundaries
- Matches [EPContext](https://onnxruntime.ai/docs/execution-providers/EP-Context-Design.html) design pattern: One compilation unit per subgraph

**Trade-offs:**

| Aspect | Full Model Fusion | Per-Op CustomOps |
|--------|------------------|------------------|
| **GPU efficiency** | Intermediates never leave GPU | CPU-GPU copy per operation |
| **Compilation** | All-or-nothing (one op fails = all fails) | Gradual fallback to CPU |
| **Debugging** | Harder to isolate failures | Easier per-op debugging |
| **Flexibility** | Cannot mix CustomOp + native ORT ops | Can fallback individual ops |

---

### 6. Standalone Resources vs Shared Context

**Decision:** Each compiled model creates its own GPU handles (hipStream, miopenHandle, etc.).

**Rationale:**
- **Zero CustomOp dependencies:** CustomOp has no HIP headers, backend-agnostic interface
- **Extensibility:** Adding libraries (rocFFT, rocRAND) doesn't change interface
- **Portability:** Same CustomOp code works with different GPU backends
- **Architectural cleanliness:** Clean separation between runtime and compiled code

**Trade-offs:**

| Aspect | Standalone (Chosen) | Shared Context |
|--------|----------------------|----------------|
| **Memory overhead** | Higher (multiple GPU contexts) | Lower (shared context) |
| **CustomOp deps** | Zero GPU headers | Requires HIP headers |
| **Portability** | Backend-agnostic | HIP-specific interface |
| **Extensibility** | Interface stable (add libraries = no change) | Interface grows with libraries |
| **GPU utilization** | More memory used | Better memory efficiency |

See [MEMORY-MANAGEMENT.md](MEMORY-MANAGEMENT.md) for detailed memory allocation strategy.

---

### 7. LLVM IR Merging for Zero-Cost Runtime Abstraction

**Decision:** Embed Runtime as LLVM bitcode in EP DLL, merge with generated IR using `llvm::Linker` API at model compilation time.

**Rationale:**
- **Zero-cost abstraction:** Runtime accessor functions (e.g., `hipdnn_ep_get_stream()`) are inlined to single load instructions
- **Clean separation:** Generated code uses abstraction layer without performance penalty
- **Cross-module optimization:** LLVM optimizer can inline across compilation units
- **No runtime overhead:** Function call overhead eliminated during optimization

**Trade-offs:**

| Aspect | IR Merging (Chosen) | Direct Linking |
|--------|---------------------|----------------|
| **Runtime overhead** | Zero (inlined) | 5-10 cycles per call |
| **DLL size** | Slightly larger (embedded .bc) | Smaller |
| **Build complexity** | Moderate (clang + xxd.py) | Simple |
| **Abstraction cost** | Free (optimizer eliminates) | Paid at runtime |
| **Dependencies** | Clang for bitcode generation | None |

**Requirements:**
- Clang compiler with `-emit-llvm` support
- Python 3 for xxd.py embedding script
- LLVM Linker API (`llvm::Linker::linkInModule()`)
- LLVM Bitcode APIs (`llvm::parseBitcodeFile()`)

See [RUNTIME-ARCHITECTURE.md](RUNTIME-ARCHITECTURE.md) for implementation details, build configuration, C++ integration code, and verification procedures.

---

## Design Principles

These principles guided the architectural decisions:

### 1. Zero GPU Backend Dependencies in CustomOp

**Principle:** CustomOp must be backend-agnostic (no HIP/CUDA/SYCL headers).

**Rationale:** Enables future portability, clean separation of concerns

**Application:** All GPU operations abstracted behind `inference_*()` function pointers loaded from DLL

Related decision: [#6 Standalone Resources vs Shared Context](#6-standalone-resources-vs-shared-context)

### 2. Allocate GPU Resources Once

**Principle:** Create GPU handles/memory at initialization, reuse across inferences.

**Rationale:** GPU memory allocation is expensive, would dominate inference time

**Application:** Three-function lifecycle (init allocates, compute reuses, cleanup frees)

Related decision: [#2 Stateful Interface](#2-stateful-interface-initcomputecleanup)

### 3. Type-Safe MLIR Patterns

**Principle:** Use typed [MLIR](https://mlir.llvm.org/) operations, not string matching.

**Rationale:** Catch errors at compile time, improve maintainability

**Application:** [Pattern-based lowering](#3-pattern-based-lowering-vs-manual-transformation) with [onnx-mlir](https://github.com/onnx/onnx-mlir) typed operations

### 4. Compile-Time Known, Runtime Flexible

**Principle:** Ranks static (compile-time), dimensions dynamic (runtime).

**Rationale:** Balance between optimization (static info) and flexibility (dynamic shapes)

**Application:** [MLIR types](https://mlir.llvm.org/docs/Dialects/Builtin/#ranked-tensor-types) encode static ranks with dynamic dimensions (`memref<1x?x?x?xf32>`)

### 5. Single Source of Truth for Constants

**Principle:** Embed constants in DLL, not external files.

**Rationale:** Simplifies deployment, ensures consistency, enables optimizations

**Application:** ONNX Constant nodes lowered to LLVM globals in compiled DLL

Related decision: [#3 Embedded Constants vs External Files](#3-embedded-constants-vs-external-files)

### 6. Zero-Cost Abstractions via IR Merging

**Principle:** Runtime abstraction layer should have zero performance overhead.

**Rationale:** Clean code separation shouldn't sacrifice performance

**Application:** Merge Runtime LLVM bitcode with generated IR, enabling cross-module inlining

Related decision: [#7 LLVM IR Merging for Zero-Cost Runtime Abstraction](#7-llvm-ir-merging-for-zero-cost-runtime-abstraction)

---

## ONNX-MLIR Integration

### Rationale

Use [onnx-mlir](https://github.com/wcy123/onnx-mlir) fork for type-safe ONNX dialect instead of manual string parsing.

**Key Benefits:**

| Benefit | Without onnx-mlir | With onnx-mlir |
|---------|------------------|----------------|
| **Type safety** | String matching: `if (opName == "Conv")` | Typed patterns: `OpConversionPattern<ONNXConvOp>` |
| **Attribute access** | Manual cast + null check | Semantic getters: `getKernelShape()` |
| **Maintainability** | Manual updates per ONNX spec change | Auto-updated via submodule |
| **Error messages** | "Invalid operand 0" | "Conv input X must be 4D tensor" |

### Integration Points

1. **Level-1 Pass:** Parse ONNX → ONNXOps (typed) → HIP dialect
2. **Transform Passes:** [Pattern-based lowering](#3-pattern-based-lowering-vs-manual-transformation) using typed ONNX operations
3. **Shape Inference:** Reuse onnx-mlir's shape inference for dynamic shapes

See [MLIR-COMPILATION-OVERVIEW.md](MLIR-COMPILATION-OVERVIEW.md) for complete pipeline details.

### Non-Integration

**Runtime:** Custom Op has zero MLIR dependencies (only consumes pre-compiled DLL)

---

## Open Architectural Questions

### Native DLL vs LLVM IR Storage

**Status:** Under Evaluation

**Question:** Should EPContext store native machine code (DLL) or LLVM Intermediate Representation (IR)?

**Context:**
- Both approaches achieve the goal of ahead-of-time compilation
- Trade-offs exist between performance, portability, and deployment complexity
- Decision impacts runtime dependencies, startup time, and cross-platform support

**Current Implementation:**
- System architecture diagrams show both possibilities
- Runtime can potentially support either format

**For detailed comparison:** See [NATIVE-VS-IR-COMPARISON.md](alternatives/NATIVE-VS-IR-COMPARISON.md)

**Key Trade-offs:**

| Aspect | Native DLL | LLVM IR |
|--------|-----------|---------|
| **Startup time** | Fast (load from memory) | Slower (JIT compile) |
| **Runtime deps** | Minimal (MemoryModule) | Large (LLVM libraries) |
| **Portability** | GPU arch-specific | Cross-architecture |
| **Storage size** | Larger | Smaller |

**Decision Criteria:**
- Primary goal: Eliminate JIT overhead → favors Native DLL
- Primary goal: Cross-platform portability → favors LLVM IR
- Support both: Hybrid approach possible but adds complexity

---

## References

### Architectural Decisions
- [NATIVE-VS-IR-COMPARISON.md](alternatives/NATIVE-VS-IR-COMPARISON.md) - Detailed comparison of Native DLL vs LLVM IR storage
- [RUNTIME-ARCHITECTURE.md](RUNTIME-ARCHITECTURE.md) - Runtime design and zero-cost abstraction rationale

### Specifications
- [INTERFACE-DESIGN.md](mlir/INTERFACE-DESIGN.md) - Complete C interface specification
- [HIP-DIALECT-DESIGN.md](mlir/HIP-DIALECT-DESIGN.md) - HIP dialect operations and semantics
- [MLIR-COMPILATION-OVERVIEW.md](MLIR-COMPILATION-OVERVIEW.md) - MLIR module structure and lowering

### Implementation Details
- [MEMORY-MANAGEMENT.md](MEMORY-MANAGEMENT.md) - Memory allocation strategy
- [DYNAMIC-SHAPE-DESIGN.md](DYNAMIC-SHAPE-DESIGN.md) - Dynamic shape handling
- [CONSTANT-HANDLING-DESIGN.md](CONSTANT-HANDLING-DESIGN.md) - Constant embedding in DLL

### External References
- [ONNX Runtime](https://onnxruntime.ai/) - AI inference runtime
- [ONNX Runtime EP Context Design](https://onnxruntime.ai/docs/execution-providers/EP-Context-Design.html) - EPContext mechanism
- [ONNX Runtime Custom Operators](https://onnxruntime.ai/docs/reference/operators/add-custom-op.html) - CustomOp API
- [MLIR](https://mlir.llvm.org/) - Multi-Level Intermediate Representation
- [MLIR Dialect Conversion Guide](https://mlir.llvm.org/docs/DialectConversion/) - Pattern rewriting framework
- [MLIR Pattern Rewriter](https://mlir.llvm.org/docs/PatternRewriter/) - Pattern matching API
- [onnx-mlir](https://github.com/onnx/onnx-mlir) - ONNX MLIR dialect (upstream)
- [onnx-mlir fork](https://github.com/wcy123/onnx-mlir) - Windows build fixes
- [MemoryModule](https://github.com/fancycode/MemoryModule) - In-memory DLL loading library
- [AMD ROCm](https://rocm.docs.amd.com/) - AMD GPU computing platform
- [HIP Programming Guide](https://rocm.docs.amd.com/projects/HIP/) - HIP API documentation
- [HIP Performance Guidelines](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html) - Optimization best practices
- [LLVM Linker API](https://llvm.org/doxygen/classllvm_1_1Linker.html) - LLVM module linking interface
- [LLVM Bitcode Format](https://llvm.org/docs/BitCodeFormat.html) - LLVM bitcode specification
