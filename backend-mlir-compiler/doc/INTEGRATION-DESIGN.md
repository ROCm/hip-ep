<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Backend-MLIR-Compiler Integration Design

**Date:** 2026-02-20
**Document Type:** Design
**Status:** Draft
**Related:** [morphizen-mlir-compiler/ARCHITECTURE.md](../../3rd-party/morphizen/morphizen-mlir-compiler/doc/design/ARCHITECTURE.md), [morphizen-mlir-compiler/RUNTIME-ARCHITECTURE.md](../../3rd-party/morphizen/morphizen-mlir-compiler/doc/design/RUNTIME-ARCHITECTURE.md), [morphizen-mlir-compiler/INTERFACE-DESIGN.md](../../3rd-party/morphizen/morphizen-mlir-compiler/doc/design/mlir/INTERFACE-DESIGN.md)

---

## Overview

MorphiZen graph optimization framework requires a backend to compile ONNX subgraphs into executable artifacts. The backend-mlir-compiler bridges MorphiZen's Level-1 Pass interface with the morphizen-mlir-compiler AOT compilation plugin, enabling zero-JIT-overhead inference through native DLL generation and EPContext embedding.

**100% GPU Offloading Strategy:** Entire ONNX model graph fused into single CustomOp node (see [ARCHITECTURE.md](../../3rd-party/morphizen/morphizen-mlir-compiler/doc/design/ARCHITECTURE.md#5-full-model-fusion-vs-per-op-execution)). All intermediate tensors remain on GPU, eliminating CPU round-trips. **Constraint:** morphizen-mlir-compiler must successfully compile the full graph (all-or-nothing compilation, no partial fallback).

Two integration points: compilation (Level-1 Pass) and execution (CustomOp).

---

## Design

### Integration Architecture

```
Compile Time:
┌──────────┐   MLIR bytecode    ┌─────────────┐  invoke plugin  ┌──────────────┐
│MorphiZen │ ────────────────▶ │ Level-1 Pass│ ──────────────▶ │morphizen-mlir│
└──────────┘   + options        └─────────────┘                  │  -compiler   │
                                      ▲                          └──────────────┘
                                      │                                 │
                                      │ DLL bytes                       │ temp DLL
                                      │ + metadata                      ▼
                                      └──────────────────────────────────
                                      │
                                      ▼
                                 EPContext

Runtime:
┌──────────┐   Artifact bytes   ┌──────────┐   Output tensors   ┌──────────────┐
│EPContext │ ────────────────▶ │ CustomOp │ ────────────────▶ │ ONNX Runtime │
└──────────┘   + metadata        │          │                    └──────────────┘
                                 │ temp DLL │◀── Input tensors
                                 │          │
                                 │ Plugin:  │
                                 │ init     │
                                 │ compute  │
                                 │ cleanup  │
                                 └──────────┘
```

### Module Interfaces

**Level-1 Pass:**
- **Input:**
  - MLIR bytecode (from MorphiZen `graph.save_string()`)
  - Provider options: `artifact_format` ("native" or "llvm_ir"), `optimization_level` (0-3)
- **Output:**
  - Native DLL bytes written to EPContext
  - Metadata JSON (output tensor descriptors) attached to MetaDefProto
- **Responsibility:** Orchestrate compilation pipeline (extract bytecode, invoke plugin, write artifact, build metadata, fuse graph)

**CustomOp:**
- **Input:**
  - EPContext artifact bytes
  - Metadata JSON (from MetaDefProto)
  - ONNX Runtime input tensors
- **Output:**
  - ONNX Runtime output tensors (shapes determined by metadata)
- **Responsibility:** Load DLL from EPContext, manage three-function lifecycle (init/compute/cleanup), marshal tensors to span_t interface

**Metadata Schema (Protobuf):**
- **Structure:**
  - `artifact_filename`: Lookup key for EPContext file
  - `outputs[]`: Array of output descriptors (name, elem_type, rank, shape[])
- **Responsibility:** Enable output tensor allocation without loading DLL

### Data Flow

**Compilation Path:**
```
Graph.save_string() → MLIR bytecode
                           ↓
              morphizen_mlir_compile(bytecode, temp_path, options_json)
                           ↓
                   Temporary DLL file
                           ↓
            Read DLL bytes + Build metadata JSON
                           ↓
         Write to EPContext + Attach to MetaDefProto
```

**Execution Path:**
```
Read EPContext bytes → Write temp DLL → Load as Plugin
                                             ↓
                                    inference_init(&state)
                                             ↓
Marshal ORT tensors → span_t → inference_compute(state, in, out)
                                             ↓
                              Unmarshal span_t → ORT tensors
```

### Integration Flow

**Compilation (Offline):**
1. MorphiZen detects fusible ONNX subgraph
2. Level-1 Pass extracts MLIR bytecode via `graph.save_string()`
3. Level-1 Pass loads morphizen-mlir-compiler plugin dynamically
4. Level-1 Pass invokes `morphizen_mlir_compile(bytecode, temp_path, options_json, error)`
5. Plugin compiles to native DLL at temp_path (see [LOWERING-PIPELINE.md](../../3rd-party/morphizen/morphizen-mlir-compiler/doc/design/mlir/LOWERING-PIPELINE.md))
6. Level-1 Pass reads DLL bytes, builds metadata Protobuf from output tensor shapes
7. Level-1 Pass writes DLL to EPContext, attaches metadata JSON to MetaDefProto
8. Level-1 Pass fuses graph to single CustomOp node referencing EPContext

**Execution (Runtime):**
1. ONNX Runtime encounters CustomOp node, invokes MlirCustomOp constructor
2. CustomOp parses metadata JSON, reads artifact bytes from EPContext
3. CustomOp writes artifact to temporary DLL file, loads as plugin
4. CustomOp calls `inference_init(&state)` to allocate GPU handles and upload constants (see [CONSTANT-HANDLING-DESIGN.md](../../3rd-party/morphizen/morphizen-mlir-compiler/doc/design/CONSTANT-HANDLING-DESIGN.md))
5. Per inference: CustomOp marshals ORT tensors to `span_t`, calls `inference_compute(state, inputs, outputs)`
6. Session end: Destructor calls `inference_cleanup(state)`, unloads plugin, deletes temp DLL

### Tensor Interface

**Interface Types (C ABI):**

```c
struct tensor_t {
  void *data;       // Pointer to tensor buffer
  int64_t *shape;   // Pointer to shape array
  size_t rank;      // Number of dimensions
};

struct span_t {
  tensor_t *data;   // Pointer to tensor array
  size_t count;     // Number of tensors
};
```

**Marshaling Contract:**
- CustomOp extracts raw pointers from ORT tensors
- Input tensors: Copy data/shape pointers to `tensor_t` array
- Output tensors: Allocate via `GetOutput(index, shape)`, extract mutable pointers
- Pass `span_t` containing tensor array to `inference_compute()`
- Lifetime: Stack-allocated per invocation, destroyed after compute returns

### Configuration

**Provider Options:**

| Option | Values | Default | Purpose |
|--------|--------|---------|---------|
| `artifact_format` | "native", "llvm_ir" | "native" | Compilation output format |
| `optimization_level` | "0", "1", "2", "3" | "2" | MLIR optimization passes |

**Environment Variables:**

| Variable | Values | Purpose |
|----------|--------|---------|
| `MORPHIZEN_DEBUG_MLIR_BACKEND` | 0 (off), 1 (basic), 2+ (detailed) | Logging for Level-1 Pass and CustomOp |

**Options JSON Format (passed to plugin):**
```json
{
  "opt_level": 2,
  "output_mode": "OUTPUT_MODE_DLL"
}
```

### Design Decisions

**Stateless Compilation:**
Level-1 Pass maintains no state between compilations. Rationale: simplifies concurrency, no cleanup needed.

**Plugin-Based Compilation:**
morphizen-mlir-compiler loaded dynamically via plugin API. Invoked via C API `morphizen_mlir_compile()`. Rationale: decouples backend from compiler implementation.

**Artifact Format Extensibility:**
Provider option `artifact_format` supports "native" or "llvm_ir" (see [NATIVE-VS-IR-COMPARISON.md](../../3rd-party/morphizen/morphizen-mlir-compiler/doc/design/alternatives/NATIVE-VS-IR-COMPARISON.md)). Current implementation uses native DLL.

**Metadata Separation:**
Metadata stored in MetaDefProto separately from DLL bytes. Rationale: enables output tensor allocation without loading DLL.

**Temporary File Usage:**
Plugin writes DLL to temp file. CustomOp writes EPContext bytes to temp DLL before loading. Rationale: DLL loading requires file on disk.

**Lifecycle Management:**
Destructor sequence: `inference_cleanup(state)` → plugin unload → temp DLL deletion. Rationale: prevents GPU resource leaks.

**Opaque State Pattern:**
Uses opaque `void *state` handle (see [RUNTIME-ARCHITECTURE.md](../../3rd-party/morphizen/morphizen-mlir-compiler/doc/design/RUNTIME-ARCHITECTURE.md)). CustomOp passes state to three-function interface without inspecting contents.

---

## Related Documents

- [morphizen-mlir-compiler/ARCHITECTURE.md](../../3rd-party/morphizen/morphizen-mlir-compiler/doc/design/ARCHITECTURE.md) - 7 major architectural decisions for MLIR compilation
- [morphizen-mlir-compiler/RUNTIME-ARCHITECTURE.md](../../3rd-party/morphizen/morphizen-mlir-compiler/doc/design/RUNTIME-ARCHITECTURE.md) - RuntimeState design and opaque pointer pattern
- [morphizen-mlir-compiler/INTERFACE-DESIGN.md](../../3rd-party/morphizen/morphizen-mlir-compiler/doc/design/mlir/INTERFACE-DESIGN.md) - Three-function interface specification (init/compute/cleanup)
- [morphizen-mlir-compiler/LOWERING-PIPELINE.md](../../3rd-party/morphizen/morphizen-mlir-compiler/doc/design/mlir/LOWERING-PIPELINE.md) - MLIR pass pipeline (OnnxToHip → HipToLLVM → GenerateInterfacePass)
- [morphizen-mlir-compiler/CONSTANT-HANDLING-DESIGN.md](../../3rd-party/morphizen/morphizen-mlir-compiler/doc/design/CONSTANT-HANDLING-DESIGN.md) - Constant registry pattern and GPU upload strategy
- [morphizen-mlir-compiler/NATIVE-VS-IR-COMPARISON.md](../../3rd-party/morphizen/morphizen-mlir-compiler/doc/design/alternatives/NATIVE-VS-IR-COMPARISON.md) - Trade-offs for artifact storage formats
