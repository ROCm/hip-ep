<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Backend-MLIR-Compiler Integration Design

**Date:** 2026-02-20
**Document Type:** Design
**Status:** Draft
**Related:** Design documents formerly in `morphizen-mlir-compiler/doc/design/` (now merged into this project)

---

## Overview

MorphiZen graph optimization framework requires a backend to compile ONNX subgraphs into executable artifacts. The backend-mlir-compiler bridges MorphiZen's Level-1 Pass interface with the hip-compiler AOT compilation plugin: hip-compiler emits OS-portable LLVM bitcode (no triple, no datalayout, no MSVC/glibc symbols), which the EP DLL JIT-loads in-process via `LlvmIrJit` at session creation. There is no per-model DLL written to disk and no `LoadLibrary` call at inference time — required by Microsoft's signed-DLL-only loading policy.

**100% GPU Offloading Strategy:** Entire ONNX model graph fused into single CustomOp node. All intermediate tensors remain on GPU, eliminating CPU round-trips. **Constraint:** the hip-compiler plugin must successfully compile the full graph (all-or-nothing compilation, no partial fallback).

Two integration points: compilation (Level-1 Pass) and execution (CustomOp).

---

## Design

### Integration Architecture

```
Compile Time:
┌──────────┐   MLIR bytecode    ┌─────────────┐  invoke plugin  ┌──────────────┐
│MorphiZen │ ────────────────▶ │ Level-1 Pass│ ──────────────▶ │ hip-compiler │
└──────────┘   + options        └─────────────┘                  │   plugin     │
                                      ▲                          └──────────────┘
                                      │                                 │
                                      │ bitcode bytes                   │ writes
                                      │ + metadata                      │ model_compiled
                                      └─────────────────────────────────▼
                                      │                          (per-model .bc)
                                      ▼
                                 EPContext

Runtime:
┌──────────┐   Bitcode bytes    ┌──────────┐   Output tensors   ┌──────────────┐
│EPContext │ ────────────────▶ │ CustomOp │ ────────────────▶ │ ONNX Runtime │
└──────────┘   + metadata        │          │                    └──────────────┘
                                 │ Bitcode  │◀── Input tensors
                                 │   JIT    │
                                 │ (LLJIT,  │
                                 │  in-proc)│
                                 │ Symbols: │
                                 │  init    │
                                 │  compute │
                                 │  cleanup │
                                 └──────────┘
```

### Module Interfaces

**Level-1 Pass:**
- **Input:**
  - MLIR bytecode (from MorphiZen `graph.save_string()`)
  - Provider options: `optimization_level` (0-3) and `artifact_format` — `"LLVM_IR"` (default) emits OS-portable LLVM bitcode JIT-loaded in-process by `LlvmIrJit`; `"NATIVE"` emits a per-OS `.dll`/`.so` loaded via `morphizen::Plugin`. Unknown values are coerced to `"LLVM_IR"` (see `pass_main.cpp::load_config`).
- **Output:**
  - Per-model LLVM bitcode bytes written to EPContext
  - Metadata JSON (output tensor descriptors) attached to MetaDefProto
- **Responsibility:** Orchestrate compilation pipeline (extract bytecode, invoke plugin, write artifact, build metadata, fuse graph)

**CustomOp:**
- **Input:**
  - EPContext artifact bytes (per-model `.bc`)
  - Metadata JSON (from MetaDefProto)
  - ONNX Runtime input tensors
- **Output:**
  - ONNX Runtime output tensors (shapes determined by metadata)
- **Responsibility:** Hand bitcode bytes to `LlvmIrJit::create` (in-process LLJIT load — no temp file, no `LoadLibrary`), manage four-symbol lifecycle (init/compute/cleanup + optional begin_compute), marshal tensors to span_t interface

**Metadata Schema (Protobuf):**
- **Structure:**
  - `artifact_filename`: Lookup key for the EPContext entry holding the per-model bitcode (typically `"model_compiled"`)
  - `outputs[]`: Array of output descriptors (name, elem_type, rank, shape[])
- **Responsibility:** Enable output tensor allocation without paying JIT codegen cost

### Data Flow

**Compilation Path:**
```
Graph.save_string() → MLIR bytecode
                           ↓
              hip_compile_with_fs(bytecode, temp_path, options_json, fs)
                           ↓
              Per-model LLVM bitcode (.bc, OS-portable)
                           ↓
              Read bitcode bytes + Build metadata JSON
                           ↓
              Write to EPContext + Attach to MetaDefProto
```

**Execution Path:**
```
Read EPContext bytes (per-model .bc) → LlvmIrJit::create
                                              ↓
                              parseBitcodeFile(model.bc + runtime.bc)
                                       in one LLJIT JITDylib
                                              ↓
                                       inference_init(&state, fs)
                                              ↓
Marshal ORT tensors → span_t → inference_compute(state, in, out)
                                              ↓
                              Unmarshal span_t → ORT tensors
```

### Integration Flow

**Compilation (Offline):**
1. MorphiZen detects fusible ONNX subgraph
2. Level-1 Pass extracts MLIR bytecode via `graph.save_string()`
3. Level-1 Pass loads hip-compiler plugin dynamically
4. Level-1 Pass invokes `hip_compile_with_fs(bytecode, temp_path, options_json, error, fs)`
5. Plugin compiles to per-model LLVM bitcode at temp_path; producer strips triple + datalayout to keep the artifact OS-portable
6. Level-1 Pass reads bitcode bytes, builds metadata Protobuf from output tensor shapes
7. Level-1 Pass writes the bitcode to EPContext (under filename `model_compiled`), attaches metadata JSON to MetaDefProto
8. Level-1 Pass fuses graph to single CustomOp node referencing EPContext

**Execution (Runtime):**
1. ONNX Runtime encounters CustomOp node, invokes MlirCustomOp constructor
2. CustomOp parses metadata JSON, reads artifact bytes from EPContext
3. CustomOp hands the bitcode to `LlvmIrJit::create`, which adds `model_compiled` and the EP-DLL-embedded `runtime.bc` into a single LLJIT JITDylib (one shared `LLVMContext`); ORC's `IRCompileLayer` codegens lazily on first symbol lookup
4. CustomOp resolves and calls `inference_init(&state, fs)` to allocate GPU handles and upload constants
5. Per inference: CustomOp marshals ORT tensors to `span_t`, calls `inference_compute(state, inputs, outputs)`
6. Session end: Destructor calls `inference_cleanup(state)`, then `~LlvmIrJit()` runs JIT-registered atexit/`__cxa_atexit` handlers and tears down LLJIT

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
| `artifact_format` | `"LLVM_IR"` \| `"NATIVE"` (unknown coerced to `LLVM_IR`) | "LLVM_IR" | Per-model artifact format: `LLVM_IR` = OS-portable LLVM bitcode JIT-loaded by `LlvmIrJit`; `NATIVE` = per-OS `.dll`/`.so` loaded via `morphizen::Plugin` (opt-in, benchmarking/dev). |
| `optimization_level` | "0", "1", "2", "3" | "2" | LLVM optimization level (target-independent PerModule pipeline) |

**Environment Variables:**

| Variable | Values | Purpose |
|----------|--------|---------|
| `MORPHIZEN_DEBUG_MLIR_BACKEND` | 0 (off), 1 (basic), 2+ (detailed) | Logging for Level-1 Pass and CustomOp |

**Options JSON Format (passed to plugin):**
```json
{
  "opt_level": 2
}
```

### Design Decisions

**Stateless Compilation:**
Level-1 Pass maintains no state between compilations. Rationale: simplifies concurrency, no cleanup needed.

**Plugin-Based Compilation:**
hip-compiler loaded dynamically via morphizen plugin API. Invoked via C API `hip_compile_with_fs()`. Rationale: decouples backend from compiler implementation.

**Artifact Format (LLVM IR default, native opt-in):**
By default the compiler emits binary LLVM bitcode, which the EP DLL JITs in-process via `LlvmIrJit` (`backend-mlir-compiler/custom-op-mlir/src/LlvmIrJit.h`). Rationale: meets Microsoft's security requirement that only signed DLLs may be loaded at inference time — there is no per-model DLL written to disk and no `LoadLibrary` call. The custom HIP kernels are folded into the single signed EP DLL. A `NATIVE` `.dll`/`.so` format is available opt-in for benchmarking/dev (loaded via `morphizen::Plugin`).

**Metadata Separation:**
Metadata stored in MetaDefProto separately from the bitcode bytes. Rationale: enables output tensor allocation without paying JIT codegen cost.

**In-process JIT Loading:**
CustomOp hands bitcode bytes directly to ORC LLJIT — no temp files, no `LoadLibrary`, nothing that violates Microsoft's signed-DLL-only loading requirement. The 5-symbol contract (`inference_init`, `inference_compute`, `inference_cleanup`, `inference_get_metadata_json`, `inference_runtime_begin_compute`) is resolved from the JITted module.

**No `thread_local` in the JIT'd runtime (emulated-TLS avoidance):**
`runtime.bc` must not define a `thread_local`. The ORC JIT lowers any thread-local in a JIT'd module to *emulated* TLS — a call to `__emutls_get_address` (a compiler-rt symbol) that has no definition inside the JIT process, so the module fails to materialize at load. Native TLS would avoid the emutls call, but the ORC JIT's native-TLS support is unimplemented on Windows/COFF (it exists only on macOS and, partially, Linux). The native `.dll`/`.so` path is unaffected — `lld-link` lowers `thread_local` to OS-managed native TLS like any normal DLL. The one per-thread slot the runtime needs (the current session stream) therefore lives in natively-compiled host code (`lib/Runtime/tls_stream.cpp`, linked into the EP/tools and into native model DLLs): the JIT'd runtime only *calls* `hipdnn_ep_get/set_current_stream` as ordinary external functions, which `LlvmIrJit` resolves as absolute symbols. See `LlvmIrJit.cpp` and `tls_stream.cpp`.

**Lifecycle Management:**
Destructor sequence: `inference_cleanup(state)` → `LlvmIrJit` destructor (tears down `LLJIT`). Rationale: prevents GPU resource leaks; no temp files to remove.

**Opaque State Pattern:**
Uses opaque `void *state` handle. CustomOp passes state to three-function interface without inspecting contents.

---

## Related Documents

Design documents formerly maintained in `morphizen-mlir-compiler/doc/design/` have been merged
into this project (`backend-mlir-compiler/`). Key topics covered:

- MLIR compilation architecture and major design decisions
- RuntimeState design and opaque pointer pattern
- Five-symbol interface specification (init/compute/cleanup + inference_get_metadata_json + optional begin_compute)
- MLIR pass pipeline (OnnxToHip -> HipToLLVM -> GenerateInterfacePass)
- Constant registry pattern and GPU upload strategy
- Trade-offs for artifact storage formats (per-model bitcode vs textual LLVM IR; per-OS runtime.bc on the consumer side)
