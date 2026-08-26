<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# MorphiZen EP Integration

**Date:** 2026-03-05
**Document Type:** Design
**Status:** Draft
**Related:** [constant-handling-design.md](constant-handling-design.md), [compiler-runtime-contract.md](compiler-runtime-contract.md), [compilation-options.md](compilation-options.md)

---

## Table of Contents

- [Overview](#overview)
- [Contract 1: morphizen-ep.dll → hip-compiler](#contract-1-morphizen-epdll--hip-compiler)
- [Contract 2: morphizen-ep.dll → model.dll](#contract-2-morphizen-epdll--modeldll)
- [FileSystem: The Common Thread](#filesystem-the-common-thread)
- [Boundary Contracts](#boundary-contracts)
- [Two-Channel Metadata Architecture](#two-channel-metadata-architecture)
- [Related Documents](#related-documents)

---

## Overview

The system spans two phases:

| Phase | Path |
|-------|------|
| Compile time (session creation) | `onnxruntime.dll` → `morphizen-ep.dll` → hip-compiler |
| Runtime (inference) | `onnxruntime.dll` → `morphizen-ep.dll` → `model.dll` |

The compiler is **not invoked at inference time**. `model.dll` is **not
loaded at compile time**. The EP tar cache — accessed through the `FileSystem`
abstraction — is the only persistent link between the two phases.

**Where hip-compiler lives.** The default build (`BUILD_EP=ON`) links the
compiler statically into `morphizen-ep.dll` and reaches it through MorphiZen's
in-process static plugin registry; no `hip-compiler.dll` is produced. A
tools-only build (`BUILD_EP=OFF, BUILD_HIP_TOOLS=ON`) still builds the
standalone `hip-compiler.dll` / `libhip-compiler.so`, which the same
`morphizen::Plugin` call loads via `LoadLibrary`/`dlopen`. The contract below is
identical either way, so the diagrams keep the compiler in its own box.

`morphizen-ep.dll` integrates with `onnxruntime.dll` via the standard
[ORT Plugin EP v2 API](https://onnxruntime.ai/docs/execution-providers/plugin-ep-libraries.html).
The two contracts specific to this project are described below.

**Compile time (session creation):**

```
  ╔══════════════════════╗
  ║   onnxruntime.dll    ║
  ║   GetCapability()    ║
  ║   Compile()          ║
  ╚══════════╤═══════════╝
             │ Compile callback
             ▼
  ╔═════════════════════════════════╗
  ║         morphizen-ep.dll        ║
  ║                                 ║
  ║  ┌─────────────────────────┐    ║
  ║  │ level-1 pass (HipDnnEP) │    ║
  ║  │ OrtGraph* → IRConverter │    ║
  ║  │        → input_mlir     │    ║
  ║  │ fs = get_file_system()  │    ║
  ║  └─────────────────────────┘    ║
  ╚═════════════════╤═══════════════╝
                    │ hip_compile_with_fs(input_mlir, .., fs)
                    ▼
  ╔═════════════════════════════════╗
  ║      hip-compiler (plugin)      ║
  ║  fs->create_writer(name)        ║
  ║  → FileWriter → fwrite()        ║
  ╚══════════════════╤══════════════╝
                     │
               EP tar cache
```

**Runtime (inference) — the compiler is not involved:**

```
  ╔══════════════════════╗
  ║   onnxruntime.dll    ║
  ║   CreateState()      ║
  ║   Compute()          ║
  ║   ReleaseState()     ║
  ╚══════════╤═══════════╝
             │ CreateState / Compute / ReleaseState callbacks
             ▼
  ╔═════════════════════════════════╗
  ║         morphizen-ep.dll        ║
  ║                                 ║
  ║  ┌─────────────────────────┐    ║
  ║  │   MyCustomOp (HipDnnEP) │    ║
  ║  │ fs = get_file_system()  │    ║
  ║  └─────────────────────────┘    ║
  ╚═════════════════╤═══════════════╝
                    │ inference_init(fs)
                    │ hipdnn_ep_runtime_begin_compute(state)  [optional]
                    │ inference_compute(state, inputs, outputs)
                    │ inference_cleanup(state)
                    ▼
  ╔═════════════════════════════════╗
  ║           model.dll             ║
  ║  fs->create_reader(name)        ║  ← reads from EP tar cache
  ║  → FileReader                   ║     (written at compile time)
  ║     fread() / mmap()            ║
  ╚═════════════════════════════════╝
```

---

## Contract 1: morphizen-ep.dll → hip-compiler

During `GetCapability`, the level-1 pass first converts the `OrtGraph` object
to MLIR IR bytes:

```
OrtGraph* → IRConverter::to_onnx_model()  →  MLIR bytes (input_mlir)
```

It then calls into the compiler via a single C function:

```c
// include/hip/compiler_api.h
CompilerErrorCode hip_compile_with_fs(
    const void*    input_mlir,   // MLIR IR bytes (from IRConverter)
    size_t         input_size,
    const char*    output_path,  // logical name for the compiled artifact in the cache
    const char*    options_json, // JSON-serialized CompilationOptionsT; NULL = defaults
    CompilerError* error,
    void*          fs);          // morphizen::FileSystem* — storage backend
```

The symbol is resolved through `morphizen::Plugin::get("hip-compiler")`, which
searches the in-process static registry before falling back to
`LoadLibrary`/`dlopen`.

`morphizen-ep.dll` has no knowledge of how the compiler compiles the model. Any
compiler that implements this API and registers under the same plugin id can
replace it without changes to the EP.

---

## Contract 2: morphizen-ep.dll → model.dll

`model.dll` is generated by the compiler and exports the following C functions:

```c
int  inference_init(void** out_state, void* fs);
int  inference_compute(void* state, span_t* inputs, span_t* outputs);
int  inference_cleanup(void* state);
void hipdnn_ep_runtime_begin_compute(void* state);   // optional, see below
```

`inference_init` receives the same `FileSystem*` that the level-1 pass used
during compilation, giving `model.dll` access to the EP cache.
`inference_compute` is called once per inference run. `inference_cleanup` is
called at session destruction.

`hipdnn_ep_runtime_begin_compute` is called by the EP at the top of every
`Compute()` (before input marshaling) to invalidate per-`Compute()` runtime
caches — currently only the GQA `seqlens_k` D2H cache, which amortizes one
device read across all GQA layers in a single forward pass. The export is
**optional** for backward compatibility: `InferenceState::create()` resolves
the symbol once via `get_method`, leaves the function pointer null when the
symbol is absent, and `InferenceState::begin_compute()` becomes a no-op. Old
DLLs detected at session creation emit a `LOG(WARNING)` and require
`HIPDNN_EP_GQA_CACHE_SEQLENS=0` to avoid stale-cache reads across forward
passes (see CLAUDE.md "GQA seqlens_k caching across layers in one Compute()").

`morphizen-ep.dll` has no knowledge of what `model.dll` uses internally. The
compiler has full freedom in what it generates inside `model.dll` — target
hardware, libraries, or compute strategy — without any modification to the EP.

---

## FileSystem: The Common Thread

`morphizen::FileSystem` (defined in `morphizen-foundation/file_io.hpp`) is the
only data channel that crosses both DLL boundaries:

```
PassContext::get_file_system()
        │
        │  fs passed to hip-compiler at compile time
        │  fs passed to model.dll at runtime
        ▼
  ┌─────────────────────────┐
  │      EP tar cache        │  ← storage backend, opaque to compiler and model
  │  (or DiskFileSystem for  │
  │   CLI / standalone use)  │
  └─────────────────────────┘
```

| Interface    | Key methods                   | Used by             |
|--------------|-------------------------------|---------------------|
| `FileSystem` | `create_writer(path)`         | hip-compiler        |
| `FileSystem` | `create_reader(path)`         | `model.dll`         |
| `FileWriter` | `fwrite(buf, size)`           | hip-compiler        |
| `FileReader` | `fread(buf, size)`, `mmap()`  | `model.dll`         |

`DiskFileSystem` (`include/hip/Support/DiskFileSystem.h`) is the concrete
implementation used by the CLI tools (`hip-compiler`, `hip-mlir-opt`).

---

## Boundary Contracts

The compiler and EP live in the same repository (hip-ep), and in the default
build in the same binary, but they communicate only through minimal C APIs:

- **hip-compiler** and **`morphizen-ep.dll`** are coupled only through
  `hip_compile_with_fs()`. The compiler implementation — passes, optimisations,
  code generation strategy — can change without modifying the EP interface.

- **`model.dll`** (generated at compile time) and the EP are coupled only through
  the three C function signatures and the `FileSystem*` channel. The compiler has
  full freedom in what it generates inside `model.dll` — target hardware,
  libraries, or compute strategy.

- **Storage backend** (`FileSystem`) is opaque to both the compiler and
  `model.dll`. The EP decides where artifacts live — in-memory tar cache,
  disk, or a network store — without either side needing to know.

---

## Two-Channel Metadata Architecture

The `model_metadata` that travels alongside a compiled artifact has **two
distinct representations** that look identical at the C-interface level
but differ in what they actually carry:

| Channel | Owner | Format | Visible to |
|---------|-------|--------|------------|
| **A** (in-process) | `morphizen-ep.dll` level-1 pass | ProtoBuf message in MlirCustomOp `MetaDef` | session-creation path only |
| **B** (on-disk in DLL) | hip-compiler GenerateInterface | FlatBuffer in the compiled `model.dll`'s constants blob | runtime path; `hip-inspect` |

**Channel A** is constructed at session creation as a ProtoBuf message
holding the OrtGraph→MLIR metadata (input/output names, ranks, element
types, and static shapes; dynamic dims carried as `-1`). The level-1
pass attaches Channel A to the in-process MlirCustomOp's `MetaDef`.

**Channel B** is generated by the compiler as a FlatBuffer that
survives into the compiled `model.dll`. It carries the same shape
information used at runtime by `MlirCustomOp::Compute()`.

**Dynamic output dims** are not resolved from session-creation metadata.
The DLL sizes each dynamic output **in-graph at runtime**: the EP's
`output_allocate_cb` receives the shape the DLL computed and passes it to
`GetOutput` verbatim, so no per-dim metadata is needed.

**Why two channels.** The FlatBuffer format avoids a ProtoBuf dependency
in `model.dll`, which is loaded into the inferencing process and must
stay minimal. The ProtoBuf version stays in-process where the
compile-time machinery is already loaded.

---

## Related Documents

- [constant-handling-design.md](constant-handling-design.md) — constants file format, write order, and runtime upload
- [compiler-runtime-contract.md](compiler-runtime-contract.md) — `model_metadata` schema; how metadata is embedded in `model.dll`
- [compilation-options.md](compilation-options.md) — `CompilationOptionsT` fields and CLI mapping
