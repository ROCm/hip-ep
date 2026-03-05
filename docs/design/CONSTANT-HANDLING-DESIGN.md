<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Constant Handling Design

**Date:** 2026-03-04
**Document Type:** Design
**Review Status:** Self-Reviewed
**Related:** ARCHITECTURE.md, RUNTIME-ARCHITECTURE.md, COMPILER-RUNTIME-CONTRACT.md, mlir/passes/06-GenerateInterfacePass.md

---

## Table of Contents

- [Overview](#overview)
- [Design](#design)
- [Data Flow](#data-flow)
- [Troubleshooting Mode](#troubleshooting-mode)
- [Open Questions](#open-questions)
- [Related Documents](#related-documents)

---

## Overview

ONNX models contain constant tensors (weights, biases, etc.), represented as
`onnx.Constant` operations in ONNX-MLIR.

**Problem:**

1. **Function signature explosion**: Passing constants as function arguments
   leads to unmaintainable signatures. A ResNet50 with 200 constants would
   require `@main_graph(%input, %output, %w0, %w1, ..., %w199)`. Adding one
   constant changes the signature everywhere.

2. **Repeated GPU uploads**: Without persistent storage, constants must be
   re-uploaded to GPU on every inference. For a 100 MB model this adds 50–100 ms
   overhead per inference call.

3. **DLL size and link time**: Constants embedded in the DLL data section are
   subject to PE/COFF section size limits; large models exceed them. Link time
   also becomes unacceptable when hundreds of MB of constant data are processed
   by the linker.

**Solution:** External-constants-only design. All constant data is written to
the configured constants file (default: [`constants.bin`](COMPILATION-OPTIONS.md))
via a `morphizen::FileSystem*` at compile time. The DLL contains only code; the
runtime reads the file once at `inference_init` time and uploads each constant
to the GPU.

**Result:**

1. **Stable signatures**: `@main_graph(%ctx, %inputs, %outputs)` regardless of
   constant count.
2. **One-time upload**: GPU memory allocated and uploaded once during init.
3. **No DLL size or link-time constraint**: Constants live in a separate file;
   no PE/COFF section limit and no linker processing of constant data.

---

## Design

### Architecture

The constants pipeline has two phases separated by the `FileSystem` abstraction:

```
Compile time
  ╔═════════════════════════════════╗                               ╔══════════════════════════════════╗
  ║         morphizen-ep.dll        ║                               ║       udna-compiler.dll          ║
  ║                                 ║                               ║                                  ║
  ║  ┌─────────────────────────┐    ║  udna_compile_with_fs(fs)    ║  fs->create_writer(name)         ║
  ║  │ level-1 pass (HipDnnEP) ├────╫──────────────────────────────▶  → FileWriter → fwrite()        ║
  ║  │ fs = get_file_system()  │    ║                               ║                                  ║
  ║  └─────────────────────────┘    ║                               ╚══════════════════╤═══════════════╝
  ╚═════════════════════════════════╝                                                   │
                                                                              EP tar cache (or
                                                                              DiskFileSystem for
                                                                              CLI / standalone use)
                                                                                        │
Runtime  (udna-compiler.dll NOT loaded)                                                 │
  ╔═════════════════════════════════╗                               ╔═══════════════════╧══════════════╗
  ║         morphizen-ep.dll        ║                               ║          model.dll               ║
  ║                                 ║                               ║                                  ║
  ║  ┌─────────────────────────┐    ║  inference_init(fs)           ║  fs->create_reader(name)         ║
  ║  │ MyCustomOp (HipDnnEP)   ├────╫──────────────────────────────▶  → FileReader                   ║
  ║  │ fs = get_file_system()  │    ║                               ║     fread() / mmap()             ║
  ║  └─────────────────────────┘    ║                               ╚══════════════════════════════════╝
  ╚═════════════════════════════════╝
```

**Compile time**: the `OnnxToHip` pass calls
`fs->create_writer(constants_filename)` to stream raw constant bytes into
storage. `constants_filename` is the configured constants file
(default: [`constants.bin`](COMPILATION-OPTIONS.md)).

**Runtime**: the generated `inference_init` calls
`fs->create_reader(constants_filename)` to retrieve the bytes.
`FileReader::mmap()` returns a direct pointer when the backend supports it;
`FileReader::fread()` is the streaming fallback.

For the full DLL contract, EP integration diagram, and `FileSystem` interface
definition, see [MORPHIZEN-EP-INTEGRATION.md](MORPHIZEN-EP-INTEGRATION.md).

### constants.bin Format

Chunk-aligned binary layout (default chunk size: 256 bytes; configurable).
`computeOffsets()` sorts constants by ascending size (a convenience; no strong
architectural requirement), then bin-packs small constants that fit within the
remaining space of the current chunk — multiple small constants can share one
chunk — and places each large constant at the next chunk boundary. A large
constant may span multiple chunks. Zero-filled padding bytes fill the gaps:

```
[constant_a (small)][constant_b (small)][padding to chunk boundary]
[constant_c (large — may span multiple chunks ...  ][padding]
...
```

The exact byte offset of each constant is recorded in the
`hipdnn.constant_offsets` module attribute and packed into
[`model_metadata`](COMPILER-RUNTIME-CONTRACT.md) at code-generation time. The
runtime uses these offsets to index directly into the buffer after a single
bulk read.

---

## Data Flow

```
  onnx.Constant ops
        │ globalIndex (0,1,2,…)
        ▼
  ┌─────────────────────────────────────────┐  compile time
  │ Step 1: OnnxToHip                       │
  │  → constants file (via fs)              │
  │  → hipdnn.constant_sizes                │
  │  → hipdnn.constant_offsets              │
  └──────────────────┬──────────────────────┘
                     │ module attributes
                     ▼
  ┌─────────────────────────────────────────┐
  │ Step 2: GenerateInterface               │
  │  → model_metadata (in DLL):            │
  │     constants_filename                  │
  │     constants[i].size, .offset         │
  └──────────────────┬──────────────────────┘
                     │ DLL + fs
                     ▼
  ┌─────────────────────────────────────────┐  runtime
  │ Step 3: inference_init                  │
  │  → reads constants file via fs          │
  │  → gpu_constants[] (offsets into blob)  │
  └──────────────────┬──────────────────────┘
                     │ globalIndex
                     ▼
  Step 4: O(1) GPU pointer lookup
```

`model_metadata` schema: [COMPILER-RUNTIME-CONTRACT.md](COMPILER-RUNTIME-CONTRACT.md).

### Step 1 — Discover constants and assign index

The `OnnxToHip` pass walks the ONNX functions, assigns each `onnx.Constant`
op a sequential `globalIndex` (0, 1, 2, …), and computes its byte size.
Layout order in the constants file may differ from index order (see
[constants.bin Format](#constants.bin-format) above), but
`hipdnn.constant_offsets` preserves the mapping.

The pass then writes raw constant bytes to the configured constants file via
`fs`, and emits two module attributes:

- `hipdnn.constant_sizes` — per-constant byte sizes, indexed by `globalIndex`.
- `hipdnn.constant_offsets` — byte offset of each constant in the configured
  constants file, indexed by `globalIndex`.

The constant count is derived from `hipdnn.constant_sizes`; no separate count
attribute is emitted.

### Step 2 — Code generation: `GenerateInterface` pass

Packs `hipdnn.constant_sizes`, `hipdnn.constant_offsets`, and
`hipdnn.constants_filename` into `model_metadata`
([COMPILER-RUNTIME-CONTRACT.md](COMPILER-RUNTIME-CONTRACT.md)). The
constants-relevant fields in `proto/model_metadata.fbs`:

```
table ConstantInfo {
  size:   int64;   // byte size of this constant
  offset: int64;   // byte offset within the constants file
}

table UdnaModelMetaInfo {
  constants_filename: string;         // configured constants file name
  constants:          [ConstantInfo]; // one entry per constant, indexed by globalIndex
  // ... (tensor I/O fields omitted — see COMPILER-RUNTIME-CONTRACT.md)
}
```

### Step 3 — Runtime: `inference_init`

Via `model_metadata` embedded in the DLL
(see [COMPILER-RUNTIME-CONTRACT.md](COMPILER-RUNTIME-CONTRACT.md)), the
runtime reads `constants_filename` and per-constant offsets, then:

1. Opens the configured constants file via `fs`.
2. Allocates a single GPU buffer for the entire constants data and uploads it
   in one operation. On iGPU, pinned host memory is used instead and the GPU
   reads in-place without a copy.
3. Sets each per-constant GPU pointer to the stored offset within the buffer —
   no per-constant allocation or copy.

The GPU pointers are stored in the opaque [runtime state](RUNTIME-ARCHITECTURE.md).

### Step 4 — Inference

During inference, generated code retrieves GPU constant pointers by
`globalIndex` — an O(1) lookup into the [runtime state](RUNTIME-ARCHITECTURE.md).

---

## Related Documents

**Architecture:**
- [ARCHITECTURE.md](ARCHITECTURE.md) — System architecture and design decisions
- [RUNTIME-ARCHITECTURE.md](RUNTIME-ARCHITECTURE.md) — Runtime bitcode design, opaque RuntimeState pattern

**Implementation:**
- [mlir/passes/06-GenerateInterfacePass.md](mlir/passes/06-GenerateInterfacePass.md) — How `inference_init` is generated
- [mlir/passes/02-OnnxToHip.md](mlir/passes/02-OnnxToHip.md) — Constant discovery and writing
- [mlir/passes/05-HipToLLVM.md](mlir/passes/05-HipToLLVM.md) — `hip.get_constant` lowering

**Related Design:**
- [COMPILER-RUNTIME-CONTRACT.md](COMPILER-RUNTIME-CONTRACT.md) — `proto/model_metadata.fbs` schema; how `constants_filename`, sizes, and offsets are embedded in `__metadata_blob`
- [EPCONTEXT-MEMORY-OPTIMIZATION.md](EPCONTEXT-MEMORY-OPTIMIZATION.md) — Future DLL loading memory optimisations
