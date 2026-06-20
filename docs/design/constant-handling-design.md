<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Constant Handling Design

**Date:** 2026-03-05
**Document Type:** Design
**Status:** Draft
**Related:** [compiler-runtime-contract.md](compiler-runtime-contract.md), [morphizen-ep-integration.md](morphizen-ep-integration.md), [compilation-options.md](compilation-options.md)

---

## Table of Contents

- [Overview](#overview)
- [Design](#design)
- [Data Flow](#data-flow)
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
the configured constants file (default: `constants.bin`, see
[compilation-options.md](compilation-options.md)) via a `morphizen::FileSystem*`
at compile time. The DLL contains only code; the runtime reads the file once at
`inference_init` time and uploads each constant to the GPU.

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
  ║         morphizen-ep.dll        ║                               ║       hip-compiler.dll           ║
  ║                                 ║                               ║                                  ║
  ║  ┌─────────────────────────┐    ║  hip_compile_with_fs(fs)    ║  fs->create_writer(name)         ║
  ║  │ level-1 pass (HipDnnEP) ├────╫──────────────────────────────▶  → FileWriter → fwrite()        ║
  ║  │ fs = get_file_system()  │    ║                               ║                                  ║
  ║  └─────────────────────────┘    ║                               ╚══════════════════╤═══════════════╝
  ╚═════════════════════════════════╝                                                   │
                                                                              EP tar cache (or
                                                                              DiskFileSystem for
                                                                              CLI / standalone use)
                                                                                        │
Runtime  (hip-compiler.dll NOT loaded)                                                  │
  ╔═════════════════════════════════════════════════════════════════════════════════════╗
  ║                                morphizen-ep.dll                                     ║
  ║                                                                                     ║
  ║  ┌─────────────────────────┐    LlvmIrJit::create(model.bc bytes)                  ║
  ║  │ MyCustomOp (HipDnnEP)   ├──▶ JIT-loaded into the EP DLL's address space          ║
  ║  │ fs = get_file_system()  │       │                                                ║
  ║  └─────────────────────────┘       │ inference_init(fs)                             ║
  ║                                    ▼                                                ║
  ║                              JITted in-memory module                                ║
  ║                                fs->create_reader(name)  ─── reads constants.bin ───▶║──┐
  ║                                  → FileReader                                       ║  │
  ║                                    fread() / mmap()                                 ║  │
  ╚═════════════════════════════════════════════════════════════════════════════════════╝  │
                                                                                           │
                                                                              EP tar cache ◀┘
```

**Compile time**: the `OnnxToHip` pass calls
`fs->create_writer(constants_filename)` to stream raw constant bytes into
storage. `constants_filename` is the configured constants file
(default: `constants.bin`, see [compilation-options.md](compilation-options.md)).

**Runtime**: the generated `inference_init` calls
`fs->create_reader(constants_filename)` to retrieve the bytes.
`FileReader::mmap()` returns a direct pointer when the backend supports it;
`FileReader::fread()` is the streaming fallback.

For the full DLL contract and `FileSystem` interface definition, see
[morphizen-ep-integration.md](morphizen-ep-integration.md).

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
`model_metadata` (see [compiler-runtime-contract.md](compiler-runtime-contract.md))
at code-generation time. The runtime uses these offsets to index directly into
the buffer after a single bulk read.

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

`model_metadata` schema: [compiler-runtime-contract.md](compiler-runtime-contract.md).

### Step 1 — Discover constants and assign index

The `OnnxToHip` pass (`lib/Conversion/OnnxToHip/OnnxToHip.cpp`) walks the ONNX
functions, assigns each `onnx.Constant` op a sequential `globalIndex`
(0, 1, 2, …), and computes its byte size. Layout order in the constants file
may differ from index order (see [constants.bin Format](#constants.bin-format)
above), but `hipdnn.constant_offsets` preserves the mapping.

The pass then writes raw constant bytes to the configured constants file via
`fs`, and emits two module attributes:

- `hipdnn.constant_sizes` — per-constant byte sizes, indexed by `globalIndex`.
- `hipdnn.constant_offsets` — byte offset of each constant in the configured
  constants file, indexed by `globalIndex`.

The constant count is derived from `hipdnn.constant_sizes`; no separate count
attribute is emitted.

### Step 2 — Code generation: `GenerateInterface` pass

The `GenerateInterface` pass (`lib/Dialect/Transforms/GenerateInterface.cpp`) packs
`hipdnn.constant_sizes`, `hipdnn.constant_offsets`, and
`hipdnn.constants_filename` into `model_metadata`
(see [compiler-runtime-contract.md](compiler-runtime-contract.md)). The
constants-relevant fields in `schemas/model_metadata.fbs`:

```
table ConstantInfo {
  size:   int64;   // byte size of this constant
  offset: int64;   // byte offset within the constants file
}

table HipModelMetaInfo {
  constants_filename: string;         // configured constants file name
  constants:          [ConstantInfo]; // one entry per constant, indexed by globalIndex
  // ... (tensor I/O fields omitted — see compiler-runtime-contract.md)
}
```

### Step 3 — Runtime: `inference_init`

Via `model_metadata` embedded in the DLL
(see [compiler-runtime-contract.md](compiler-runtime-contract.md)), the
runtime reads `constants_filename` and per-constant offsets, then:

1. Opens the configured constants file via `fs`.
2. Reads the file size and allocates a single GPU buffer for the entire
   constants blob (`hipMalloc`). Same path on both dGPU and iGPU — the
   one-time copy cost at init is negligible compared to per-inference VRAM
   access savings.
3. Copies the file contents into the GPU buffer in one operation.
4. Sets each per-constant GPU pointer to the stored offset within the buffer —
   no per-constant allocation or copy.

Note: This constants buffer is separate from the working memory pool managed
by `PoolAllocsPass`, which dynamically packs intermediate tensor allocations
during inference (see `lib/Dialect/Transforms/PoolAllocs.cpp`).

### Step 4 — Inference

During inference, generated code retrieves GPU constant pointers by
`globalIndex` — an O(1) lookup into the runtime state.

---

## Related Documents

- [compiler-runtime-contract.md](compiler-runtime-contract.md) — `model_metadata` schema; how `constants_filename`, sizes, and offsets are embedded in `__metadata_blob`
- [morphizen-ep-integration.md](morphizen-ep-integration.md) — DLL contracts, `FileSystem` interface, EP tar cache
- [compilation-options.md](compilation-options.md) — `constants_file` option and CLI flags
