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
  ║         morphizen-ep.dll        ║                               ║       hip-compiler (plugin)      ║
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
Runtime  (the compiler is not involved)                                                 │
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

**Compile time**: `convert-onnx-to-hip` runs semantic ONNX rewrites first, then
its two constant sweeps lower `onnx.Constant` to the neutral `hip.constant`
carrier. Ordinary compute conversion runs between the sweeps and can inspect a
dense carrier payload directly. A carrier contains an inline dense `value`, a
complete file `location`/`offset`/`size`, or a complete
`memory_address`/`size`. The ONNX converter translates the ORT memory sentinel
into the generic memory form; no ORT encoding enters the HIP dialect.

After the `AfterConvertOnnxToHip` plugin slot, `hip-infer-shapes` consumes any
inspectable carrier payload needed for compile-time shape refinement. The
standalone `hip-externalize-constants` pass then validates and plans all
carriers, writes the artifacts, and commits the IR and metadata. Plugins may
emit `hip.constant` at that slot and receive the same policy as in-tree
constants.
The externalizer calls
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

Constants remain in declaration order. Each constant begins at the next
64-byte boundary; zero-filled bytes occupy alignment gaps. The file ends at the
end of the final constant (there is no mandatory trailing 64-byte pad):

```
[constant_a][zero padding to 64-byte boundary]
[constant_b][zero padding to 64-byte boundary]
[constant_c]
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
  │ Step 1a: convert-onnx-to-hip            │
  │  → hip.constant carriers                │
  │ Step 1b: hip-infer-shapes               │
  │  → consume compile-time shape payloads  │
  │ Step 1c: hip-externalize-constants      │
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

### Step 1 — Produce carriers, then externalize

`convert-onnx-to-hip`
(`lib/Conversion/OnnxToHip/OnnxToHip.cpp`) lowers each `onnx.Constant` to
`hip.constant` after the semantic Gather/Reshape, activation/projector, and
GatherBlockQuantized rewrites have had access to generic ONNX structure.
Pad, Slice, Tile, Expand, ConstantOfShape, and OneHot consume dense carrier
payloads during normal compute conversion; they do not require ONNX stamping
attributes or constant-preservation prepasses. Conversion does no filesystem
access, layout, index assignment, or global creation.

`hip-externalize-constants`
(`lib/Dialect/Transforms/ExternalizeConstants.cpp`) validates each explicit
`serialization_order`, then assigns every externalized carrier a sequential
`globalIndex` (0, 1, 2, …). The ONNX converter's absolute counter preserves its
original visitation across functions: function 0 imported sweep, function 0
synthesized sweep, function 1 imported sweep, function 1 synthesized sweep, and
so on. Unordered plugin carriers follow in stable module walk order. Inline
values below the threshold become `arith.constant`; larger values and external
references become the existing extern `memref.global` bridge. The direct pass
threshold defaults to `0` (inline); production compilation passes the default
`1`, so every non-empty production constant, including a scalar, is
externalized. External sources require a positive byte size, matching the
pre-split behavior.

The ORT memory-address sentinel is handled only by the ONNX converter. It
becomes the generic carrier `memory_address`, which remains process-local and
non-serializable. ORT retains initializer storage through compilation, and the
production pass receives an injected `FileSystem` while that storage is live.
Direct/textual pass invocation rejects memory-address carriers before
dereferencing them. Keeping the handoff zero-copy avoids duplicating
potentially multi-gigabyte weights throughout shape inference.

For externalized constants, the pass writes raw bytes to the configured
constants file via `fs`, and emits the existing module attributes:

- `hipdnn.constant_sizes` — per-constant byte sizes, indexed by `globalIndex`.
- `hipdnn.constant_offsets` — byte offset of each constant in the configured
  constants file, indexed by `globalIndex`.

The constant count is derived from `hipdnn.constant_sizes`; no separate count
attribute is emitted.

With `skip-constant-data`, the current streaming/hybrid contract is preserved:
splat and file references stay in the existing parallel source arrays, while
live memory-address and non-splat inline bytes are packed into a compact
64-byte-aligned partial constants file. The runtime FlatBuffer schema and
`MemSource` contract are unchanged. Pure streaming and hybrid compilation do
not open file-reference sources because those bytes are read later by the
runtime. Full-sidecar and threshold-zero inline modes validate/read file ranges
because compilation actually consumes the bytes.

Threshold-zero inline materialization preserves the carrier's exact tensor
type, including `ui8` and `si8`. Byte-aligned element types use
`DenseElementsAttr::getFromRawBuffer` with that exact tensor type, preserving
MLIR/ONNX raw-storage interpretation without host-endian integer decoding.
External `i1` is the sole normalization case because the runtime contract uses
one byte per element while DenseElementsAttr bit-packs it. Type agreement is
checked before any IR mutation.

The implementation validates carriers, explicit serialization order, source ranges
that compilation reads, layout arithmetic, metadata freshness, and symbol
availability before serialization, then mutates IR only after serialization
succeeds. The shared `ConstantsIO` contract intentionally remains unchanged:
it does not surface `FileWriter::fwrite` short writes. `FileSystem` also has no
atomic rename/remove transaction, so a later failed write can leave a partial
artifact even though the IR remains unchanged.

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
