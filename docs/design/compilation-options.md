<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Compilation Options

**Date:** 2026-03-05
**Document Type:** Design
**Status:** Draft
**Related:** [constant-handling-design.md](constant-handling-design.md)

**Source:** [`schemas/compilation_options.fbs`](../../schemas/compilation_options.fbs)

---

## Overview

Compilation options are defined as a [FlatBuffers](https://flatbuffers.dev/) schema in
`schemas/compilation_options.fbs` and generated into `compilation_options_generated.h`
as the `CompilationOptionsT` native struct. They are passed directly to the compiler
via `CompilerDriver::compile()` and through the C API via `hip_compile_with_fs()`
(see `lib/CInterface/CompilerAPI.cpp`).

Callers may pass `NULL` for `options_json` to accept all defaults when using the C API.

---

## Options

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `opt_level` | `int` | `2` | LLVM optimization level (0–3) |
| `verbose` | `bool` | `false` | Enable verbose diagnostic output |
| `constants_file` | `string` | `"constants.bin"` | Filename (relative to the `FileSystem` root) where constant weight data is written during compilation and read back at `inference_init` time. See [constant-handling-design.md](constant-handling-design.md). |
| `output_mode` | `OutputMode` enum | `LLVM_IR` | Per-model artifact format. `LLVM_IR` (default) emits OS-portable LLVM IR (serialized as `.bc`), JIT-loaded in-process by `LlvmIrJit`. `NATIVE` emits a per-OS native `.dll`/`.so` (runtime merged at producer time, linked via `DLLLinker`) loaded via `morphizen::Plugin` (`LoadLibrary`/`dlopen`). See [native-vs-ir-comparison.md](../native-vs-ir-comparison.md). |

The default `LLVM_IR` format produces a single OS-portable `.bc` consumed by
`LlvmIrJit`. The opt-in `NATIVE` format is per-OS and is intended for
internal benchmarking/dev (the signed-DLL policy keeps it out of production).
The EP records the chosen format in the EPContext metadata
(`mlir_metadata::Metadata.artifact_format`) and selects the matching loader at
session creation.

The EP provider option `artifact_format` (`"LLVM_IR"` | `"NATIVE"`) maps to
this field; the `hip-compiler` CLI exposes it as `--mode {LLVM_IR,NATIVE}`.

---

## Usage

### From `hip-compiler` CLI

```sh
hip-compiler --constants-file weights.bin --constants-dir ./out -O 2 model.mlir -o model.bc
```

| CLI flag | Maps to |
|----------|---------|
| `--constants-file <name>` | `constants_file` |
| `-O <level>` | `opt_level` |
| `-v / --verbose` | `verbose` |
| `--mode {LLVM_IR,NATIVE}` | `output_mode` |

`--constants-dir` is CLI-only — it sets the `DiskFileSystem` root directory and
is never embedded in compilation options or the bitcode metadata.

### From `hip-mlir-opt` pipeline

```sh
hip-mlir-opt --onnx-to-hip-pipeline --hip-to-llvm-pipeline='constants-file=weights.bin' model.mlir
```

### From C API

```c
// include/hip/compiler_api.h
CompilerErrorCode hip_compile_with_fs(
    const void*    input_mlir,
    size_t         input_size,
    const char*    output_path,
    const char*    options_json,  // JSON-serialized CompilationOptionsT; NULL = defaults
    CompilerError* error,
    void*          fs);           // morphizen::FileSystem*
```

### Example `options_json`

```json
{
  "opt_level": 0,
  "verbose": true,
  "constants_file": "my_model/weights.bin"
}
```

---

## Related Documents

- **[constant-handling-design.md](constant-handling-design.md)** — How `constants_file` is used during constant extraction and runtime upload
- **[compiler-runtime-contract.md](compiler-runtime-contract.md)** — How `constants_file` is embedded in `__metadata_blob` inside the bitcode
- **[morphizen-ep-integration.md](morphizen-ep-integration.md)** — End-to-end compilation and inference flow
