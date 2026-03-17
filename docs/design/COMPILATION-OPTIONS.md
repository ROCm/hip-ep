<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Compilation Options

**Date:** 2026-03-05
**Document Type:** Design
**Status:** Draft
**Related:** [CONSTANT-HANDLING-DESIGN.md](CONSTANT-HANDLING-DESIGN.md)

**Source:** [`proto/compilation_options.fbs`](../../proto/compilation_options.fbs)

---

## Overview

Compilation options are defined as a [FlatBuffers](https://flatbuffers.dev/) schema in
`proto/compilation_options.fbs` and generated into `compilation_options_generated.h`
as the `CompilationOptionsT` native struct. They are passed directly to the compiler
via `CompilerDriver::compile()` and through the C API via `hip_compile_with_fs()`
(see `lib/CInterface/CompilerAPI.cpp`).

Callers may pass `NULL` for `options_json` to accept all defaults when using the C API.

---

## Options

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `opt_level` | `int` | `2` | LLVM optimization level (0–3) |
| `output_mode` | `OutputMode` | `DLL` | Output format: `DLL` or `LLVM_IR` |
| `verbose` | `bool` | `false` | Enable verbose diagnostic output |
| `constants_file` | `string` | `"constants.bin"` | Filename (relative to the `FileSystem` root) where constant weight data is written during compilation and read back at `inference_init` time. See [CONSTANT-HANDLING-DESIGN.md](CONSTANT-HANDLING-DESIGN.md). |

### `output_mode` values

| Value | Meaning |
|-------|---------|
| `DLL` | Produce a shared library (`.dll` / `.so`) |
| `LLVM_IR` | Emit LLVM IR text (`.ll`) for inspection |

---

## Usage

### From `hip-compiler` CLI

```sh
hip-compiler --mode dll --constants-file weights.bin --constants-dir ./out -O 2 model.mlir -o model.dll
```

| CLI flag | Maps to |
|----------|---------|
| `--mode dll\|ir` | `output_mode` |
| `--constants-file <name>` | `constants_file` |
| `-O <level>` | `opt_level` |
| `-v / --verbose` | `verbose` |

`--constants-dir` is CLI-only — it sets the `DiskFileSystem` root directory and
is never embedded in compilation options or the DLL metadata.

### From `hip-mlir-opt` pipeline

```sh
hip-mlir-opt --pass-pipeline="morphizen-pipeline{constants-file=weights.bin constants-dir=./out opt-level=2}" model.mlir
```

### From C API

```c
// include/hip/CInterface/compiler_api.h
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
  "output_mode": "LLVM_IR",
  "verbose": true,
  "constants_file": "my_model/weights.bin"
}
```

---

## Related Documents

- **[CONSTANT-HANDLING-DESIGN.md](CONSTANT-HANDLING-DESIGN.md)** — How `constants_file` is used during constant extraction and runtime upload
- **[COMPILER-RUNTIME-CONTRACT.md](COMPILER-RUNTIME-CONTRACT.md)** — How `constants_file` is embedded in `__metadata_blob` inside the DLL
- **[MORPHIZEN-EP-INTEGRATION.md](MORPHIZEN-EP-INTEGRATION.md)** — End-to-end compilation and inference flow
