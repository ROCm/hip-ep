<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Reference: MorphiZen MLIR dump

## Pipeline slice

1. ONNX to MorphiZen graph (import passes)
2. Pass pipeline (`hip-ep/etc/morphizen_config.json`: init, mlir-pass)
3. Level-1 pass `get_mlir_bytecode()` writes `mlir_bytecode_dump.mlir` when `MORPHIZEN_DEBUG_MLIR_BACKEND >= 2`
4. Script renames that file to `<stem>.mlir` beside the ONNX model
5. hip-compiler may fail after the dump

## EP options

- `dump_dir`: `--mlir-dump-dir` (script sets this to the ONNX directory)
- EP registration name in hip-ep builds: **hipgpu**

## Code pointers (hip-ep repo)

- `backend-mlir-compiler/level-1-pass/src/pass_main.cpp`
- `tools/hip-onnx-runner/hip-onnx-runner.cpp` (`--dump-compiler-mlir`, `--mlir-dump-dir`)
