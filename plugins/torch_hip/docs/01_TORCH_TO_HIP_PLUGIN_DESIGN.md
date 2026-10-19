<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Torch → HIP plugin (`torch_hip`): design

## Goal

Compile PyTorch models with hip-ep. A model (or a Torch-MLIR `.mlir` file) is
lowered from Torch-MLIR's **registered** `torch` dialect to the existing `hip.*`
ops; everything after that (shape inference, bufferization, HIP → LLVM, runtime)
is shared with the ONNX path.

## Decisions

| # | Decision |
|---|---|
| D1 | Use Torch-MLIR's registered `torch` dialect (no stub / unknown-dialect ops). |
| D2 | One LLVM: Torch-MLIR is built from hip-ep's own LLVM (rocmlirTriton `external/llvm-project`). |
| D3 | Importer: Torch-MLIR's official Python FX importer, built on the same LLVM source. |
| D4 | Delivered as a static compiler plugin (`docs/design/plugin-interface.md`) until the design is agreed; later moved into `lib/Conversion/TorchToHip/`. |
| D5 | Emit existing `hip.*` ops; HIP op changes are discussed with the owners first. |
| D6 | First functional step: infrastructure + one op (`torch.aten.add.Tensor` → `hip.add`). |

## Enable / disable

The plugin is off by default. Turn it on at configure time (same on Windows and Linux):

```
-DHIPDNN_EP_COMPILER_PLUGIN_PATHS=<hip-ep>/plugins/torch_hip
-DHIPDNN_EP_COMPILER_PLUGINS=torch_hip
```

Turn it off by leaving both out (or setting both empty and re-configuring).
When enabled, the plugin is linked into `hip-compiler`, `hip-mlir-opt`,
`hip-test` and the EP DLL.

## How it plugs in

- `src/plugin_main.cpp` (the only plugin-specific file) registers the pass
  `convert-torch-to-hip` and requests slot `AfterOnnxLoopOutline`: after
  `hip-add-context-arg`, before `convert-onnx-to-hip`.
- The pass first checks whether the module contains `torch` ops. If not (any
  ONNX model), it returns without changes.

## Split into small PRs

Each PR is small, keeps the default build unchanged (plugin off by default) and
has its own test. The next PR starts from `main` after the previous one merged.
The compiler side (A–D) is tested with hand-written Torch `.mlir`, so it does not
wait for the Python importer (F).

| PR | Branch | Content | Test / exit | Depends on |
|---|---|---|---|---|
| **A (this PR)** | `feat/torch-hip-plugin-skeleton` | Plugin skeleton: CMake, registration, pass with the guard only, LIT feature `torch_hip_plugin` | LIT: ONNX IR unchanged; Torch op → `[torch_hip]` error (direct and via `--hipdnn-pipeline`) | – |
| B | `feat/torch-hip-plugin-dialect` | Fetch Torch-MLIR (pinned, no submodules, patches), build its Torch dialect (7 files) against hip-ep's LLVM, register it | LIT: `hip-mlir-opt` parses a hand-written Torch `.mlir` | A |
| C | `feat/torch-hip-plugin-conversion-infra` | Type converter (`!torch.vtensor` → `tensor`, signed → signless), signature conversion, `torch.vtensor.literal` → `hip.constant`, cleanup, leftover check, unsupported-op precheck | LIT: identity model and returned-literal model lower to HIP | B |
| D | `feat/torch-hip-plugin-aten-add` | First op `torch.aten.add.Tensor` → `hip.add` (+ `hip.mul` for `alpha ≠ 1`), scalar and HIP-build helpers | LIT: broadcast, alpha, literal operand, unsupported-op message; `hip-test --validate` smoke | C |
| E | `feat/hip-test-npy-io` | `hip-test --inputs <npy dir> --dump-outputs <dir>` (general tool change) | ONNX model outputs match ORT CPU | – (parallel) |
| F | `feat/torch-hip-plugin-importer` | Python importer on the same LLVM: build script, `import_model.py`, version stamp, `compare.py` | PyTorch Add model matches PyTorch CPU on the GPU | D, E |
| G | `feat/torch-hip-plugin-input-check` | Input checks, all `[torch_hip]` messages, `normalize_mlir.py`, negative tests | each rejected input form prints its message | F |
| later | per item | One PR per new op; external weights file; shared metadata/constant helpers in hip-ep; move into `lib/Conversion/TorchToHip/` | per item | G |

Risk handling: the importer build (F) is tried early as a spike (no PR) in
parallel with A–C. The compiler-side risk (Torch-MLIR patch count) shows up in B.

## PR A scope

- New: `plugins/torch_hip/` (`CMakeLists.txt`, `include/torch_hip/Passes.h`,
  `src/ConvertTorchToHip.cpp`, `src/plugin_main.cpp`, this doc).
- Changed: `test/lit/CMakeLists.txt`, `lit.site.cfg.py.in`, `lit.cfg.py` add the
  `torch_hip_plugin` LIT feature (same pattern as `hip_static_plugins`).
- Tests: `test/lit/Plugin/torch_hip_guard_onnx_noop.mlir`,
  `test/lit/Plugin/torch_hip_guard_torch_unsupported.mlir` (UNSUPPORTED unless
  the plugin is selected).
- Default build: unchanged.
