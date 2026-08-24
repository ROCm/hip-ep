<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# gpuep-rel-2610.2 gpu-test-package

Vendored Windows GPU test package for gpuep release validation. Layout matches CI `gpu-test-package` (`bin/`, `lib/`, `wheels/`, root `morphizen_config.json`).

## Sources

| Component | Source |
|---|---|
| EP / ROCm / ORT runtime DLLs | [Artifactory modelbench-ort gpuep-rel-2610.2](https://mkmartifactory.amd.com/artifactory/SW-UIF2.0-DEV-LOCAL/UAI/packaging/Pre-Prod/modelbench-ort/gpuep-releases/gpuep-rel-2610.2/3fc3e050338ca5ec592b8da0b0ce382a064d521a/56/3fc3e05_modelbench-ort_56.zip) (commit `3fc3e050338ca5ec592b8da0b0ce382a064d521a`, build 56) |
| Test executables (8), `lib/`, `wheels/`, `bin/rocblas/` | [CI gpu-test-package](https://github.com/ROCm/hip-ep/actions/runs/32719961059/artifacts/9518622988) (main, run `32719961059`, artifact `9518622988`) |
| morphizen_config.json | `etc/morphizen_config.json` from hip-ep main |

## Supplemental from CI

**Test executables** (Artifactory Release had only `ModelBench-Ort.exe` and `migraphx-hiprtc-driver.exe`):

- `onnxruntime_perf_test.exe`
- `model_benchmark.exe`
- `model_mm.exe`
- `hip-onnx-runner.exe`
- `hip-test.exe`
- `hip-inspect.exe`
- `hip-mlir-opt.exe`
- `hip-compiler.exe`

**`lib/` and `wheels/`** — required by Ryzen-AI-CI `verify_package` for `daily_hip_ep_genai`. Also keeps multiple top-level directories so Jenkins artifact extraction preserves `bin/` (single-dir packages were normalized incorrectly).

**`bin/rocblas/`** — missing from Artifactory; copied from CI for rocblas runtime data.

## Package

| File | SHA256 |
|---|---|
| `gpuep-rel-2610.2-gpu-test-package.zip` | `DE32E0B0D4CFB84665E5EE6CBD46E598844DB0AF258739862112963CE34AD306` |

## Notes

- Artifactory `Release/` contents (including MIGraphX stack and `ModelBench-Ort.exe`) are preserved under `bin/`.
- gpuep-signed EP DLLs are not replaced by CI copies.
