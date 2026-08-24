<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# gpuep-rel-2610.2 gpu-test-package

Vendored Windows GPU test package for gpuep release validation. Layout matches CI `gpu-test-package` (`bin/` + root `morphizen_config.json`).

## Sources

| Component | Source |
|---|---|
| EP / ROCm / ORT runtime DLLs | [Artifactory modelbench-ort gpuep-rel-2610.2](https://mkmartifactory.amd.com/artifactory/SW-UIF2.0-DEV-LOCAL/UAI/packaging/Pre-Prod/modelbench-ort/gpuep-releases/gpuep-rel-2610.2/3fc3e050338ca5ec592b8da0b0ce382a064d521a/56/3fc3e05_modelbench-ort_56.zip) (commit `3fc3e050338ca5ec592b8da0b0ce382a064d521a`, build 56) |
| Test executables (8) | [CI gpu-test-package](https://github.com/ROCm/hip-ep/actions/runs/32719961059/artifacts/9518622988) (main, run `32719961059`, artifact `9518622988`) |
| morphizen_config.json | `etc/morphizen_config.json` from hip-ep main |

## Supplemental test executables

Copied from main CI artifact into `bin/` (Artifactory Release had only `ModelBench-Ort.exe` and `migraphx-hiprtc-driver.exe`):

- `onnxruntime_perf_test.exe`
- `model_benchmark.exe`
- `model_mm.exe`
- `hip-onnx-runner.exe`
- `hip-test.exe`
- `hip-inspect.exe`
- `hip-mlir-opt.exe`
- `hip-compiler.exe`

## Package

| File | SHA256 |
|---|---|
| `gpuep-rel-2610.2-gpu-test-package.zip` | `887982018A324AA20F14955912D9A1D9A2CEBC1484E59725F381B16049F089A6` |

Contents: 139 files, ~260 MB uncompressed layout in zip.

## Notes

- Does not include `lib/` or `wheels/` from CI; full `gpu-test` native-link / wheel smoke gates are not expected to pass with this package.
- Artifactory `Release/` contents (including MIGraphX stack and `ModelBench-Ort.exe`) are preserved under `bin/`.
