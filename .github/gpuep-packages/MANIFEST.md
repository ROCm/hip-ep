<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# gpuep-rel-2610.2 gpu-test-package

Vendored Windows GPU test packages for gpuep release validation on `ci/gpuep-test-package*` branches. Layout matches CI artifacts (`gpu-test-package` + `hip-python-package`).

## Sources

| Component | Source |
|---|---|
| EP / ROCm / ORT runtime DLLs + `bin/rocblas/` | [Artifactory modelbench-ort gpuep-rel-2610.2 build 64](https://mkmartifactory.amd.com/artifactory/SW-UIF2.0-DEV-LOCAL/UAI/packaging/Pre-Prod/modelbench-ort/gpuep-releases/gpuep-rel-2610.2/3fc3e050338ca5ec592b8da0b0ce382a064d521a/64/3fc3e05_modelbench-ort_64.zip) (commit `3fc3e050`, build 64) |
| Test executables (8), `lib/`, `wheels/` | [CI gpu-test-package](https://github.com/ROCm/hip-ep/actions/runs/32827368750/artifacts/9555929267) (main run `32827368750`, artifact `9555929267`) |
| `hip-python-package` (EP wheel + scripts) | [CI hip-python-package](https://github.com/ROCm/hip-ep/actions/runs/32827368750/artifacts/9555931992) (main run `32827368750`, artifact `9555931992`) |
| morphizen_config.json | `etc/morphizen_config.json` from hip-ep main |

## Supplemental from main CI

**Test executables** (Artifactory Release had only `ModelBench-Ort.exe` and `migraphx-hiprtc-driver.exe`):

- `onnxruntime_perf_test.exe`, `model_benchmark.exe`, `model_mm.exe`, `hip-onnx-runner.exe`, `hip-test.exe`, `hip-inspect.exe`, `hip-mlir-opt.exe`, `hip-compiler.exe`

**`lib/`, `wheels/`, `hip-python-package`** — from latest main Windows Build so `gpu-test` (OGA wheel smoke / VLM benchmark) can run against gpuep-signed EP DLLs in `bin/`.

## Packages

| File | SHA256 |
|---|---|
| `gpuep-rel-2610.2-gpu-test-package.zip` | `E8667DEA60CF994B3AC06D423544953602A3A17ACC314C89EEB8F9197A0AE5A5` |
| `gpuep-rel-2610.2-hip-python-package.zip` | `EAF803AEDEAEE91E6E33C67D7A22377EEAD34E9614B773D8765C9465E5B8A69B` |

## Notes

- Artifactory `Release/` contents (MIGraphX, `ModelBench-Ort.exe`, `rocblas/`) are preserved under `bin/`.
- gpuep-signed EP DLLs are not replaced by CI copies; main `onnxruntime_ep_hip` wheel ships via `hip-python-package`.
