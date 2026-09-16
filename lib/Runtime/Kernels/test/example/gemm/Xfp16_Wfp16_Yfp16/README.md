# hip_gemm test

Standalone `hipcc`-only correctness + bench test for `hip_gemm()`
(`gemm_kernel.hip`). No CMake, no EP build.

```
make test        OFFLOAD=--offload-arch=%GFX% HIP_SDK=%HIP_SDK%
make test_shapes OFFLOAD=... HIP_SDK=... [SHAPES=shapes.csv]
make test_model  OFFLOAD=... HIP_SDK=... [MODEL_JSON=../../models/gpt_oss_20b.json]
make test_custom OFFLOAD=... HIP_SDK=... M=128 N=4096 K=2880 TA=0 TB=1 TYPE=0
make clean
```

`TYPE`: 0=f16, 1=f32, 3=bf16.

| Target | Meaning |
|---|---|
| `test` | `gen_data.py` writes `data/A.bin`/`B.bin` for `M=128 N=512 K=256 TA=0 TB=1`, then the exe loads them (`--data-dir data`). One shape, not the old 11-case matrix -- see note below. |
| `test_shapes` | Every row in `SHAPES` (default `shapes.csv`); each row re-runs `gendata` for its own shape via `test_custom`. |
| `test_model` | Every `M_array` x `KN_pairs` shape in `MODEL_JSON` (shared `../../models/*.json`), `TA=0 TB=1 TYPE=0`. Still in-process random data (no `--data-dir`). |
| `test_custom` | One shape from `M=`, `N=`, `K=`, `TA=`, `TB=`, `TYPE=`; also `gendata`+`--data-dir`. |
| `clean` | Removes `out/` and `data/`. |

`test`/`test_custom` inputs now come from `gen_data.py` (numpy) instead of the
in-process `std::mt19937` fill, per the kernel-UT dtype convention (every leaf
has a `gen_data.py`, `make test` runs it first). This replaced the previous
bare-binary smoke that exercised an 11-case built-in matrix (decode NT/NN,
WMMA NT/NN, K-remainder, bias, fp32 GEMV, split-K) with a single `TYPE=0`
no-bias shape -- that matrix still exists in `test_gemm.cpp`'s `main()` and
runs when the exe is invoked with no arguments at all (not reachable through
any Makefile target), so no coverage was deleted, just demoted to
not-Make-driven. `test_model` and `bench_*` are unaffected (still in-process
random data; only the single-shape CLI path reads `--data-dir`).

`MODE=autotune` (default) links an empty `resolve()` stub so the kernel runs
its own runtime autotune sweep (`HIPDNN_EP_DEBUG=1` logs it;
`HIPDNN_EP_GEMM_AUTOTUNE=0` forces the heuristic). `MODE=lut`: if
`hip/autotune/gemm/lut/<arch>.fb` exists for the arch in `OFFLOAD`, this build
prints a one-line notice and falls back to `MODE=autotune` -- the real
FlatBuffers LUT resolver needs a flatc-generated header that only the CMake
build produces, and this test intentionally builds without CMake/flatc. It
never fails because of this.

`bench_hipblaslt`/`bench_suite` (not part of the uniform API, not CI) compare
against hipBLASLt.

CI passes `OFFLOAD`/`HIP_SDK` explicitly; there is no personal default.
