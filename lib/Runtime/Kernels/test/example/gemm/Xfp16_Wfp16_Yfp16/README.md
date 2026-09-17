# hip_gemm test

Standalone `hipcc`-only correctness + bench test for `hip_gemm()`
(`gemm_kernel.hip`). No CMake, no EP build. Entirely self-contained: inputs
and the CPU reference are generated in-process (`std::mt19937`), no python,
no data files, no `example/common/`.

```
make test        OFFLOAD=--offload-arch=%GFX% HIP_SDK=%HIP_SDK% [COVERAGE=1|2|3]
make test_custom OFFLOAD=... HIP_SDK=... M=128 N=4096 K=2880 TA=0 TB=1 TYPE=0
make clean
```

`TYPE`: 0=f16, 1=f32.

| Target | Meaning |
|---|---|
| `test` | Runs the exe with no positional shape, which hits the built-in `cases[]` matrix in `test_gemm.cpp`'s `main()` (every TA/TB/bias/dtype situation x a small typical-shape list -- see "Shape coverage" below). |
| `test_custom` | One shape from `M=`, `N=`, `K=`, `TA=`, `TB=`, `TYPE=` (in-process rng). |
| `clean` | Removes `out/`. |

## Shape coverage (3 tiers)

`test_gemm.cpp`'s `main()` has an explicit `cases[]` array (M/N/K x TA/TB
transpose x bias on/off x dtype path fp16/fp32, each row commented with
which dispatch path it targets -- decode/WMMA/K-remainder/bias/split-K/etc,
~16 rows). This is a hand-picked set of (categorical situation, typical
shape) pairs, not a full cross product -- a blind M x N x K sweep would
mostly re-test the same path a targeted case already covers.
`COVERAGE=1|2|3` (default 3, `make test COVERAGE=N` or env
`HIPDNN_UT_COVERAGE`) picks a fixed index subset of that array (`kTier1`
5 rows, `kTier2` 11 rows, tier3 all 16) chosen so every tier still touches
every TA/TB/bias/dtype value. At startup the exe prints `coverage=N ->
running <n>/16 gemm cases`. No human edits a shape list -- there is no
`shapes.csv` or `gen_data.py` in this leaf.

`MODE=autotune` (default) links an empty `resolve()` stub so the kernel runs
its own runtime autotune sweep (`HIPDNN_EP_DEBUG=1` logs it;
`HIPDNN_EP_GEMM_AUTOTUNE=0` forces the heuristic). `MODE=lut`: if
`hip/autotune/gemm/lut/<arch>.fb` exists for the arch in `OFFLOAD`, links the
real resolver + embeds the `.fb` via a one-line C23 `#embed` in
`test_gemm.cpp` (see `example/README.md` "`MODE=auto`"); needs
`FLATC=<path> FLATBUFFERS_INC=<dir>`. Falls back to `MODE=autotune` (with a
notice) if the `.fb` is missing for this arch -- never a hard failure.

CI passes `OFFLOAD`/`HIP_SDK` explicitly; there is no personal default.
