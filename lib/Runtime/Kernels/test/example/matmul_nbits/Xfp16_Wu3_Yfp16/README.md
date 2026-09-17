# MatMulNBits bits=3 (uint3, continuous bitstream) test

Standalone `hipcc`-only test for `hip_matmul_nbits()` (bits=3, a continuous
per-row 3-bit bitstream -- value `k` occupies bits `[3k, 3k+3)`, LSB-first;
NOT an ONNX MatMulNBits convention, see `matmul_nbits_kernel.hip`). No CMake,
no EP build. Entirely self-contained: inputs (uint3 packing, scales,
zero-points) and the CPU fp32 dequant+matmul reference are all generated
in-process, no python, no data files, no `example/common/`.

```
make test        OFFLOAD=--offload-arch=%GFX% HIP_SDK=%HIP_SDK% [COVERAGE=1|2|3]
make test_custom OFFLOAD=... HIP_SDK=... SIZE=128x2880x5120 GS=128 [NO_ZEROS=1] [FP32=1]
make clean
```

| Target | Meaning |
|---|---|
| `test` | Coverage-tiered sweep -- see "Shape coverage" below. |
| `test_custom` | One shape from `SIZE=MxKxN`, `GS=`, `NO_ZEROS=1`, `FP32=1` (in-process rng). |
| `clean` | Removes `out/`. |

## Shape coverage (3 tiers)

Same model as `matmul_nbits/Xfp16_Wu4_Yfp16` (see its README for the full
rationale): `group_size {32,64,128} x zero-points {on,off} x dtype
{fp16,fp32}` (12 combos, always full) x a typical-shape list thinned by
tier (tier3 = 4 shapes, tier2 = 3, tier1 = 2 -- "1 decode + 1 prefill").
`COVERAGE=1|2|3` only picks how many typical shapes run. At startup the exe
prints `coverage=N -> running <n> matmul_nbits_u3 cases`. No human edits a
shape list -- there is no `shapes.csv` or `gen_data.py` in this leaf.

`make test`'s sweep exercises bits=3 only, with zero-points passed as a raw
per-group uint8 buffer (`zp_elem_size=1`, plain per-element convention,
default zero point 4 when absent) -- the 3-bit-packed zero_points path this
op also supports is a lower-level kernel behavior, not exercised by this
small correctness UT.

`MODE=auto` (default) -- `lut` if `hip/autotune/matmul_nbits/lut/<arch>.fb`
exists for the arch in `OFFLOAD`, else `autotune`. `MODE=lut` forces it
(falls back to `autotune` with a warning if the `.fb` is missing). `MODE=lut`
needs `flatc` + its `include/` (a build tool, not part of this repo --
see `example/README.md`): `FLATC=<path to flatc(.exe)> FLATBUFFERS_INC=<its
include dir>`. The LUT bytes are embedded directly into
`test_matmul_nbits_u3.cpp` via a one-line C23 `#embed` -- no `embed_lut.py`,
no generated `.cpp`. Both modes append to `out/results.csv`.

CI passes `OFFLOAD`/`HIP_SDK` explicitly; there is no personal default.
