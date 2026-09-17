# MatMulNBits bits=4 (uint4) test

Standalone `hipcc`-only test for `hip_matmul_nbits()` (bits=4, nibble-packed).
No CMake, no EP build. Entirely self-contained: inputs (uint4 packing,
scales, zero-points) and the CPU fp32 dequant+matmul reference are all
generated in-process, no python, no data files, no `example/common/`.

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

`test_matmul_nbits.cpp`'s `sweep` namespace crosses two axis groups:

- **Categorical, always full**: `group_size` in `{32,64,128}` x zero-points
  `{on,off}` x dtype `{fp16,fp32}` -- 12 combos, every tier.
- **Typical shapes, thinned by tier**: tier3 (default, 4 shapes) = decode
  `M=1` at two real model layer sizes (`K=4096,N=11008` FFN-ish;
  `K=2880,N=5120` attn-proj-ish, which also exercises the group_size
  zero-padding path since 2880 isn't a multiple of 64/128) plus two smaller
  prefill-representative shapes (`M=128` and `M=512` at `K=512,N=1024`).
  tier2 drops the second decode shape (3 shapes); tier1 keeps just the first
  decode shape and the `M=512` prefill shape (2 shapes, "1 decode + 1
  prefill"). The `M=128/512` shapes are deliberately smaller than a real FFN
  layer -- the CPU reference is a plain `O(M*K*N)` triple loop with no BLAS,
  so `M=512` at real FFN size (`K=4096,N=11008`) would take the reference
  alone minutes-to-hours; see the comment above `genCase()` in the cpp.

`COVERAGE=1|2|3` (`make test COVERAGE=N` or env `HIPDNN_UT_COVERAGE`) only
picks how many typical shapes run (2/3/4); it never drops a group_size/
zero-points/dtype combo. At startup the exe prints `coverage=N -> running
<n> matmul_nbits cases`. No human edits a shape list -- there is no
`shapes.csv` or `gen_data.py` in this leaf.

`MODE=auto` (default) -- `lut` if `hip/autotune/matmul_nbits/lut/<arch>.fb`
exists for the arch in `OFFLOAD`, else `autotune`. `MODE=lut` forces it
(falls back to `autotune` with a warning if the `.fb` is missing). `MODE=lut`
needs `flatc` + its `include/` (a build tool, not part of this repo --
see `example/README.md`): `FLATC=<path to flatc(.exe)> FLATBUFFERS_INC=<its
include dir>`. The LUT bytes are embedded directly into `test_matmul_nbits.cpp`
via a one-line C23 `#embed` (see `example/README.md` "`MODE=auto`") -- no
`embed_lut.py`, no generated `.cpp`. Both modes append to `out/results.csv`.

CI passes `OFFLOAD`/`HIP_SDK` explicitly; there is no personal default.
