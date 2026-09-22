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
- **Typical shapes, thinned by tier**: tier3 (default, 12 shapes) = M in
  `{1,16,64,128,512}` (decode/GEMV through several prefill points)
  round-robined -- not a full M x KN cross -- across 4 representative real
  layer `(K,N)` families: FFN gate/up-proj (`K=4096,N=11008`), attn-proj
  (`K=2880,N=5120`, which also exercises the group_size zero-padding path
  since 2880 isn't a multiple of 64/128), FFN down-proj (`K=11008,N=4096`),
  and a square o-proj/attn-combine shape (`K=4096,N=4096`). `M=1` (decode,
  cheap) touches all 4 families; `M=16/128` touch gate/up-proj + attn-proj;
  `M=64/512` touch down-proj + o-proj -- every `M` and every `(K,N)` family
  appears at least once without paying for the full `5x4=20`-shape cross
  (which would multiply the already-12-way categorical cross to 240
  cases/leaf). tier2 (8 shapes) keeps all 4 `M=1` families plus one mid-`M`
  point per remaining `M`; tier1 (2 shapes) keeps just the smallest `M=1`
  family and one `M>1` point ("1 decode + 1 prefill"). Every shape is ALSO
  crossed with the full 12-way categorical set in `buildCases()`, so the
  shape-list size directly multiplies total CPU-reference cost by 12 -- the
  reference (`genCase()`'s `O(M*K*N)` triple loop, no BLAS) is now
  multithreaded over `N` so this wider set stays affordable; see the comment
  above `genCase()` in the cpp.

`COVERAGE=1|2|3` (`make test COVERAGE=N` or env `HIPDNN_UT_COVERAGE`) only
picks how many typical shapes run (2/8/12); it never drops a group_size/
zero-points/dtype combo. At startup the exe prints `coverage=N -> running
<n> matmul_nbits cases`. No human edits a shape list -- there is no
`shapes.csv` or `gen_data.py` in this leaf.

`MODE=lookup` (default) -- resolves from `hip/autotune/matmul_nbits/lut/<arch>.fb`
if it exists for the arch in `OFFLOAD`, else falls back to `autotune` with a
warning. `MODE=autotune` always sweeps, ignoring any `.fb`. `MODE=lookup`
needs `flatc` + its `include/` (a build tool, not part of this repo --
see `example/README.md`): `FLATC=<path to flatc(.exe)> FLATBUFFERS_INC=<its
include dir>`. The LUT bytes are embedded directly into `test_matmul_nbits.cpp`
via a one-line C23 `#embed` (see `example/README.md` "`MODE=lookup`") -- no
`embed_lut.py`, no generated `.cpp`. Both modes append to `out/results.csv`.

CI passes `OFFLOAD`/`HIP_SDK` explicitly; there is no personal default.
