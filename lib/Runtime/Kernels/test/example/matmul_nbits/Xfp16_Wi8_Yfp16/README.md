# MatMulNBits bits=8 (uint8, unpacked) test

Standalone `hipcc`-only test for `hip_matmul_nbits()` (bits=8; B is one byte
per weight, not bit-packed). No CMake, no EP build. Entirely self-contained:
inputs and the CPU fp32 dequant+matmul reference are all generated
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
{fp16,fp32}` (12 combos, always full) x a comprehensive M-in-`{1,16,64,128,
512}` x 4-(K,N)-family typical-shape grid thinned by tier (tier3 = 12
shapes, tier2 = 8, tier1 = 2 -- "1 decode + 1 prefill"). The CPU reference
is multithreaded over `N` so the wider grid stays affordable.
`COVERAGE=1|2|3` only picks how many typical shapes run. At startup the exe
prints `coverage=N -> running <n> matmul_nbits_i8 cases`. No human edits a
shape list -- there is no `shapes.csv` or `gen_data.py` in this leaf.

The widened `K=11008` down-proj family (at `M=64`, `gs=128`, `zero=on`)
pushed a handful of near-zero-reference elements just past the old
per-element tolerance -- bits=8's full uint8 dequant range (0..255, ~16x
bits=4's 0..15) accumulates proportionally more fp16 rounding over that many
terms. The absolute-tolerance floor was widened from `0.1` to `0.2` to
absorb this (confirmed via `test_custom`: relL2 stays ~5-7e-4 either way --
accumulation-depth noise, not a K-dependent correctness bug).

bits=8 zero-points are passed to the kernel as a **raw uint8 buffer**
(`zp_elem_size=1`, default zero point 128 when absent) -- unlike the u4 leaf,
which passes fp16-cast zero-points (`zp_elem_size=2`). See the comment above
`genCase()` / `runOne()` in the cpp.

`MODE=lookup` (default) resolves from `hip/autotune/matmul_nbits/lut/<arch>.fb`
if it exists, else falls back to `autotune` with a warning. `MODE=lookup`
needs `flatc` plus its `include/` (a build tool outside this
repo): `FLATC=<path to flatc(.exe)> FLATBUFFERS_INC=<its include dir>`.

The bits=8 GEMV and WMMA paths issue a real lookup keyed on their own weight
width, so the table decides: a table measured only at other widths reports a
miss and the kernel sweeps. `out/results.csv`'s `config` column records what was
actually used for each case -- `lookup:config[...]` on a hit, or
`autotune:best config[...]` on the sweep that follows a miss -- read back from
the op's own `HIPDNN_MATMUL_LUT_LOG` / `HIPDNN_MATMUL_AUTOTUNE_LOG` output.

CI passes `OFFLOAD`/`HIP_SDK` explicitly; there is no personal default.
