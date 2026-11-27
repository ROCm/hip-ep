# MatMulNBits bits=2 (uint2, 4-per-byte packed) test, fp32 activations

Standalone `hipcc`-only test for `hip_matmul_nbits()` (bits=2, packed 4
values/byte LSB-first -- a plain continuous 2-bit stream, no group_size
padding). No CMake, no EP build. Entirely self-contained: inputs (uint2
packing, scales, zero-points) and the CPU fp32 dequant+matmul reference are
all generated in-process, no python, no data files, the shared `unit_test/common/` timer and ranking APIs.

Activations (`A`) and the output (`C`) are fp32; otherwise this is
`matmul_nbits/Xfp16_Wu2_Yfp16`.

```
make test        OFFLOAD=--offload-arch=%GFX% HIP_SDK=%HIP_SDK% [COVERAGE=1|2|3]
make test_custom OFFLOAD=... HIP_SDK=... SIZE=128x2880x5120 GS=128 [NO_ZEROS=1]
make clean
```

| Target | Meaning |
|---|---|
| `test` | Coverage-tiered sweep -- see "Shape coverage" below. |
| `test_custom` | One shape from `SIZE=MxKxN`, `GS=`, `NO_ZEROS=1` (in-process rng). |
| `clean` | Removes `out/`. |

## Shape coverage (3 tiers)

Same model as `matmul_nbits/Xfp16_Wu4_Yfp16` (see its README for the full
rationale): `group_size {32,64,128} x zero-points {on,off}` (6
combos, always full) x a comprehensive M-in-`{1,16,64,128,
512}` x 4-(K,N)-family typical-shape grid thinned by tier (tier3 = 12
shapes, tier2 = 8, tier1 = 2 -- "1 decode + 1 prefill"). The CPU reference
is multithreaded over `N` so the wider grid stays affordable.
`COVERAGE=1|2|3` only picks how many typical shapes run. At startup the exe
prints `coverage=N -> running <n> matmul_nbits_u2 cases`. No human edits a
shape list -- there is no `shapes.csv` or `gen_data.py` in this leaf.

`make test` uses raw per-group uint8 zero points. `test_custom` additionally
accepts `ZP_FORMAT=raw|packed|fp16` (executable `--zp-format`): packed runs the
production 2-bit unpack once, verifies every group and reuses the unpacked
buffer like the runtime wrapper; fp16 uses fractional 1.5 zero points. Both
integer formats and fp16 share the existing asymmetric LUT key; the native
memo still distinguishes the zero-point element type.

`MODE=lookup` (default) resolves from `hip/autotune/matmul_nbits/lut/<arch>.fb`
if it exists, else falls back to `autotune` with a warning. `MODE=lookup`
needs `flatc` plus its `include/` (a build tool outside this
repo): `FLATC=<path to flatc(.exe)> FLATBUFFERS_INC=<its include dir>`.

The B2 GEMV and WMMA paths consult the in-process memo, then the table,
then the existing online tuner on a miss. GEMV queries Decode below M=16;
WMMA queries Prefill/Arrival from M=16 for supported K/group sizes. u2 has
no dp4a or padded-row route. Config geometry is checked against the real u2
catalog, including unfused dequant grid limits. `out/results.csv` records
actual lookup source and config, including intervals.

`RANK=1` exposes the real u2 candidates to the native rank APIs. Rank shape
suffixes are `_noz`, `_zp16`, and `_zppacked` (raw asymmetric has no suffix).
Primary table import uses one measured activation/zero-point representation;
other representations qualify the shared key separately. The optional
`HIPDNN_UT_GPU_ADMISSION` handshake brackets each case's device allocation,
work, synchronization and cleanup after CPU reference preparation.

CI passes `OFFLOAD`/`HIP_SDK` explicitly; there is no personal default.

`--cases <CSV>` runs ordinary actual lookup over a native `leaf,shape,priority,config` workload CSV. The config field is ignored; no rank selection adapter is invoked. It is mutually exclusive with `HIPDNN_UT_RANK`.
