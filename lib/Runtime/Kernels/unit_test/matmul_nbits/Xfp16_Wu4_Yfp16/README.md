# MatMulNBits bits=4 (uint4) test

Standalone `hipcc`-only test for `hip_matmul_nbits()` (bits=4, nibble-packed).
No CMake, no EP build. Entirely self-contained: inputs (uint4 packing,
scales, zero-points) and the CPU fp32 dequant+matmul reference are all
generated in-process, no python, no data files, no `unit_test/common/`.

Activations (`A`) and the output (`C`) are fp16; `matmul_nbits/Xfp32_Wu4_Yfp32` is
the same test with fp32 activations.

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

`test_matmul_nbits.cpp`'s `sweep` namespace crosses two axis groups:

- **Categorical, always full**: `group_size` in `{32,64,128}` x zero-points
  `{on,off}` -- 6 combos, every tier.
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
  (which would multiply the already-6-way categorical cross to 120
  cases/leaf). tier2 (8 shapes) keeps all 4 `M=1` families plus one mid-`M`
  point per remaining `M`; tier1 (2 shapes) keeps just the smallest `M=1`
  family and one `M>1` point ("1 decode + 1 prefill"). Every shape is ALSO
  crossed with the full 6-way categorical set in `buildCases()`, so the
  shape-list size directly multiplies total CPU-reference cost by 6 -- the
  reference (`genCase()`'s `O(M*K*N)` triple loop, no BLAS) is now
  multithreaded over `N` so this wider set stays affordable; see the comment
  above `genCase()` in the cpp.

`COVERAGE=1|2|3` (`make test COVERAGE=N` or env `HIPDNN_UT_COVERAGE`) only
picks how many typical shapes run (2/8/12); it never drops a group_size/
zero-points combo. At startup the exe prints `coverage=N -> running
<n> matmul_nbits cases`. No human edits a shape list -- there is no
`shapes.csv` or `gen_data.py` in this leaf.

`MODE=lookup` (default) -- resolves from `hip/autotune/matmul_nbits/lut/<arch>.fb`
if it exists for the arch in `OFFLOAD`, else falls back to `autotune` with a
warning. `MODE=autotune` always sweeps, ignoring any `.fb`. `MODE=lookup`
needs `flatc` + its `include/` (a build tool, not part of this repo --
see `unit_test/README.md`): `FLATC=<path to flatc(.exe)> FLATBUFFERS_INC=<its
include dir>`. The LUT bytes are embedded directly into `test_matmul_nbits.cpp`
via a one-line C23 `#embed` (see `unit_test/README.md` "`MODE=lookup`") -- no
`embed_lut.py`, no generated `.cpp`. Both modes append to `out/results.csv`.

CI passes `OFFLOAD`/`HIP_SDK` explicitly; there is no personal default.

## EP entry and zero-point coverage

Single shapes accept `--zp-type u8` for ONNX nibble-packed zero points, or
`--no-zeros`. The packed rows use `ceil(groups_k/2)` bytes independently;
zero points are expanded once like the EP's per-weight cache.

The fp16 leaf additionally accepts `--route dp4a|padrow|nbits` (default
`nbits`). DP4A is restricted to the EP's default single-row decode route,
with `32 | K`, `group_size | K`, and u8 or absent zero points. Its time
includes per-call activation quantization. Padrow time includes the
per-call weight repack and follows the EP's generic fallback on a miss.
The fp32 leaf rejects the separate fp16-only routes. Unknown arguments
fail explicitly instead of silently selecting a different operation.

The original 20-launch cold-weight rotation mean is unchanged. Coverage
adds packed-zero-point cases and inexpensive separate-entry boundary
cases. CPU checks retain the per-element absolute/relative tolerance;
DP4A is checked against scalar per-group dynamic int8 quantization, with
a separate approximation relL2 limit of 0.01 against the original full
reference. Each result prints the actual entry and zero-point format.

## Pinned comparisons with checked timing

An explicit shape accepts `--configs production,38,39`. The test generates its
inputs and CPU reference once, then checks and times production and each pinned
WMMA config on the same rotated weight copies. Arrival pins use the stride API;
`--route padrow` requires fused configs and includes the per-call weight repack.
Odd K includes the activation copy and zero tail used by production.

Enable the checked timing path with `HIPDNN_KERNEL_UT_TIMING_BLOCKS=5` and
`HIPDNN_KERNEL_UT_TIMING_LAUNCHES=0`. Zero launch count selects adaptive work;
the first accepted count requires at least 12 ms of measured GPU work. Set
`HIPDNN_KERNEL_UT_TIMING_TARGET_MS=30` when CPU completion overhead prevents
short blocks from passing corroboration. The target applies to both the pilot
and the first accepted work gate; supported targets are 12 through 100 ms. Each
accepted block must have waited behind the coherent hold without timeout and
have CPU completion corroborate the 100 MHz REALTIME clock within 2%. HIP event
agreement is reported separately. The process result is the median of five
accepted block values, with at most 15 block attempts by default. Incomplete
timing retains the separate CPU correctness verdict.

For a matched A/B comparison, repeat each side in five independent processes,
alternate side and config order, and aggregate the median of process medians
with both ranges. Freeze the first side's observed counts on subsequent runs
using `--launch-counts production=<n>,38=<n>,39=<n>`. Counts are per config and
must match across sides. Re-settle after idle gaps and retain all raw blocks;
clock agreement alone does not establish a universal performance resolution.
In-kernel autotune timings do not supply comparison or LUT winner evidence.

Compile both test and kernel with `HIPDNN_KERNEL_UT_LAUNCH_AUDIT` defined to log
`UT_LAUNCH_ID` during the untimed correctness launch. Product builds omit this
observer, and timed loops disable it. `UT_CONFIG_BEGIN`/`UT_CONFIG_END`, runtime
and DLL records, raw block clocks and CSV medians provide the audit trail.

`make lut_measure_library` builds the audited one-kernel library for the existing
LUT driver's `--requests` mode. `make lut_measure_driver` builds that driver from
a clean output directory. Both targets enable the launch observer for test and
kernel compilation; pass the same SDK, offload target and FlatBuffers settings
as `make test`. Use separate `BUILD_DIR` values for baseline and candidate builds.

Large formal shapes can set `HIPDNN_KERNEL_UT_FAST_REFERENCE=1` to compute the
full CPU reference in blocks of 32 output columns and 8 rows. Every output
retains its K accumulation order and the original `A * (q - zp) * scale`
arithmetic; independent columns vectorize and quant groups move outside the
inner K loop. This changes CPU preparation only. The same pinned kernel,
rotated weights and checked timing protocol run afterward. Small validation
controls can also set `HIPDNN_KERNEL_UT_REFERENCE_CHECK=1` to compare every
blocked output with the original scalar reference and print the maximum
absolute difference. Default coverage uses the original scalar reference.

For sampled ownership monitoring, each config prints its UTC
`UT_TIMING_WINDOW`. An optional `HIPDNN_KERNEL_UT_MONITOR_DRAIN_MS` adds an
untimed drain before timing; whole-process activity remains diagnostic, and
timing-window observations must still satisfy the monitor's ownership limits.
