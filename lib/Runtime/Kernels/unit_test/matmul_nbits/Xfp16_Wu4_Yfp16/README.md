# MatMulNBits bits=4 (uint4) test

Standalone `hipcc`-only test for `hip_matmul_nbits()` (bits=4, nibble-packed).
No CMake, no EP build. Entirely self-contained: inputs (uint4 packing,
scales, zero-points) and the CPU fp32 dequant+matmul reference are all
generated in-process. Timing uses the owning shared
[`unit_test/common`](../../common/README.md) adaptive timer; inputs and the
CPU reference remain in this leaf.

Activations (`A`) and the output (`C`) are fp16; `matmul_nbits/Xfp32_Wu4_Yfp32` is
the same test with fp32 activations.

```
make test        OFFLOAD=--offload-arch=%GFX% HIP_SDK=%HIP_SDK% [COVERAGE=1|2|3]
make test_custom OFFLOAD=... HIP_SDK=... SIZE=128x2880x5120 GS=128 [NO_ZEROS=1]
make clean
```

Use `TIMING=screen` for config ranking and `TIMING=confirm` for selected
performance confirmations. The default is `standard`. These named presets
change sampling; they keep the selected production entry and CPU check.
The result is the median of five group means, each valid GPU total divided by
complete calls. `UT_TIMING` and `UT_TIMING_BINDING` record the actual preset,
shape, config, source, CPU verdict and returned time. See the shared timer
README for accumulation requirements and invalid-event handling.

| Target | Meaning |
|---|---|
| `test` | Coverage-tiered sweep -- see "Shape coverage" below. |
| `test_custom` | One shape from `SIZE=MxKxN`, `GS=`, `NO_ZEROS=1` (in-process rng). |
| `clean` | Removes `out/`. |

After the sweep, the exe forces every extended WMMA entry (on WMMA devices),
then checks the weight byte orders: `hip_matmul_nbits_dp4a_layout` at M = 1,
2, 4 and 7 on the rows, XOR-swizzled lines and 16-row blocks of each weight
against a CPU reference of the dp4a arithmetic, and
`hip_matmul_nbits_u4_unlayout` back to the dense rows byte for byte. The
byte orders are written out in the test from their definition
(`include/hip/matmul_nbits_weight_layout.h`). The forced phase also runs
every extended entry with a nonzero variant on both byte orders.
`--layouts-only` and `--forced-only` run just one phase.

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

## Explicit production entries and a supplied table

The direct executable accepts `MxKxN`, `--group-size`, and `--no-zeros`.
Additional entry flags distinguish the production layouts:

| Entry | Direct flags and supported scope |
|---|---|
| Arrival FP16/W4 | No additional entry flag; ordinary `hip_matmul_nbits`. |
| Padded FP16/W4 | `--b-row-bytes <pitch>`; `M >= 2`, `K % group_size == 0`, pitch `K/2 + 128`. The measured entry is `hip_matmul_nbits_u4_wmma_padrow`. |
| Decode DP4A | `--dp4a --no-zeros`; `M == 1`, `K % 32 == 0`, no padded stride. The complete `hip_matmul_nbits_dp4a` entry includes activation quantization and the integer GEMV. |

`make test_custom` exposes the ordinary entry. Run the built executable
directly for the two additional entry flags. For example, in Windows `cmd`:

```bat
set "HIPDNN_UT_TIMING=screen"
set "HIPDNN_MATMUL_AUTOTUNE_MODE=lookup"
<build-dir>\test_direct.exe 68x320x10240 --group-size 32 --no-zeros
<build-dir>\test_direct.exe 68x6144x2560 --group-size 32 --no-zeros --b-row-bytes 3200
<build-dir>\test_direct.exe 1x640x2560 --group-size 32 --no-zeros --dp4a
```

`--list-configs` prints the actual WMMA and GEMV geometry inventory.
For a legal explicit pin, set `HIPDNN_MATMUL_NBITS_WMMA_CFG` for WMMA or
`HIPDNN_EP_MATMUL_DP4A_CFG` for DP4A in that child process. Remove both pins
when qualifying the actual LUT lookup. Runtime config IDs come from this
inventory; they are independent of the table's config-pool indices.

Build the same owning leaf with an explicit final FB using the existing
Makefile override:

```bat
make build HIP_SDK=<sdk> OFFLOAD=--offload-arch=gfx11-generic MODE=lookup LUT_ARCH=gfx1151 BUILD_DIR=<private-build-dir> LUT_FB=<final-gfx1151.fb> FLATC=<flatc.exe> FLATBUFFERS_INC=<flatbuffers-include>
```

The kernel compiles at `-O2`; test and reader host code use the Makefile's
`-O3` settings. `LUT_FB` selects the bytes embedded by the existing `#embed`
path. A v5 table containing M intervals requires the matching interval-aware
reader/schema from the source patch; the reader also accepts the old v4 table.
Check the `[MODE=... -> lookup]` build output and the actual lookup source
(`exact`, `nearest`, `interval`, or `fallback`) in each CPU binding.
An unexpected fallback or online tune is an unresolved LUT selection.

Retain the source/build/FB hashes, logs and exact executable with the runtime
DLLs used for qualification. Put both qualified DLLs beside a copied executable
so its runtime selection is reproduced. A rebuilt executable needs complete
code/input/entry retention evidence or fresh ranking; an unchanged kernel ISA
alone does not qualify old measurements for a new table build.
