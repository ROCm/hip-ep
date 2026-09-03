# MatMulNBits offline autotune LUT

Ships the winning kernel config for shapes we have measured, so the first call
on a new shape does not have to pay for a runtime sweep. Modelled on the GQA
LUT next door (`../gqa/`), with three match tiers instead of five — the
MatMulNBits key space is much smaller, because `N` and `K` are model constants
rather than runtime geometry.

Currently populated for **bits=4 on gfx1151**. The schema reserves the other
bit widths; adding them is a table regeneration, not a schema change.

## Where it sits

```
map cache hit           -> use it
map miss, LUT hit       -> use it, and write it into the map (and the disk file)
both miss               -> runtime sweep, as before
```

The map stays in front of the LUT deliberately. It holds measurements from this
exact machine and build, which beat a table tuned on a reference part. The LUT's
job is to make the *first* encounter with a shape free, not to override
something already measured locally. Writing LUT hits into the map means the
lookup runs once per shape per process rather than once per call.

A miss at every tier is not an error — the caller sweeps and everything behaves
as it did before the LUT existed. That is also what happens on an arch with no
table, or when the table is rejected as incompatible.

## Files

| | |
|---|---|
| `matmul_nbits_autotune.fbs` | schema; row layout, key enums, config payload |
| `matmul_nbits_autotune.h` | lookup API (no flatbuffers in the header, so the kernel .hip can include it cheaply) |
| `matmul_nbits_autotune.cpp` | loader, key packing, tier probe |
| `lut/<arch>.json` | reviewable source of truth |
| `lut/<arch>.fb` | binary artifact from flatc |
| `lut/<arch>_lut_data.cpp` | embedded payload, checked in; produced by `compile` |
| `shapes/oga_models_bits4.csv` | the shape inventory the exact tier is built from |
| `scripts/extract_shapes.py` | ONNX graphs -> shape inventory |
| `scripts/update_lut.py` | measure -> build -> compile |
| `tools/matmul_nbits_autotune_sweep.cpp` | GPU sweep driver |
| `tools/empty_lut_data.cpp` | zero-size payload, for builds that must not consult a table |

## The three tiers

Probed finest first; the first row that exists **and** whose geometry the live
config table still accepts wins.

| Tier | Key | Answers |
|---|---|---|
| **Exact** | phase, bits, `N`, `K`, group_size, zero-point, row-stride, M-bucket | the shapes in `shapes/` — every MatMulNBits node in 20 shipped models |
| **Fuzzy** | same, but `N` and `K` collapsed to octave buckets | a model nobody measured whose layer dims land near a measured one |
| **Fallback** | phase + bits only | guarantees a hit; the config that won most often for that phase |

`phase` is not just prefill/decode. It is one of `Prefill` (WMMA tiles),
`Decode` (fp16 GEMV) and `DecodeDp4a` (W4A8 integer-dot GEMV). The two decode
kernels read the same `kGemvConfigs` table but rank it differently, which is why
they already keep separate tune caches at runtime — they need separate rows for
the same reason.

Row stride is part of the prefill key because a padded buffer has a different
winner (measured: cfg11 -> cfg3 at N=5120 K=4096), so the two layouts must not
share a row.

## Why the row stores geometry, not a config index

`kWmmaConfigs` and `kGemvConfigs` are edited from time to time — the
`TILE_N=32/64` removal is a recent example — and an index into them is only
meaningful for the exact table it was measured against. A stale index silently
launches the wrong kernel.

So a row stores the tile geometry (`bm, bn, swizzle, wt_m, wt_n, bk, fused`, or
`threads, tile_n`), and the runtime looks that geometry up in the live table
while probing. A geometry the table no longer contains, or that is illegal for
this shape, makes the probe **fall through to the next tier** instead of
returning something unlaunchable. Stale table, degraded answer, never a wrong
one.

`kernel_abi` (`"matmul_nbits-v1"`) is the coarser guard: bump it when a config's *meaning*
changes rather than its existence, and the whole table is rejected.

## Regenerating

```powershell
# 1. shape inventory (only when the model set changes)
python scripts/extract_shapes.py `
    --models ../../../../../../ModelFiles/oga_models `
    --out shapes/oga_models_bits4.csv --bits 4

# 2. build the sweep driver. It links tools/empty_lut_data.cpp on purpose:
#    with the real table linked it would get LUT hits for the very shapes it is
#    supposed to be measuring.
clang++ -x hip --offload-arch=gfx1151 -O3 -std=c++17 -w `
    -I lib/Runtime/Kernels/include -I <flatbuffers include> -I <generated header dir> `
    tools/matmul_nbits_autotune_sweep.cpp tools/empty_lut_data.cpp matmul_nbits_autotune.cpp `
    ../../matmul_nbits_kernel.hip -o mn_sweep.exe

# 3. measure -> build -> compile
python scripts/update_lut.py all --sweep mn_sweep.exe --flatc <flatc>
```

`measure` clears the on-disk tune caches first; without that, an already-cached
shape is never re-tuned and would be silently missing from the table.

The winners come from the in-kernel autotuner, read off its debug log. Nothing
in the pipeline re-implements "which config is fastest" — there is one
implementation of that and it is the one that ships, so the table cannot drift
from the runtime's own judgement.

### Sweep scope

`N >= 65536` (the vocab projection) is swept at `M=1` only. Its B runs to
600 MB and a WMMA sweep is ~680 dispatches over it — tens of minutes per point —
for an M that inference never issues, since logits are computed for the last
token. Fuzzy and Fallback still answer if some caller does hand one a wide M.

## Embedding

`lut/<arch>_lut_data.cpp` is linked into `custom_kernels_<arch>` (see
`lib/Runtime/Kernels/CMakeLists.txt`), not into `runtime.bc` the way the GQA
table is: the table is per-arch and so is that target, so each DLL carries
exactly the table it can use. The C++ file is generated at LUT regeneration
time by `update_lut.py compile` — the hip-ep customer build does not run Python.
flatbuffers is header-only for reading, so this adds an include path and no
link dependency.

An arch with no measured table still builds — CMake falls back to
`tools/empty_lut_data.cpp`, the loader finds no table, and every shape falls
through to the sweep. Adding an arch is a table addition, never a build break.

`HIPDNN_MATMUL_LUT_LOG=1` logs load status, per-lookup tier hits, and misses.
`HIPDNN_MATMUL_AUTOTUNE_MODE=online` bypasses the table and runs the in-kernel
autotune sweep instead (default `lookup` uses the table).
