# MatMulNBits offline autotune LUT

Ships the winning kernel config for shapes we have measured, so the first call
on a new shape does not have to pay for a runtime sweep. Modelled on the GQA
LUT next door (`../gqa/`). MatMulNBits matches categorical fields exactly and
uses weighted log-space distance over the literal `M`, `N` and `K` dimensions.
An optional M interval can override that lookup for one literal K/N key.

Currently populated for **bits=4 on gfx1151**. The schema reserves the other
bit widths; adding them is a table regeneration, not a schema change.

## Where it sits

```
map cache hit           -> use it
map miss, LUT hit       -> use it, and write it into the map
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
| `lut/<arch>.fb` | binary artifact from flatc; the only shipped table form, embedded at build time |
| `shapes/oga_models_bits4.csv` | the shape inventory the exact tier is built from |
| `scripts/extract_shapes.py` | ONNX graphs -> shape inventory |
| `scripts/update_lut.py` | measure -> build -> compile |
| `tools/matmul_nbits_autotune_sweep.cpp` | GPU sweep driver |
| `tools/empty_lut_data.cpp` | zero-size payload, for builds that must not consult a table |

## Lookup order

Probed finest first; the first row that exists **and** whose geometry the live
config table still accepts wins.

| Tier | Key | Answers |
|---|---|---|
| **Interval** (optional) | phase, bits, group_size, zero-point, row-stride, literal `K` and `N`, and `(m_lo, m_hi]` | an explicitly qualified config for the covered M values |
| **Exact / Nearest** | exact categorical fields; weighted log-space distance over literal `M`, `N`, `K` | the closest usable measured point; zero distance is an Exact hit |
| **Fallback** | phase + bits only | the phase's usable last-resort config when no point can answer |

There is no octave-bucket voting tier. The table stores the positive weights
`weight_m`, `weight_n` and `weight_k`; the distance is
`sqrt((wm*log2(M/Mp))^2 + (wn*log2(N/Np))^2 + (wk*log2(K/Kp))^2)`.
If no interval, point or fallback passes the live legality callback, the answer
is `Source::None` and the caller uses its existing runtime sweep.

`phase` is not just prefill/decode. It is one of `Prefill` (WMMA tiles),
`Decode` (fp16 GEMV), `DecodeDp4a` (W4A8 integer-dot GEMV) and
`DecodeDp4aBlocks` (the W4A8 GEMV that reads a weight stored in 16-row blocks).
The first two decode kernels read the same `kGemvConfigs` table but rank it
differently, which is why they already keep separate tune caches at runtime —
they need separate rows for the same reason. `DecodeDp4aBlocks` rows name a
`kDp4aBlocksConfigs` entry as a Gemv config whose `tile_n` packs the 16-row
groups per workgroup (bits 0-3) and the K blocks per step (bits 4-7).

Row stride is part of the prefill key because a padded buffer has a different
winner (measured: cfg11 -> cfg3 at N=5120 K=4096), so layouts must not share a
row. Each pad is its own category, matched exactly: `Arrival` (dense),
`Padded` (arrival + 128 B: the per-call repack, and the 128-byte row pads) and
`Padded256` (arrival + 256 B: the load-time default pad). The lookup takes the
pad from the request's row pitch (a pitch of K/2 is dense); a pad with no
category skips the point groups and goes to the phase's fallback, so a
lookup never uses a point measured at another pitch. The sweep driver logs the
pad of each `prefill_padrow` measurement (`pad=`), and `update_lut.py` maps it
to the category.

A weight stored in another byte order (`XorLines`, `Blocks`; see Row pads) is a
category of its own, taken from the request's `layout`. A group size and zero
point with no points there is answered from its `Arrival` points, through the
caller's validator, which keeps only the extended configs that read the byte
order. The sweep
driver measures them with `--layouts xor,blocks` (it needs `--extended`): the
markers `prefill_xor` and `prefill_blocks` for prefill, and `dp4a_blocks` for
the block-reading decode GEMV at M = 1.

`--out` names the JSON that `update_lut.py build` and `merge` write and
`compile` reads; `--fb-out` names the binary, which defaults to `lut/<arch>.fb`
only for the default JSON, so a scratch build never overwrites the shipped
table. Every step checks the table it writes or reads (`check_table`) and
rejects one the loader would read differently, for example a config index out
of range or an extended entry the kernel library does not compile.

### Optional M intervals

`m_intervals` is an optional root vector. Its full key is
`phase, bits, group_size, zero_point, row_stride, k, n`; K and N must match
literally. An interval matches only `m_lo < M <= m_hi` and runs before point
lookup, including an old Exact point inside the interval. The in-process tune
map still runs before the LUT. Missing intervals, unclassified row pitches and
runtime-illegal interval configs fall through to the ordinary point/fallback
lookup. Invalid or overlapping interval records are excluded by the loader;
the builder rejects such data before writing it.

Bounds satisfy `0 <= m_lo < m_hi <= 65535`. K is a positive uint16 and N is a
positive uint32. Prefill uses `Arrival` or `Padded`; both decode phases use
`Any` for row stride. Two intervals with the same full key may meet at an
endpoint, such as `(61,64]` and `(64,68]`, but may not overlap.

The import file is a JSON **list**, separate from the root LUT JSON. For
example, the following illustrates the format; its bounds and config are not
a measured selection:

```json
[
  {
    "phase": "Prefill",
    "bits": "B4",
    "group_size": "G128",
    "zero_point": "Symmetric",
    "row_stride": "Arrival",
    "k": 2048,
    "n": 2048,
    "m_lo": 61,
    "m_hi": 68,
    "config": {
      "kind": "Wmma",
      "bm16": 8,
      "bn16": 16,
      "swizzle": 16,
      "wt_m": 2,
      "wt_n": 2,
      "bk": 32,
      "fused": 0,
      "threads": 0,
      "tile_n": 0,
      "ext": 0,
      "variant": 0
    }
  }
]
```

Use the full config geometry when importing measured selections. `bm16` and
`bn16` are the tile dimensions divided by 16; they are not LUT M coordinates.
Gemv configs use `kind: "Gemv"`, `threads` and `tile_n`. `ext: 1` and `variant`
identify a specific extended WMMA entry; the writer records them through the
root `extended_configs` vector. An integer `config` is also accepted, but it
indexes the existing LUT config pool, not `kWmmaConfigs` or `kGemvConfigs`.
The builder reuses equal full geometries and appends new ones, preserving old
indices and distinguishing extended variants. Importing the same full key and
bounds replaces that interval explicitly; disjoint existing intervals remain.

From this directory, import accepted intervals without consuming sweep logs or
changing any old point records, fallbacks or fitted weights:

```powershell
python scripts/update_lut.py build --arch gfx1151 --merge-existing `
    --intervals-only --m-intervals <accepted-intervals.json> `
    --out <review-table.json>
python scripts/update_lut.py compile --arch gfx1151 `
    --out <review-table.json> --fb-out <review-table.fb> --flatc <flatc>
```

`--merge-existing` reads `lut/<arch>.json` as the input even when `--out`
names a review file. `--intervals-only` requires both `--merge-existing` and
`--m-intervals`. Omitting `--out` writes the selected architecture's canonical
JSON; compile that same JSON to produce its FB. Other architecture files are
not rewritten by this scoped import.

Interval writers explicitly set `schema_version: 5`. The schema's field
default remains 4 so old payloads that omitted the field still read as v4.
The new reader accepts v4 without ranges and v5, and rejects v4 with nonempty
ranges, unsupported versions, a wrong architecture or a wrong kernel ABI.
Deliver v5 JSON/FB with the matching reader/schema patch and build; an old
reader does not support this format. Empty/no-range data preserves ordinary
lookup. `Source::Interval` is appended without renumbering the old sources.

Accept a range only after checking every covered required integer M against
the measured best legal config under the same CPU-checking entry, build,
layout and timing protocol. Qualify overridden old Exact keys or narrow the
range to exclude them; inspect tile/grid changes and both outside endpoints.
Endpoint agreement alone does not qualify an interior. After compiling the
real JSON/FB, check their semantic agreement, old-record preservation and
actual final lookups, with CPU correctness and a separate
`T_lookup / T_best <= 1.05` verdict for each required target.

## Row pads

A table may opt its device into storing int4 weights row-padded at load
(`hip-pad-matmul-nbits-weights`), which removes the per-call repack:

- `row_pads` lists int4 weight shapes (N, K, group size, zero points) with
  how each is stored: `layout` `Rows` and the pad, in bytes (a listed pad of 0
  keeps that weight dense with no repack), or `layout` `XorLines` / `Blocks`
  with pad 0, the dense rows in that byte order
  (`include/hip/matmul_nbits_weight_layout.h`; K % 64 == 0, and N % 16 == 0
  for `Blocks`). A byte order is applied only to a weight without bias whose
  zero points are absent or uint8, the users that read it directly.
- Only listed weights are padded. An unlisted weight stays dense; where its
  stride aliases (`hip::matmulNBitsRowStrideAliases` in
  `include/hip/matmul_nbits_row_alias.h`, shared by the compiler, the runtime
  and the sweep driver), the runtime keeps repacking it per call. A table
  without `row_pads` pads nothing at load.

Each entry comes from measurement: the pad from {0, 16, 128, 256} bytes with the
lowest prefill (prompt 1024) plus 128-token decode time, in the pad sweep and
two re-measurements, taking the smaller pad when it is within noise of the
fastest. A listed weight needs prefill rows measured at its stored pad
(`Padded256`, or `Padded` for a 128-byte entry), so layout and rows come from
the same data; without them its prefill LUT lookup misses. A byte-order entry
falls back to the arrival rows instead, so its rows (and, for `Blocks`, the
`DecodeDp4aBlocks` rows) refine it rather than enable it. The compiler reads
the entries through `hip_matmul_nbits_u4_row_pads(arch, ...)`
(`matmul_nbits_row_pad_abi.h`). `update_lut.py build` and `merge` carry them
over unchanged from `lut/<arch>.json` (`--carry-from` names another table). Only
gfx1151 lists them today.

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

A config that names an extended WMMA entry (`WMMA_EXT_CONFIGS`: the
`GemmFp16U4ImplX` pipeline options, `variant` bit 7 selecting the CU-mode
build) is listed in `extended_configs` with that entry's `variant`; it matches
only extended entries, and an unlisted config only the original ones. A table
that lists none also keeps the extended entries out of the runtime sweep, so a
device without measurements searches exactly the original table. The
extended entries run only on gfx1151, where they were measured: on any other
device the kernel library refuses them whatever the table says. To measure
them, run the sweep driver with `--extended` (`update_lut.py all
--sweep-args --extended`) on gfx1151; its winners then carry the entry and
variant, and `build` lists them in `extended_configs`.

`kernel_abi` (`"matmul_nbits-v1"`) is the coarser guard: bump it when a config's *meaning*
changes rather than its existence, and the whole table is rejected.

## Regenerating

```powershell
# 1. shape inventory (only when the model set changes). extract_shapes.py does
#    not emit the hand-added `Internal` rows; keep them in the new csv.
#    The inventory can be wider than the decode Exact coverage.
python scripts/extract_shapes.py `
    --models ../../../../../../ModelFiles/oga_models `
    --out shapes/oga_models_bits4.csv --bits 4

# 2. build the sweep driver. It links tools/empty_lut_data.cpp on purpose:
#    with the real table linked it would get LUT hits for the very shapes it is
#    supposed to be measuring. Build it like the product (see "Measure the
#    product binary" below): gfx11-generic -O2 for the gfx1150/1151/1152
#    tables, gfx1170 -O2 for gfx1170.
clang++ -x hip --offload-arch=gfx11-generic -O2 -std=c++17 -w `
    -I lib/Runtime/Kernels/include -I include -I <flatbuffers include> -I <generated header dir> `
    tools/matmul_nbits_autotune_sweep.cpp tools/empty_lut_data.cpp matmul_nbits_autotune.cpp `
    ../../matmul_nbits_kernel.hip -o mn_sweep.exe

# 3. measure -> build -> compile. --arch is the chip the sweep runs on (the
#    table is per chip even though the binary is generic).
python scripts/update_lut.py all --arch gfx1151 --sweep mn_sweep.exe --flatc <flatc>

# Byte-order rows (prefill_xor, prefill_blocks, dp4a_blocks) as a supplementary
# sweep next to the main log, then merged phase by phase.
python scripts/update_lut.py measure --arch gfx1151 --tag layouts --sweep mn_sweep.exe `
    --sweep-args --extended --layouts xor,blocks --layouts-only
python scripts/update_lut.py build --merge-existing --arch gfx1151 --phase Prefill
python scripts/update_lut.py build --merge-existing --arch gfx1151 --phase DecodeDp4aBlocks
```

Each sweep process starts with an empty in-process tune map (the tuners no
longer persist to an on-disk cache), so every shape is measured fresh and none
is silently skipped.

The winners come from the in-kernel autotuner, read off its debug log. Nothing
in the pipeline re-implements "which config is fastest" — there is one
implementation of that and it is the one that ships, so the table cannot drift
from the runtime's own judgement.

### Measure the product binary

The table ranks configs by their speed in the binary that is measured, so that
binary has to be the one that ships: the SDK pinned in `cmake/deps.txt`, the
release flags of `cmake/hip_utils.cmake` (`-O2`), and the offload arch of the
product DLL -- `gfx11-generic` for the gfx1150 / gfx1151 / gfx1152 tables (all
three run the generic code object), `gfx1170` for the gfx1170 table. A
`--offload-arch=gfx1151` build compiles every kernel against gfx1151's
1536-entry VGPR file instead of the generic 1024, so tiles that fit there spill
in the product: the 128x256 WT2x4 tile was the fastest config on 127 gfx1151
points in a gfx1151 build and 3.5-4x slower than the 7.14 product in the 7.16
product binary. Re-measure after every compiler bump
(`lib/Runtime/Kernels/tools/check_kernel_resources.py` lists the kernels whose
code changed).

### Sweep scope

`N >= 65536` (the vocab projection) is swept at `M=1` only. Its B runs to
600 MB and a WMMA sweep is ~680 dispatches over it — tens of minutes per point —
for an M that inference never issues, since logits are computed for the last
token. Fuzzy and Fallback still answer if some caller does hand one a wide M.

For the hand-added `Internal` inventory, the gfx1151 table covers 11 poorly
matched DecodeDp4a shapes at M=1 and every listed prefill shape at all 12 M
buckets from 8 through 16384. The 14 prefill shapes have Arrival rows; the
three padded-K shapes also have Padded rows for the same M ladder. The vocab
projection has decode coverage only. fp16 GEMV rows are unchanged.

## Embedding

Only `lut/<arch>.fb` ships. `lib/Runtime/Kernels/CMakeLists.txt` turns it into
the `kMatmulNbitsLutBlobs[]` symbol linked into `custom_kernels_<arch>` using
pure CMake (`file(READ ... HEX)`) — no Python, and no generated `.cpp` is
committed (the compiled array is exactly the `.fb`'s bytes, ~40 KB; the source
text is not). This is per-arch, not in `runtime.bc` the way the GQA table is, so
each DLL carries exactly the table it can use. flatbuffers is header-only for
reading, so this adds an include path and no link dependency.

An arch with no measured `.fb` still builds — `_emit_lut_registry` emits an
empty registry (`kMatmulNbitsLutBlobs` with count 0), the loader finds no table,
and every shape falls through to the sweep. Adding an arch is a table addition (a
`.fb`), never a build break.

`HIPDNN_MATMUL_LUT_LOG=1` logs load status, per-lookup tier hits, and misses.
`HIPDNN_MATMUL_AUTOTUNE_MODE=online` bypasses the table and runs the in-kernel
autotune sweep instead (default `lookup` uses the table); the
`matmul_autotune_mode` provider option selects the same thing at lower
precedence, and the mode is latched process-wide by whichever session resolves
it first.
`HIPDNN_MATMUL_AUTOTUNE_LOG=1` logs the in-kernel tuner's decisions — every
candidate's timing, the winning config's full geometry, LUT hits, and the
cached selection — to stderr. Unlike `HIPDNN_MATMUL_LUT_LOG` (which only covers
the offline-table lookup that `online` mode skips), this shows what the sweep
actually picked, so an `online` run can be diffed against the shipped table.
`HIPDNN_EP_DEBUG=1` still enables these lines too (plus everything else).
