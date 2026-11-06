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
| `scripts/update_lut.py` | measure -> build -> compile; import accepted pinned-UT data |
| `tools/matmul_nbits_autotune_sweep.cpp` | GPU sweep driver |
| `tools/empty_lut_data.cpp` | zero-size payload, for builds that must not consult a table |

## The three tiers

Probed finest first; the first row that exists **and** whose geometry the live
config table still accepts wins.

| Tier | Key | Answers |
|---|---|---|
| **Exact** | phase, bits, `N`, `K`, group_size, zero-point, row-stride, M-bucket | measured MatMulNBits nodes in 20 shipped models, plus selected decode cells and full prefill ladders from the hand-added `Internal` inventory |
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

Incremental LUT refresh uses candidate-only fast ranking and direct import.
Kernel acceptance uses checked pinned-config UT confirmation on priority,
boundary and production-screen suspect shapes. The existing sweep commands
also retain the older autotune-log regeneration path.

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
    -I lib/Runtime/Kernels/include -I <flatbuffers include> -I <generated header dir> `
    tools/matmul_nbits_autotune_sweep.cpp tools/empty_lut_data.cpp matmul_nbits_autotune.cpp `
    ../../matmul_nbits_kernel.hip -o mn_sweep.exe

# 3. measure -> build -> compile. --arch is the chip the sweep runs on (the
#    table is per chip even though the binary is generic).
python scripts/update_lut.py all --arch gfx1151 --sweep mn_sweep.exe --flatc <flatc>
```

Each sweep process starts with an empty in-process tune map (the tuners no
longer persist to an on-disk cache), so every shape is measured fresh and none
is silently skipped.

The `measure`/`build` path reads the in-kernel autotuner's debug log. Those
timings do not provide the checked symmetric pinned-UT evidence required to
accept a kernel optimization or its LUT winners.

### Import accepted pinned-config UT measurements

Use the existing FP16/U4 UT to confirm selected configs on priority model keys,
class-boundary sentinels and production-screen suspects. Each selected config
needs CPU correctness and at least five accepted timing blocks in each of an
odd count of at least five processes per side. Use matched launch counts,
symmetric baseline/candidate order, the same cache/preheat protocol, and
medians of block medians across processes. Preserve raw clock checks, launch
identities, process ranges and a declared noise/resolution policy.

The importer below also consumes previously accepted complete pinned datasets.
Reuse such evidence when compatible; do not create a full-family formal
shape/config/source/process product to supply it. New full-family ranking
uses the fast incremental path in the next section.

After the complete raw audit and code acceptance decision, pass the accepted
packet to the existing builder:

```powershell
python scripts/update_lut.py import-ut --arch gfx1151 `
    --accepted-ut accepted-ut.json --report import-receipt.json
python scripts/update_lut.py compile --arch gfx1151 --flatc <flatc>
```

The packet binds `CANDIDATE_ACCEPTED`, the original JSON SHA-256, the complete
actual launch-family scope, all per-shape/config baseline and candidate block
arrays, and the accepted winner IDs. Its dataset digest is SHA-256 of canonical
UTF-8 JSON with sorted keys and compact separators; the dataset source is
`standard_pinned_ut`. Runtime config IDs stay paired with their full serialized
geometry. Raw auditing establishes that the readings and identities came from
the declared architecture, sources and executables before creating this packet.

`import-ut` performs no GPU work. It derives winners from that exact accepted
dataset and refuses missing/duplicate rows or configs, failed CPU checks,
invalid timing arrays, incompatible original table hashes, winner mismatches,
and regressions beyond each row's declared acceptance policy. It preserves
existing pool indices, unrelated points, fallbacks, fitted weights and
metadata, then adds requested priority exact points from the same readings.
The output replaces the JSON atomically after validation. `--out` can write a
review copy; `--report` records input/output hashes and the dataset digest.

Before delivery, verify schema/kernel-ABI compatibility, compile the FB, query
every original key with the existing `--requests`/`--dump` host tool, and check
that unrelated resolver outcomes remain unchanged. Verify priority exact
points and actual on-device LUT consumption, then run final affected leaves
at coverage 3 after the code and table are final. Production timing here
checks consumption; it does not select another set of winners.

### Fast incremental family refresh

For an existing family, rank the final candidate configs once with the ordinary
small warmup/median policy, then import that canonical dataset directly:

```powershell
python scripts/update_lut.py import-fast --arch gfx1151 `
    --winner-data fast-winners.json --report import-receipt.json
python scripts/update_lut.py compile --arch gfx1151 --flatc <flatc>
```

`import-fast` requires `FAST_RANKING_ACCEPTED`, the canonical dataset SHA-256,
current kernel/core source hashes, measured DLL/device identity, every required
row/config and its retained samples, actual launch mapping, original LUT hash
and accepted winner IDs. Fast ranking uses a 2% tie policy. An independently
checked pinned-UT correction may select another considered config when the
fast choice fails formal acceptance. Its `formal_confirmation` retains both
sides' five-process/five-block arrays, shape/config/source identity and raw
audit digests; the importer recomputes the 2% acceptance check. It never replaces
prepared-kernel fast samples with padded-wrapper timings. Record both values
and verify the corrected winner in the final production regression screen.
When displacement and correction come from separate paired checks, retain
`rejected_fast_baseline` and bind both baseline config IDs to the original
production config. Each ratio then uses its own matched baseline arrays.
For priority model exact points, `selection_reason=priority_checked_best` also
permits an already checked config when it is more than 2% faster than the
displaced fast choice and still passes baseline acceptance. This requires
`priority=true` and the same bound five-process/block arrays; an unchecked gain
or a tie cannot override fast ranking. Prefer the existing fast tie band first.
For an already confirmed loss, `--allow-pending-regression` can build a staging
table with an alternative from the same considered configs. The bound
`regression_selection` retains the displaced config's checked arrays and source
identity, and the import receipt stays `PENDING_PRODUCTION_CONFIRMATION` until
the alternative has its own checked confirmation. This option does not measure
or replace ranking samples. A bounded incomplete confirmation may instead
retain the original production config; byte-identical original ISA and launch
geometry also support retaining that config when timings vary. These retention
decisions require explicit source, key, config and raw-audit bindings.
It preserves existing pool indices, unrelated points, fallbacks and fitted
weights, and adds priority exact points that were explicitly reviewed. The
atomic import performs no GPU work and refuses a second import against the
already changed table.

The Windows sweep driver's `--requests` mode loads one-kernel libraries built
by the original fp16/U4 leaf's `lut_measure_library` target. Build its driver
with `lut_measure_driver`. Pass `--candidate-library <dll>` and
`--measurement-purpose rank`; ranking pins each requested config. The request
CSV header is `id,m,n,k,group_size,zero_point,row_stride,configs`, with numeric
group size, `Symmetric|Asymmetric`, `Arrival|Padded` and semicolon-separated
config IDs. The driver reports actual untimed launch names, warmup2/median3
event blocks, a rotated 256 MiB weight working set (max4096 copies), and
64 sampled CPU-reference output coordinates. These values rank configs;
independent pinned UT timing supplies formal acceptance.

After embedding the candidate LUT, pass both `--baseline-library <dll>` and
`--candidate-library <dll>` with `--measurement-purpose screen`. This alternates
old/new production-selected paths across the complete affected scope, including
raw-K input padding and per-call padded-row repacking. Each library exports its
CRT `fflush` so untimed launch audits are flushed before the driver's result;
timed loops keep audit output disabled. A quick loss above 2% schedules a formal pinned
repeat for that shape and the actual selected configs. Confirm all priority
and boundary shapes formally, reuse compatible completed evidence, and fix
confirmed losses at the class or winner level. Record screen and formal values
separately. A stratified pilot must project ranking, screen and the bounded
formal subset within eight GPU hours; checkpoint and improve the runner if it
does not. Do not schedule all shapes x configs x sources x five formal processes,
or repeat GPU winner selection for LUT generation. Final affected leaf coverage3,
complete host resolver diff and on-device consumption remain delivery gates.

### Measure the product binary

The gfx1151 arrival-layout K8-XOR route is restricted to fused WT4x4/BK32
tiles with at least 32 workgroups per CU and 64 K stages. Smaller warp
classes, shorter K loops and padded weight rows retain LDS padding. A route
change invalidates only the affected config cells; retain other measurements
when device payload, resources and launch geometry still match.

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
