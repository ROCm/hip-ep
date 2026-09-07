# GQA offline autotune LUTs

`gfx1151.json` is the reviewable source; `gfx1151.fb` is the FlatBuffer the DLL
embeds (see `../README.md`, "Where it lives"). Points are generated from
measurement, never edited by hand.

The full pipeline is `../scripts/update_lut.py`:

```bash
SCRIPTS=lib/Runtime/Kernels/hip/autotune/gqa/scripts

# 1. Enumerate the shapes to measure (writes lut/shapes_gfx1151.json, config=TBD).
python $SCRIPTS/update_lut.py plan

# 2. Sweep them on the GPU (writes scripts/data/*_best.csv).
#    GPU only -- run it when the card is idle, one sweep at a time.
python $SCRIPTS/update_lut.py measure --sweep <build>/.../gqa_autotune_sweep.exe

# 3. Build the JSON from the winners: dedupe configs, one point per shape.
python $SCRIPTS/update_lut.py build

# 4. Compile to the FlatBuffer the DLL embeds.
python $SCRIPTS/update_lut.py compile --flatc flatc
```

`build` reads `data/*_best.csv` (each row is a measured shape plus its winning
`cfg_*` columns), collects the distinct configs into `configs[]`, and emits one
`points[]` entry per shape with a 1-byte `config` index and literal numeric dims
(no bucketing — the nearest-neighbour metric uses the real distance), plus the
fixed per-`(phase, head_dim)` `fallbacks[]`. The five `weight_*` fields default
to 1.0.

It then **prunes saturated points**: within each exact-geometry group it drops a
point whenever every measured coordinate in the group still resolves to the same
`(config, splits)` without it — i.e. a run of lengths that share a winner (a
decode split count that has saturated, or a prefill config that barely depends on
`seq_kv`) collapses to its endpoints, while every boundary where the answer
changes is kept. This is the nearest-neighbour analogue of the old tier table's
bucketing; it is answer-preserving on every measured shape (verified by replay in
`build`) and roughly halves the point count and the `.fb`.

**`compile` is just `flatc`** against the schema one level up:

```bash
flatc --binary --strict-json -o /tmp ../gqa_autotune.fbs gfx1151.json
cp /tmp/gfx1151.bin gfx1151.fb
```

`flatc` validates field names and enum spellings and nothing else — a point whose
config kind does not match its phase, or whose categorical key is `Any`, is not a
`flatc` error. Those are dropped at *load* by `pointConsistent()`, and silent
coverage loss, so `test-gqa-autotune` loads this `.fb` and asserts
`hip_gqa_autotune_invalid_points() == 0`. Run that test after regenerating.

**Do not use `update_lut.py --rdpcapture`.** That path delegates to RdpCapture's
`build_lut.py`, which still emits the old tier schema; this loader rejects it. The
standalone best-csv path above is the supported one.

## Compatibility gates

`compatible()` in `../gqa_autotune.cpp` rejects the whole table on any of these; a
mismatch is a loud rejection (fall to heuristic) rather than a silently wrong pick:

- **GPU architecture** — must equal the running device, e.g. `gfx1151`.
- **LUT schema version** — **9** (the nearest-neighbour layout: `configs[]` /
  `points[]` / `fallbacks[]`, `file_identifier "GQAL"`, 16-byte points). Not
  compatible with the old tier files (they carried buckets and per-field
  wildcards this loader has no field for).
- **GQA kernel ABI** — `gqa-v3`. Bumped whenever a config knob's meaning changes,
  so a stale `.fb` swept against different kernels is rejected, not misread.
- **ROCm/HIP version** — advisory: a mismatch warns but still loads.
- **KV-cache dtype** — points are `Fp16` today; the schema reserves `Int8` for the
  W4A8 decode path, so adding it is a regeneration, not a schema break.

## What the table covers

The fused decode path admits **heads-per-group ∈ {1,2,3,4,5,8,16}** at
**head_dim ∈ {64,128,256}** (`flash_decode_geometry_ok` in `real/gqa.cpp`). The
sweep measures the production model geometries plus, for each, the multi-query
floor (`G = 1`) so the same-hpg fuzzy search always lands on something measured.
Prefill kernels are templated on `head_dim` alone, so a prefill config is nearly
geometry- and KV-length-independent and needs far fewer points.

The current `gfx1151` table is schema 9 / `gqa-v3`: 18 configs, 1614 points and 6
per-`(phase, head_dim)` fallbacks, over the model geometries at lengths from 128
to 65536, batch 1. The points are the saturation-pruned survivors of ~3200
measured shapes (the rest were reconstructable by nearest-neighbour and dropped).
A sliding window is folded into the scanned length (`effective_len`) at query
time, so the schema has no window field and the table needs no separate windowed
rows. Every production geometry resolves `Exact` or short-distance `Nearest`; an
unmeasured length resolves to its nearest measured neighbour rather than the
heuristic.

`../scripts/update_lut.py plan` prints the exact shape list, and `GEOMETRIES` in
that script is the source of truth for which geometries are swept. To widen
coverage, add a geometry there and re-run the pipeline; nothing already measured is
invalidated.

## Packaging

The `.fb` is embedded into `custom_kernels_<arch>` at build time by
`lib/Runtime/Kernels/CMakeLists.txt` (pure-CMake `file(READ ... HEX)`), so there is
no separate packaging step and no runtime file path. The schema
(`../gqa_autotune.fbs`) and the loader/resolver (`../gqa_autotune.cpp`) sit one
level up; see `../README.md` for how a config is resolved.
